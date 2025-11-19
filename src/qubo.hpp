#pragma once

#include "gpu.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <memory>
#include <span>
#include <utility>
#include <vector>

template <uint32_t NUM, uint32_t POW>
__host__ __device__ consteval uint32_t asPowOfTwo() {
    if constexpr ((NUM >> POW) > 1) {
        return asPowOfTwo<NUM, POW + 1>();
    }

    return POW;
}

// A very simple stack allocator that just aligns the current pointer correctly
// and moves it forward after each allocation, returning the aligned pointer.
// Used with shared memory
struct Allocator {
  private:
    // These never move
    uint8_t *const begin = nullptr;
    uint8_t *const end = nullptr;
    // This is moved as new allocation happen
    uint8_t *current = nullptr;

  public:
    __host__ __device__ Allocator(void *ptr, size_t capacity)
        : begin(static_cast<uint8_t *>(ptr)), end(begin + capacity),
          current(begin) {}

    template <typename T> __host__ __device__ T *allocate(size_t len) {
        static constexpr size_t alof = alignof(T);
        const auto misalignment =
            (alof - (reinterpret_cast<std::intptr_t>(current) & (alof - 1))) &
            (alof - 1);

        const size_t num_bytes = len * sizeof(T);
        uint8_t *ptr = current + misalignment;

        assert(ptr + num_bytes <= end);

        current = ptr + num_bytes;

        return static_cast<T *>(static_cast<void *>(ptr));
    }

    size_t allocated_bytes() const { return current - begin; }
};

// This only supports square matrices
template <typename T> struct SparseView {
    // TODO reduce size by using pointers, not spans?
  private:
    std::span<uint32_t> indptr;
    std::span<uint32_t> indices;
    std::span<T> data;

  public:
    std::span<T> diagonal;

    SparseView() {}
    SparseView(std::span<uint32_t> ip, std::span<uint32_t> in, std::span<T> da,
               std::span<T> di)
        : indptr(ip), indices(in), data(da), diagonal(di) {}

    __host__ __device__ std::span<uint32_t> rowIndices(uint32_t r) const {
        assert(r + 1 < indptr.size());
        const auto begin = indptr[r];
        const auto end = indptr[r + 1];
        return std::span<uint32_t>(&indices[begin], end - begin);
    }

    __host__ __device__ std::span<T> rowData(uint32_t r) const {
        assert(r + 1 < indptr.size());
        const auto begin = indptr[r];
        const auto end = indptr[r + 1];
        return std::span<T>(&data[begin], end - begin);
    }

    __host__ __device__ std::pair<uint32_t, uint32_t> shape() const {
        return std::make_pair(indptr.size() - 1, indptr.size() - 1);
    }

    __host__ std::vector<T> toDense() const {
        const auto n = shape().first;
        std::vector<uint32_t> indptr_h(indptr.size(), 0);
        std::vector<uint32_t> indices_h(indices.size(), 0);
        std::vector<T> data_h(data.size(), 0);

        gpu::memcpy(indptr_h.data(), indptr.data(), indptr.size_bytes());
        gpu::memcpy(indices_h.data(), indices.data(), indices.size_bytes());
        gpu::memcpy(data_h.data(), data.data(), data.size_bytes());

        std::vector<T> dense(n * n, 0);
        for (auto i = 0; i < n; i++) {
            const auto begin = indptr_h[i];
            const auto end = indptr_h[i + 1];
            const auto length = end - begin;
            for (auto j = 0; j < length; j++) {
                const auto index = i * n + indices_h[begin + j];
                dense[index] = data_h[begin + j];
            }
        }

        return dense;
    }
};

// This only supports square matrices
template <typename T, typename Mem> struct SparseMatrix {
    static size_t numBytesReq(size_t ndata, size_t nrows) {
        // Add one extra for alignment and nrows for the diagonal
        auto required = sizeof(T) * (ndata + 1 + nrows);

        // alignment required for the indices
        using I = uint32_t;
        auto misalign = required & (sizeof(I) - 1);
        required += sizeof(I) - misalign;
        // mem req of indices: ndata indices and nrows + 1 meta indices
        required += sizeof(I) * (ndata + nrows + 1);

        return required;
    }

  private:
    size_t num_bytes = 0ull;
    std::unique_ptr<uint8_t, decltype(&Mem::free)> memory;
    SparseView<T> view;

  public:
    SparseMatrix(const std::vector<uint32_t> &indptr,
                 const std::vector<uint32_t> &indices,
                 const std::vector<T> &data)
        : num_bytes(numBytesReq(data.size(), indptr.size() - 1)),
          memory(static_cast<uint8_t *>(Mem::allocate(num_bytes)), Mem::free) {
        std::vector<T> diagonal(indptr.size() - 1, 0);
        for (auto r = 0; r < indptr.size() - 1; r++) {
            const auto begin = indptr[r];
            const auto end = indptr[r + 1];
            for (auto i = 0; i < end - begin; i++) {
                const auto c = indices[begin + i];
                if (c == r) {
                    diagonal[r] = data[begin + i];
                    break;
                }
            }
        }

        void *ptr = memory.get();
        size_t space = num_bytes;

        auto setSpan = [&ptr, &space](const auto &vec) {
            using V = std::remove_cvref_t<decltype(vec[0])>;
            if (std::align(alignof(V), vec.size() * sizeof(V), ptr, space)) {
                auto span = std::span<V>(static_cast<V *>(ptr), vec.size());
                uint32_t bytes = span.size_bytes();
                Mem::memcpy(span.data(), vec.data(), bytes);
                ptr = static_cast<uint8_t *>(ptr) + bytes;
                space -= bytes;

                return span;
            }
            return std::span<V>(static_cast<V *>(ptr), 0);
        };

        auto span_data = setSpan(data);
        auto span_diagonal = setSpan(diagonal);
        auto span_indices = setSpan(indices);
        auto span_indptr = setSpan(indptr);

        assert(span_data.size() == data.size());
        assert(span_diagonal.size() == diagonal.size());
        assert(span_indices.size() == indices.size());
        assert(span_indptr.size() == indptr.size());

        view = SparseView(span_indptr, span_indices, span_data, span_diagonal);
    }

    SparseView<T> getView() const { return view; }
};

template <typename T> struct SuperSpan {
    // This structure has a span to some data that is used by the entire grid
    // Each block has a unique sub span  within the super span
    // They are aligned by ALIGN in the super span and each is the same length
  private:
    // Align to 256 byte boundary
    static constexpr size_t ALIGN = 256ull / sizeof(T);
    std::span<T> data;
    uint32_t num_subspans = 0;
    uint32_t subspan_len = 0;

    __host__ __device__ static uint32_t width(uint32_t num_subspans,
                                              uint32_t subspan_len) {
        // Compute the width of the subspan
        // If subspan_len is aligned to ALIGN, i.e. remainder == 0
        // the width is equal to subspan_len
        const uint32_t remainder = subspan_len & (ALIGN - 1);
        const uint32_t padding = (ALIGN - remainder) & (ALIGN - 1);
        return subspan_len + padding;
    }

  public:
    static uint32_t requiredLength(uint32_t num_subspans,
                                   uint32_t subspan_len) {
        return num_subspans * width(num_subspans, subspan_len);
    }

    static uint32_t memReq(uint32_t num_subspans, uint32_t subspan_len) {
        return requiredLength(num_subspans, subspan_len) * sizeof(T);
    }

    SuperSpan(std::span<T> d, uint32_t num, uint32_t len)
        : data(d), num_subspans(num), subspan_len(len) {
        static_assert(ALIGN > 0, "ALIGN must be greater than zero");
        static_assert((ALIGN & (ALIGN - 1)) == 0,
                      "ALIGN must be a power of two");
        assert(d.size() == requiredLength(num_subspans, subspan_len));
    }

    __host__ __device__ std::span<T> subspan(uint32_t i) const {
        assert(i < num_subspans);
        // Skip all the i widths before this subspan
        const uint32_t begin = i * width(num_subspans, subspan_len);

        // Are we inside the super span?
        assert(begin + subspan_len < data.size());

        return data.subspan(begin, subspan_len);
    }

    __host__ __device__ uint32_t subspanLength() const { return subspan_len; }
};

struct BitVector {
    // This structure is a bit vector, where 32 bits are stored as a single
    // value. We're using 4 byte values to avoid bank conflicts in shared memory
    // when multiple threads want to access a bit from the same value: with 4
    // bytes per value, all (potentially 32) threads access the same (starting)
    // byte within the 4 byte bank, so the value can be broadcast efficiently.
    using S = uint32_t;

  private:
    S *ptr = nullptr;
    static constexpr uint32_t BITS_PER_TYPE = 8u * sizeof(S);
    static constexpr uint32_t POW_OF_TWO = asPowOfTwo<BITS_PER_TYPE, 0>();

  public:
    // Target for copy
    __host__ __device__ BitVector() {}

    __host__ __device__ BitVector(S *p) : ptr(p) {}

    static uint32_t requiredLength(uint32_t len) {
        return (len >> POW_OF_TWO) + 1;
    }

    static size_t memReq(uint32_t len) {
        return static_cast<size_t>(requiredLength(len)) * sizeof(S);
    }

    static uint32_t dataIdx(uint32_t i) { return i >> POW_OF_TWO; }
    // N.B! Conceputally row vectors increase in index from left to right,
    // while e.g. 32bit numbers have the lsb on the right and msb on the
    // left. To think of a series of 32bit numbers as a bit vector, we
    // should reverse the order of bits for every value stored, such that
    // when i == 0, we're indexing the first element of the vector, which is
    // then the msb of the first stored value
    // 'Reverse' the order of bits, i.e. bit_idx 0 is MSB, while bit_idx 31
    // is LSB
    static uint32_t bitIdx(uint32_t i) {
        return BITS_PER_TYPE - 1 - (i & (BITS_PER_TYPE - 1));
    }

    template <typename T = uint32_t>
    __host__ __device__ T operator[](uint32_t i) const {
        // We should return the bit at index i
        // We'll unpack the correct value in data and return the correct bit
        // from that value.
        return static_cast<T>((ptr[dataIdx(i)] >> bitIdx(i)) & 1);
    }

    __host__ __device__ void flip(uint32_t i) {
        // We're flipping the bit at index i
        ptr[dataIdx(i)] ^= (1 << bitIdx(i));
    }

    // Return the underlying ptr for efficient copies/inserts of 4 byte values
    __host__ __device__ S *data() const { return ptr; }
};

// DELETE BEGIN
__host__ __device__ void *alignPtr(void *ptr, size_t alof) {
    const auto misalignment =
        (alof - (reinterpret_cast<std::intptr_t>(ptr) & (alof - 1))) &
        (alof - 1);

    return static_cast<uint8_t *>(ptr) + misalignment;
};

template <typename T> struct Scratch {
    T *values = nullptr;
    uint32_t *indices = nullptr;
    uint32_t len = 0ul;

    // Target for copy
    __host__ __device__ Scratch() {}

    __host__ __device__ Scratch(void *ptr, uint32_t num_values)
        : values(static_cast<T *>(alignPtr(ptr, alignof(T)))),
          indices(static_cast<uint32_t *>(
              alignPtr(values + num_values, alignof(uint32_t)))),
          len(num_values) {}
};

template <typename T>
__host__ __device__ std::pair<std::span<T>, std::span<uint8_t>>
makeAlignedSpan(std::span<uint8_t> data, uint32_t len) {
    assert(data.size_bytes() >= sizeof(T) * len);

    auto ptr = alignPtr(data.data(), alignof(T));
    auto aligned = std::span<T>(static_cast<T *>(ptr), len);

    // The pointer at the end of the aligned data
    ptr = static_cast<void *>(aligned.data() + aligned.size());
    uint8_t *bptr = static_cast<uint8_t *>(ptr);
    const auto bytes_remaining = data.size_bytes() - (bptr - data.data());

    return std::make_pair(aligned, std::span<uint8_t>(bptr, bytes_remaining));
}

template <typename T>
__host__ __device__ std::pair<std::array<void *, 3>, std::span<uint8_t>>
structPointers(std::span<uint8_t> span) {
    auto ptr = span.data();
    auto incrPtr = [&ptr, &span]<typename V>() {
        ptr = static_cast<uint8_t *>(alignPtr(ptr, alignof(V)));
        const auto aligned = static_cast<void *>(ptr);
        ptr += sizeof(V);
        return aligned;
    };

    return std::make_pair(
        std::array{
            incrPtr.template operator()<BitVector>(),
            incrPtr.template operator()<BitVector>(),
            incrPtr.template operator()<Scratch<T>>(),
        },
        std::span<uint8_t>(ptr, span.size_bytes() - (ptr - span.data())));
}

template <typename T>
__host__ __device__ void constructStructs(std::array<void *, 3> ptrs,
                                          std::span<uint8_t> remainder,
                                          uint32_t num_cols, uint32_t len_x) {
    using S = typename BitVector::S;

    std::span<S> aligned;
    std::tie(aligned, remainder) = makeAlignedSpan<S>(remainder, len_x);

    *static_cast<BitVector *>(ptrs[0]) = BitVector(aligned.data());

    std::tie(aligned, remainder) = makeAlignedSpan<S>(remainder, len_x);
    *static_cast<BitVector *>(ptrs[1]) = BitVector(aligned.data());

    *static_cast<Scratch<T> *>(ptrs[2]) =
        Scratch<T>(remainder.data(), gpu::nwarps());
}

// DELETE END

// Matrix-vector product with a sparse matrix m and a bitvector x
// Result is stored in y
// Implemented per-block, i.e. one block does the computation for the entire
// product. This should be called with unique x & y if called with multiple
// blocks
template <typename T>
__host__ __device__ void mx(const SparseView<T> &m, BitVector &x,
                            std::span<T> y) {
    const auto shape = m.shape();
    const auto num_rows = shape.first;
    const auto num_cols = shape.second;
    assert(y.size() == num_rows);

    const uint32_t wid = gpu::warpid();
    const uint32_t wstride = gpu::nwarps();
    const uint32_t lane = gpu::laneid();
    const uint32_t lanestride =
        gpu::WARP_SIZE < gpu::nthreads() ? gpu::WARP_SIZE : gpu::nthreads();

    for (auto row = wid; row < num_rows; row += wstride) {
        const auto row_indices = m.rowIndices(row);
        const auto row_data = m.rowData(row);
        auto sum = static_cast<T>(0);
        for (auto i = lane; i < row_indices.size(); i += lanestride) {
            // j is the column of the matrix
            const auto j = row_indices[i];
            sum += row_data[i] * x[j];
        }

        // Only thread 0 gets the correct value
        sum = gpu::warpReduce(sum, [](auto a, auto b) { return a + b; });
        if (lane == 0) {
            y[row] = sum;
        }
    }

    // Warps work independently, but entire block may use the result
    // sync all threads, so all threads see the results
    gpu::syncthreads();
}

__host__ __device__ void generateRandomX(BitVector &x, size_t n) {
    // TODO use hiprand, here we're just assing around
    auto span = x.data();
    for (uint32_t i = gpu::threadid(); i < n; i += gpu::nthreads()) {
        span[i] = i;
    }
}

// Perform a dot product between x and y
// scratch should be memory which all the threads of a block can access.
// When called with multiple blocks, data and scratch must be unique per
// block.
template <typename T>
__host__ __device__ T blockDotProd(BitVector &x, std::span<T> y, T *scratch) {
    T result = 0.0f;
    for (size_t i = gpu::threadid(); i < y.size(); i += gpu::nthreads()) {
        result += x[i] * y[i];
    }

    const uint32_t wid = gpu::warpid();
    const uint32_t lane = gpu::laneid();
    auto sum = [](auto a, auto b) { return a + b; };

    result = gpu::warpReduce(result, sum);
    if (0 == lane) {
        scratch[wid] = result;
    }

    // Wait until all warps in block are done
    gpu::syncthreads();

    // Perform the final reduction by the first warp
    if (wid == 0) {
        result = lane < gpu::nwarps() ? scratch[lane] : 0.0f;
        result = gpu::warpReduce(result, sum);

        // First thread stores the result in memory accessible to the entire
        // block
        if (0 == lane) {
            scratch[0] = result;
        }
    }

    // Others wait until first warp is done
    gpu::syncthreads();

    result = scratch[0];

    // Everyone waits until everyone has read the value
    gpu::syncthreads();

    return result;
}

template <typename T>
__host__ __device__ int32_t minimumRow(const SparseView<T> &m, BitVector &x,
                                       std::span<T> y, Scratch<T> &scratch) {
    const auto diagonal = m.diagonal;
    T minimum = std::numeric_limits<T>::max();
    int32_t minrow = -1;

    auto updatedRowValue = [&x, &y, &diagonal](auto i) {
        static constexpr T one = static_cast<T>(1.0);
        static constexpr T two = static_cast<T>(2.0);

        // If current x is zero, flipping it to 1 adds to the total, so positive
        // sign Conversely for x == 1
        const auto sign = -two * x.operator[]<T>(i) + one;

        // y may or may not contain the diagonal
        // 2*y contains the diagonal either twice or zero times
        // subtracting or adding the diagonal to y once will leave 2*y with
        // the diagonal exactly once
        const auto value = sign * (two * y[i] + sign * diagonal[i]);

        return value;
    };

    // This updates the minimum internally and returns the index corresponding
    // to that minimum
    auto argmin = [](auto a, auto b, auto i, auto j) {
        auto interpolate = [](auto s, auto a, auto b) {
            return s * a + static_cast<decltype(a)>(1 - s) * b;
        };

        // Avoid 'if' within warp by doing a linear interpolation between
        // the two endpoints with s either 0 or 1
        const auto s = static_cast<T>(a < b);
        return std::make_pair(interpolate(s, a, b), interpolate(s, i, j));
    };

    // Find the minimum for each thread among the values in y
    for (size_t i = gpu::threadid(); i < y.size(); i += gpu::nthreads()) {
        std::tie(minimum, minrow) =
            argmin(updatedRowValue(i), minimum, i, minrow);
    }

    // Do a warp reduction of the argmins
    std::tie(minimum, minrow) = gpu::warpArgSearch(minimum, minrow, argmin);
    const uint32_t lane = gpu::laneid();
    const uint32_t wid = gpu::warpid();
    if (0 == lane) {
        scratch.values[wid] = minimum;
        scratch.indices[wid] = minrow;
    }

    // Wait until ever warp has updated scratch with their minimums
    gpu::syncthreads();

    // The first warp should do the final reduction
    if (0 == wid) {
        minimum = lane < gpu::nwarps() ? scratch.values[lane]
                                       : static_cast<T>(INFINITY);
        minrow = lane < gpu::nwarps() ? scratch.indices[lane] : 0;
        std::tie(minimum, minrow) = gpu::warpArgSearch(minimum, minrow, argmin);

        // Lane 0 of warp 0 has the block minimum and the index corresponding to
        // that value. If the minimum is negative, store the row index as
        // positive. The sign of the number is used as a continue/break
        // condition in a while loop.
        if (0 == lane) {
            scratch.indices[0] = (2 * (minimum < 0) - 1) * minrow;
        }
    }

    gpu::syncthreads();

    minrow = scratch.indices[0];

    gpu::syncthreads();

    return minrow;
}

template <typename T>
__host__ __device__ void updateXY(const SparseView<T> &m, BitVector &x,
                                  std::span<T> y, uint32_t row) {
    auto data = m.rowData(row);
    auto indices = m.rowIndices(row);
    const T xi = x.operator[]<T>(row);

    for (uint32_t tid = gpu::threadid(); tid < data.size();
         tid += gpu::nthreads()) {
        const auto col = indices[tid];
        // If xi is one, we'll subtract the data, as xi will be changed to zero
        y[col] += data[tid] * (-2.0f * xi + 1);
    }

    // Don't flip the bit until every warp is ready
    gpu::syncthreads();

    if (0 == gpu::threadid()) {
        x.flip(row);
    }

    gpu::syncthreads();
}

template <typename T>
__host__ __device__ T blockSearch(const SparseView<T> &m, BitVector &x,
                                  std::span<T> y, Scratch<T> &scratch) {
    generateRandomX(x, m.shape().first);
    mx(m, x, y);

    auto row = minimumRow(m, x, y, scratch);
    while (row >= 0) {
        updateXY(m, x, y, row);
        row = minimumRow(m, x, y, scratch);
    }

    return blockDotProd(x, y, scratch.values);
}

// TODO: test everything below this
template <typename T>
__host__ __device__ void search(const SparseView<T> &m, std::span<T> block_y,
                                SuperSpan<uint32_t> min_x, std::span<T> min_y,
                                std::span<uint8_t> mem,
                                uint32_t num_values_to_search) {
    auto [ptrs, remainder] = structPointers<T>(mem);

    if (gpu::threadid() == 0) {
        const auto num_cols = m.shape().second;
        const auto len_x = min_x.subspanLength();
        constructStructs<T>(ptrs, remainder, num_cols, len_x);
    }
    gpu::syncthreads();

    auto &block_x = *static_cast<BitVector *>(ptrs[0]);
    auto &block_min_x = *static_cast<BitVector *>(ptrs[1]);
    auto &scratch = *static_cast<Scratch<T> *>(ptrs[2]);

    auto minimum_y = static_cast<T>(INFINITY);
    for (uint32_t bid = gpu::blockid(); bid < num_values_to_search;
         bid += gpu::nblocks()) {
        auto local_minimum = blockSearch(m, block_x, block_y, scratch);
        if (gpu::threadid() == 0 && local_minimum < minimum_y) {
            minimum_y = local_minimum;
            // TODO: maybe it's sufficient to store the index with which the
            // random number was generated then there's no need for the
            // block_min_x bitvector
            std::swap(block_x, block_min_x);
        }
        gpu::syncthreads();
    }

    // Copy the minimum found by the block from shared memory to global memory
    auto bmx = min_x.subspan(gpu::blockid());
    auto bmx_shared = block_min_x.data();
    for (uint32_t tid = gpu::threadid(); tid < bmx.size();
         tid += gpu::nthreads()) {
        bmx[tid] = bmx_shared[tid];
    }

    // Update the minimum_y found by the block from register to global memory
    if (0 == gpu::threadid()) {
        min_y[gpu::blockid()] = minimum_y;
    }
}

template <typename T>
__global__ void searchKernel(const SparseView<T> &m, SuperSpan<T> y,
                             SuperSpan<uint32_t> min_x, std::span<T> min_y,
                             size_t shared_mem_bytes,
                             uint32_t num_values_to_search) {
    extern __shared__ uint8_t shared_mem[];
    // Use y from global memory
    auto block_y = y.subspan(gpu::blockid());
    search(m, block_y, min_x, min_y,
           std::span<uint8_t>(shared_mem, shared_mem_bytes),
           num_values_to_search);
}

template <typename T>
__global__ void searchKernel(const SparseView<T> &m, SuperSpan<uint32_t> min_x,
                             std::span<T> min_y, size_t shared_mem_bytes,
                             uint32_t num_values_to_search) {
    extern __shared__ uint8_t shared_mem[];

    // Construct y in shared memory
    auto [block_y, remainder] = makeAlignedSpan<T>(
        std::span<uint8_t>(shared_mem, shared_mem_bytes), m.shape().second);

    search(m, block_y, min_x, min_y, remainder, num_values_to_search);
}

template <typename T> __host__ size_t numBytesOfSharedRequired() {}
