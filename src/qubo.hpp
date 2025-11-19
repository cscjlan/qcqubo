#pragma once

#include "gpu.hpp"

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

    constexpr static uint32_t requiredLength(uint32_t len) {
        return (len >> POW_OF_TWO) + 1;
    }

    constexpr static size_t memReq(uint32_t len) {
        return static_cast<size_t>(requiredLength(len)) * sizeof(S);
    }

    constexpr static uint32_t dataIdx(uint32_t i) { return i >> POW_OF_TWO; }
    // N.B! Conceputally row vectors increase in index from left to right,
    // while e.g. 32bit numbers have the lsb on the right and msb on the
    // left. To think of a series of 32bit numbers as a bit vector, we
    // should reverse the order of bits for every value stored, such that
    // when i == 0, we're indexing the first element of the vector, which is
    // then the msb of the first stored value
    // 'Reverse' the order of bits, i.e. bit_idx 0 is MSB, while bit_idx 31
    // is LSB
    constexpr static uint32_t bitIdx(uint32_t i) {
        return BITS_PER_TYPE - 1 - (i & (BITS_PER_TYPE - 1));
    }

    __host__ __device__ uint32_t operator[](uint32_t i) const {
        // We should return the bit at index i
        // We'll unpack the correct value in data and return the correct bit
        // from that value.
        return (ptr[dataIdx(i)] >> bitIdx(i)) & 1;
    }

    __host__ __device__ void flip(uint32_t i) {
        // We're flipping the bit at index i
        ptr[dataIdx(i)] ^= (1 << bitIdx(i));
    }

    // Return the underlying ptr for efficient copies/inserts of 4 byte values
    __host__ __device__ S *data() const { return ptr; }
};

template <typename T> struct SearchData {
    using ThisType = SearchData<T>;

    BitVector candidate = {};
    BitVector best_candidate = {};
    T *rows = nullptr;
    T *scratch_values = nullptr;
    uint32_t *scratch_indices = nullptr;
    size_t len_data = 0;
    size_t len_scratch = 0;

    static size_t memReq(size_t len_data, size_t len_scratch,
                         bool with_rows = true) {
        ThisType temp;

        // Make an allocator just for computing the size
        // The starting pointer is aligned for this type, so we should add
        // alignment requirement later
        Allocator allocator(&temp, std::numeric_limits<std::intptr_t>::max());

        auto p0 = allocator.allocate<ThisType>(1);
        auto p1 =
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data));
        auto p2 =
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data));
        if (with_rows) {
            auto p3 = allocator.allocate<T>(len_data);
        }
        auto p4 = allocator.allocate<T>(len_scratch);
        auto p5 = allocator.allocate<uint32_t>(len_scratch);

        // Return the alignment requirement of this type + how many bytes are
        // required for the data
        return alignof(ThisType) + allocator.allocated_bytes();
    }

    __host__ SearchData() {}

    __host__ __device__ SearchData(Allocator &allocator, size_t len_d,
                                   size_t len_s)
        : len_data(len_d), len_scratch(len_s) {
        candidate = BitVector(
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data)));
        best_candidate = BitVector(
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data)));
        rows = allocator.allocate<T>(len_data);
        scratch_values = allocator.allocate<T>(len_scratch);
        scratch_indices = allocator.allocate<uint32_t>(len_scratch);
    }

    __host__ __device__ SearchData(Allocator &allocator, size_t len_d,
                                   size_t len_s, T *rs)
        : len_data(len_d), len_scratch(len_s) {
        candidate = BitVector(
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data)));
        best_candidate = BitVector(
            allocator.allocate<uint32_t>(BitVector::requiredLength(len_data)));
        rows = rs;
        scratch_values = allocator.allocate<T>(len_scratch);
        scratch_indices = allocator.allocate<uint32_t>(len_scratch);
    }

    __host__ __device__ void swapCandidates() {
        std::swap(candidate, best_candidate);
    }
};

// Matrix-vector product with a sparse matrix m and a bitvector candidate
// Result is stored in rows
// Implemented per-block, i.e. one block does the computation for the entire
// product. This should be called with unique candidate & rows if called with
// multiple blocks
template <typename T>
__host__ __device__ void mx(const SparseView<T> &m, SearchData<T> &sd) {
    const auto shape = m.shape();
    const auto num_rows = shape.first;
    const auto num_cols = shape.second;

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
            sum += row_data[i] * sd.candidate[j];
        }

        // Only thread 0 gets the correct value
        sum = gpu::warpReduce(sum, [](auto a, auto b) { return a + b; });
        if (lane == 0) {
            sd.rows[row] = sum;
        }
    }

    // Warps work independently, but entire block may use the result
    // sync all threads, so all threads see the results
    gpu::syncthreads();
}

template <typename T>
__host__ __device__ void generateRandomX(SearchData<T> &sd) {
    // TODO use hiprand, here we're just assing around
    auto data = sd.candidate.data();
    for (uint32_t i = gpu::threadid();
         i < BitVector::requiredLength(sd.len_data); i += gpu::nthreads()) {
        data[i] = i;
    }
}

// Perform a dot product between candidate and rows
// scratch should be memory which all the threads of a block can access.
// When called with multiple blocks, data and scratch must be unique per
// block.
template <typename T> __host__ __device__ T blockDotProd(SearchData<T> &sd) {
    T result = 0.0f;
    for (size_t i = gpu::threadid(); i < sd.len_data; i += gpu::nthreads()) {
        result += sd.candidate[i] * sd.rows[i];
    }

    const uint32_t wid = gpu::warpid();
    const uint32_t lane = gpu::laneid();
    auto sum = [](auto a, auto b) { return a + b; };

    result = gpu::warpReduce(result, sum);
    if (0 == lane) {
        sd.scratch_values[wid] = result;
    }

    // Wait until all warps in block are done
    gpu::syncthreads();

    // Perform the final reduction by the first warp
    if (wid == 0) {
        result = lane < gpu::nwarps() ? sd.scratch_values[lane] : 0.0f;
        result = gpu::warpReduce(result, sum);

        // First thread stores the result in memory accessible to the entire
        // block
        if (0 == lane) {
            sd.scratch_values[0] = result;
        }
    }

    // Others wait until first warp is done
    gpu::syncthreads();

    result = sd.scratch_values[0];

    // Everyone waits until everyone has read the value
    gpu::syncthreads();

    return result;
}

template <typename T>
__host__ __device__ int32_t minimumRow(const SparseView<T> &m,
                                       SearchData<T> &sd) {
    const auto diagonal = m.diagonal;
    const auto n = m.shape().first;
    T minimum = std::numeric_limits<T>::max();
    int32_t minrow = -1;

    auto updatedRowValue = [&sd, &diagonal](auto i) {
        static constexpr T one = static_cast<T>(1.0);
        static constexpr T two = static_cast<T>(2.0);

        // If current candidate is zero, flipping it to 1 adds to the total, so
        // positive sign Conversely for candidate == 1
        const auto sign = -two * sd.candidate[i] + one;

        // rows may or may not contain the diagonal
        // 2*rows contains the diagonal either twice or zero times
        // subtracting or adding the diagonal to rows once will leave 2*rows
        // with the diagonal exactly once
        const auto value = sign * (two * sd.rows[i] + sign * diagonal[i]);

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

    // Find the minimum for each thread among the values in rows
    for (size_t i = gpu::threadid(); i < n; i += gpu::nthreads()) {
        std::tie(minimum, minrow) =
            argmin(updatedRowValue(i), minimum, i, minrow);
    }

    // Do a warp reduction of the argmins
    std::tie(minimum, minrow) = gpu::warpArgSearch(minimum, minrow, argmin);
    const uint32_t lane = gpu::laneid();
    const uint32_t wid = gpu::warpid();
    if (0 == lane) {
        sd.scratch_values[wid] = minimum;
        sd.scratch_indices[wid] = minrow;
    }

    // Wait until ever warp has updated scratch with their minimums
    gpu::syncthreads();

    // The first warp should do the final reduction
    if (0 == wid) {
        minimum = lane < gpu::nwarps() ? sd.scratch_values[lane]
                                       : static_cast<T>(INFINITY);
        minrow = lane < gpu::nwarps() ? sd.scratch_indices[lane] : 0;
        std::tie(minimum, minrow) = gpu::warpArgSearch(minimum, minrow, argmin);

        // Lane 0 of warp 0 has the block minimum and the index corresponding to
        // that value. If the minimum is negative, store the row index as
        // positive. The sign of the number is used as a continue/break
        // condition in a while loop.
        if (0 == lane) {
            sd.scratch_indices[0] = (2 * (minimum < 0) - 1) * minrow;
        }
    }

    gpu::syncthreads();

    minrow = sd.scratch_indices[0];

    gpu::syncthreads();

    return minrow;
}

template <typename T>
__host__ __device__ void updateXY(const SparseView<T> &m, SearchData<T> &sd,
                                  uint32_t row) {
    auto data = m.rowData(row);
    auto indices = m.rowIndices(row);
    const T bit = sd.candidate[row];

    for (uint32_t tid = gpu::threadid(); tid < data.size();
         tid += gpu::nthreads()) {
        const auto col = indices[tid];
        // If bit is one, we'll subtract the data, as bit will be changed to
        // zero
        sd.rows[col] += data[tid] * (-2.0f * bit + 1);
    }

    // Don't flip the bit until every warp is ready
    gpu::syncthreads();

    if (0 == gpu::threadid()) {
        sd.candidate.flip(row);
    }

    gpu::syncthreads();
}

template <typename T>
__host__ __device__ T blockSearch(const SparseView<T> &m, SearchData<T> &sd) {
    const auto n = m.shape().first;
    generateRandomX(sd);
    mx(m, sd);

    auto row = minimumRow(m, sd);
    while (row >= 0) {
        updateXY(m, sd, row);
        row = minimumRow(m, sd);
    }

    return blockDotProd(sd);
}

template <typename T>
__host__ __device__ T search(const SparseView<T> &m, SearchData<T> &sd,
                             uint32_t n) {
    auto minimum = static_cast<T>(INFINITY);
    for (uint32_t bid = gpu::blockid(); bid < n; bid += gpu::nblocks()) {
        auto local_minimum = blockSearch(m, sd);
        if (gpu::threadid() == 0 && local_minimum < minimum) {
            minimum = local_minimum;
            sd.swapCandidates();
        }
        gpu::syncthreads();
    }

    return minimum;
}

template <typename T>
__host__ __device__ void storeMinimum(T minimum, BitVector best_candidate,
                                      T *minimums,
                                      SuperSpan<uint32_t> best_candidates) {
    // Copy the minimum found by the block from shared memory to global memory
    auto mx = best_candidates.subspan(gpu::blockid());
    auto data = best_candidate.data();
    for (uint32_t tid = gpu::threadid(); tid < mx.size();
         tid += gpu::nthreads()) {
        mx[tid] = data[tid];
    }

    // Update the minimum found by the block from register to global memory
    if (0 == gpu::threadid()) {
        minimums[gpu::blockid()] = minimum;
    }
}

template <typename T>
__global__ void searchKernel(const SparseView<T> &m, SuperSpan<T> global_rows,
                             SuperSpan<uint32_t> best_candidates, T *minimums,
                             uint32_t num_values_to_search,
                             size_t shared_capacity) {
    extern __shared__ uint8_t shared_mem[];
    Allocator allocator(shared_mem, shared_capacity);

    // Use rows from global memory
    auto *rows = global_rows.subspan(gpu::blockid()).data();
    auto *s = allocator.allocate<SearchData<T>>(1);
    *s = SearchData<T>(allocator, m.shape().first, gpu::nwarps(), rows);
    assert(allocator.allocated_bytes() <= shared_capacity);

    const auto minimum = search(m, *s, num_values_to_search);
    storeMinimum(minimum, s->best_candidate, minimums, best_candidates);
}

template <typename T>
__global__ void searchKernel(const SparseView<T> &m,
                             SuperSpan<uint32_t> best_candidates, T *minimums,
                             uint32_t num_values_to_search,
                             size_t shared_capacity) {
    extern __shared__ uint8_t shared_mem[];
    Allocator allocator(shared_mem, shared_capacity);

    // Allocate also rows from shared memory
    auto *s = allocator.allocate<SearchData<T>>(1);
    *s = SearchData<T>(allocator, m.shape().first, gpu::nwarps());
    assert(allocator.allocated_bytes() <= shared_capacity);

    const auto minimum = search(m, *s, num_values_to_search);
    storeMinimum(minimum, s->best_candidate, minimums, best_candidates);
}
