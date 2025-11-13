#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#define LAUNCH_KERNEL(kernel, ...)                                             \
    gpu::launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define HIP_ERRCHK(result) gpu::hip_errchk(result, __FILE__, __LINE__)

// Here we have generic GPU related boilerplate
namespace gpu {
// clang-format off
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
    static constexpr uint32_t WARP_SIZE = 32ul;
    static constexpr uint32_t WARP_SIZE_BASE_TWO = 5ul;
#elif defined(__HIPCC__)
    static constexpr uint32_t WARP_SIZE = 64ul;
    static constexpr uint32_t WARP_SIZE_BASE_TWO = 6ul;
#endif
// clang-format on

__device__ uint32_t laneID() { return threadIdx.x & (WARP_SIZE - 1); }

__device__ uint32_t warpID() { return threadIdx.x >> WARP_SIZE_BASE_TWO; }

__device__ uint32_t numWarpsInBlock() {
    return max(blockDim.x >> WARP_SIZE_BASE_TWO, 1);
}

inline void hip_errchk(hipError_t result, const char *file, int32_t line) {
    if (result != hipSuccess) {
        std::printf("\n\n%s in %s at line %d\n", hipGetErrorString(result),
                    file, line);
        exit(EXIT_FAILURE);
    }
}

inline void *allocate(size_t num_bytes) {
    void *ptr = nullptr;
    HIP_ERRCHK(hipMalloc(&ptr, num_bytes));
    if (ptr == nullptr) {
        std::fprintf(stderr, "GPU malloc allocated a nullptr\n");
        std::abort();
    }

    return ptr;
}

inline void free(void *ptr) { HIP_ERRCHK(hipFree(ptr)); }

inline void memcpy(void *dst, const void *src, size_t num_bytes) {
    HIP_ERRCHK(hipMemcpy(dst, src, num_bytes, hipMemcpyDefault));
}

inline void synchronize() { HIP_ERRCHK(hipDeviceSynchronize()); }

template <typename... Args>
void launch_kernel(const char *kernel_name, const char *file, int32_t line,
                   void (*kernel)(Args...), dim3 blocks, dim3 threads,
                   size_t num_bytes_shared_mem, hipStream_t stream,
                   const char *function_name, Args... args) {
#if !NDEBUG
    int32_t device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    // Helper lambda for querying device attributes
    auto get_device_attribute = [&device](hipDeviceAttribute_t attribute) {
        int32_t value = 0;
        HIP_ERRCHK(hipDeviceGetAttribute(&value, attribute, device));
        return value;
    };

    // Get maximum allowed size of block for each dimension
    const dim3 max_threads(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxBlockDimZ));

    // Get maximum allowed size of grid for each dimension
    const dim3 max_blocks(
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimX),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimY),
        get_device_attribute(
            hipDeviceAttribute_t::hipDeviceAttributeMaxGridDimZ));

    // Maximum threads per block in total (i.e. x * y * z)
    const int32_t max_threads_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxThreadsPerBlock);

    // Maximum number of bytes of shared memory per block
    const int32_t max_shared_memory_per_block = get_device_attribute(
        hipDeviceAttribute_t::hipDeviceAttributeMaxSharedMemoryPerBlock);

    auto error_print_prelude = [&kernel_name, &function_name]() {
        std::fprintf(
            stderr,
            "Bad launch parameters for kernel \"%s\" with lambda \"%s\"\n",
            kernel_name, function_name);
    };
    // Helper lambda for asserting dim3 launch variable is within allowed limits
    auto assert_within_limits =
        [&error_print_prelude](const char *name, int32_t value, int32_t min,
                               int32_t max) {
            if (not(min <= value && value <= max)) {
                error_print_prelude();
                std::fprintf(stderr, "%s (%d) not within limits [%d, %d]\n",
                             name, value, min, max);
                exit(EXIT_FAILURE);
            }
        };

    assert_within_limits("threads.x", threads.x, 1, max_threads.x);
    assert_within_limits("threads.y", threads.y, 1, max_threads.y);
    assert_within_limits("threads.z", threads.z, 1, max_threads.z);
    assert_within_limits("blocks.x", blocks.x, 1, max_blocks.x);
    assert_within_limits("blocks.y", blocks.y, 1, max_blocks.y);
    assert_within_limits("blocks.z", blocks.z, 1, max_blocks.z);
    assert_within_limits("block size", threads.x * threads.y * threads.z, 1,
                         max_threads_per_block);

    // Requested amount of shared memory must be below the limit queried above
    if (num_bytes_shared_mem > max_shared_memory_per_block) {
        error_print_prelude();
        std::fprintf(stderr, "Shared memory request too large: %ld > %d\n",
                     num_bytes_shared_mem, max_shared_memory_per_block);
        exit(EXIT_FAILURE);
    }

    // Reset the error variable to success
    [[maybe_unused]] auto result = hipGetLastError();
#endif

    kernel<<<blocks, threads, num_bytes_shared_mem, stream>>>(args...);

#if !NDEBUG
    // Quoting from HIP documentation
    // (https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/error_handling.html)
    //
    //  > hipGetLastError() returns the returned error code of the last HIP
    //    runtime API call even if it’s hipSuccess, while cudaGetLastError
    //    returns the error returned by any of the preceding CUDA APIs in the
    //    same host thread. hipGetLastError() behavior will be matched with
    //    cudaGetLastError in ROCm release 7.0.
    //
    // Because of this, using the Cuda recommended pattern of cathcing kernel
    // errors by first synchronizing with the device, then calling
    // hipGetLastError doesn't work. Until ROCm 7.0, HIP will overwrite the
    // error code returned by the kernel with success from hipDeviceSynchronize.
    // This means hipGetLastError can only be used to catch launch parameter
    // errors, i.e. errors that happen during the kernel launch, like too many
    // threads per block. Any errors that happen during the asynchronous kernel
    // execution are missed. To be able to catch even the kernel launch errors,
    // one must not synchronize first, if using ROCm < 7.0, or the errors will
    // be overwritten.

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
    result = hipDeviceSynchronize();
#endif
    result = hipGetLastError();
    if (result != hipSuccess) {
        std::printf(
            "Error with kernel \"%s\" executing lambda \"%s\" in %s at line "
            "%d\n%s: %s\n",
            kernel_name, function_name, file, line, hipGetErrorName(result),
            hipGetErrorString(result));
        exit(EXIT_FAILURE);
    }
#endif
}

inline uint32_t warpSize() {
    int32_t device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    int32_t value = 0;
    HIP_ERRCHK(hipDeviceGetAttribute(
        &value, hipDeviceAttribute_t::hipDeviceAttributeWarpSize, device));
    return value;
}

template <typename T, typename Op> __device__ T warpReduce(T value, Op op) {
    value = op(value, __shfl_down(value, WARP_SIZE >> 1u));
    value = op(value, __shfl_down(value, WARP_SIZE >> 2u));
    value = op(value, __shfl_down(value, WARP_SIZE >> 3u));
    value = op(value, __shfl_down(value, WARP_SIZE >> 4u));
    value = op(value, __shfl_down(value, WARP_SIZE >> 5u));
    if constexpr ((WARP_SIZE >> 6ul) > 0ul) {
        value = op(value, __shfl_down(value, WARP_SIZE >> 6u));
    }

    return value;
}
} // namespace gpu

template <uint32_t NUM, uint32_t POW>
__device__ consteval uint32_t asPowOfTwo() {
    if constexpr ((NUM >> POW) > 1) {
        return asPowOfTwo<NUM, POW + 1>();
    }

    return POW;
}

// This only supports square matrices
template <typename T> struct SparseMatrix {
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
    std::unique_ptr<std::byte, decltype(&gpu::free)> memory;

  public:
    struct View {
        friend SparseMatrix<T>;

      private:
        std::span<uint32_t> indptr;
        std::span<uint32_t> indices;
        std::span<T> data;

      public:
        std::span<T> diagonal;

        __device__ std::span<uint32_t> rowIndices(uint32_t r) const {
            assert(r + 1 < indptr.size());
            const auto begin = indptr[r];
            const auto end = indptr[r + 1];
            return std::span<uint32_t>(&indices[begin], end - begin);
        }

        __device__ std::span<T> rowData(uint32_t r) const {
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
    } view;

    SparseMatrix(const std::vector<uint32_t> &indptr,
                 const std::vector<uint32_t> &indices,
                 const std::vector<T> &data)
        : num_bytes(numBytesReq(data.size(), indptr.size() - 1)),
          memory(static_cast<std::byte *>(gpu::allocate(num_bytes)),
                 gpu::free) {
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
            if (std::align(alignof(V), vec.size(), ptr, space)) {
                auto span = std::span<V>(static_cast<V *>(ptr), vec.size());
                uint32_t bytes = span.size_bytes();
                gpu::memcpy(span.data(), vec.data(), bytes);
                ptr = static_cast<std::byte *>(ptr) + bytes;
                space -= bytes;

                return span;
            }
            return std::span<V>(static_cast<V *>(ptr), 0);
        };

        view.data = setSpan(data);
        view.diagonal = setSpan(diagonal);
        view.indices = setSpan(indices);
        view.indptr = setSpan(indptr);

        assert(view.data.size() == data.size());
        assert(view.diagonal.size() == diagonal.size());
        assert(view.indices.size() == indices.size());
        assert(view.indptr.size() == indptr.size());
    }
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

    static uint32_t requiredLength(uint32_t num_subspans,
                                   uint32_t subspan_len) {
        return num_subspans * width(num_subspans, subspan_len);
    }

    __device__ __host__ static uint32_t width(uint32_t num_subspans,
                                              uint32_t subspan_len) {
        // Compute the width of the subspan
        // If subspan_len is aligned to ALIGN, i.e. remainder == 0
        // the width is equal to subspan_len
        const uint32_t remainder = subspan_len & (ALIGN - 1);
        const uint32_t padding = (ALIGN - remainder) & (ALIGN - 1);
        return subspan_len + padding;
    }

  public:
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

    __device__ __host__ std::span<T> subspan(uint32_t i) const {
        assert(i < num_subspans);
        // Skip all the i widths before this subspan
        const uint32_t begin = i * width(num_subspans, subspan_len);

        // Are we inside the super span?
        assert(begin + subspan_len < data.size());

        return std::span<T>(data.data() + begin, subspan_len);
    }

    __device__ __host__ uint32_t subspanLength() const { return subspan_len; }
};

struct BitVector {
    // This structure is a bit vector, where 32 bits are stored as a single
    // value. We're using 4 byte values to avoid bank conflicts in shared memory
    // when multiple threads want to access a bit from the same value: with 4
    // bytes per value, all (potentially 32) threads access the same (starting)
    // byte within the 4 byte bank, so the value can be broadcast efficiently.
    using S = uint32_t;

  private:
    uint32_t len;
    std::span<S> data;
    static constexpr uint32_t BITS_PER_TYPE = 8u * sizeof(S);
    static constexpr uint32_t POW_OF_TWO = asPowOfTwo<BITS_PER_TYPE, 0>();

  public:
    __device__ __host__ BitVector(uint32_t l, std::span<S> d)
        : len(l), data(d) {}

    static uint32_t requiredLength(uint32_t len) {
        return (len >> POW_OF_TWO) + 1;
    }

    static size_t memReq(uint32_t len) {
        return static_cast<size_t>(requiredLength(len)) * sizeof(S);
    }

    template <typename T> __device__ __host__ T operator[](uint32_t i) const {
        // We should return the bit at index i
        // We'll unpack the correct value in data and return the correct bit
        // from that value
        assert(i < data.size() * BITS_PER_TYPE);
        const auto data_idx = i >> POW_OF_TWO;
        const auto bit_idx = i & (BITS_PER_TYPE - 1);
        return static_cast<T>((data[data_idx] >> bit_idx) & 1);
    }

    // Return the underlying span for efficient copies/inserts of 4 byte values
    std::span<S> span() const { return data; }

    // How many bits are actual data
    size_t size() const { return len; }
};

// TODO: test
template <typename T>
__device__ void mx(const typename SparseMatrix<T>::View &m, BitVector x,
                   std::span<T> y) {
    const auto shape = m.shape();
    const auto num_rows = shape.first;
    const auto num_cols = shape.second;
    assert(x.size() == num_cols);
    assert(y.size() == num_rows);

    const uint32_t wid = gpu::warpID();
    const uint32_t wstride = gpu::numWarpsInBlock();
    const uint32_t lane = gpu::laneID();
    const uint32_t lanestride =
        gpu::WARP_SIZE < blockDim.x ? gpu::WARP_SIZE : blockDim.x;

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
    __syncthreads();
}

__device__ void generateRandomX(BitVector x) {
    // TODO
}

// TODO: test
template <typename T, typename Op>
__device__ T blockReduce(std::span<T> data, std::span<T> scratch, Op op,
                         T initial) {
    // When calling with multiple blocks, data and scratch must be unique per
    // block
    T result = initial;
    for (size_t i = threadIdx.x; i < data.size(); i += blockDim.x) {
        result = op(result, data[i]);
    }

    const uint32_t wid = gpu::warpID();
    const uint32_t lane = gpu::laneID();

    result = gpu::warpReduce(result, op);
    if (0 == lane) {
        scratch[wid] = result;
    }

    // Wait until all warps in block are done
    __syncthreads();

    // Perform the final reduction by the first warp
    if (wid == 0) {
        result = lane < gpu::numWarpsInBlock() ? scratch[lane] : initial;
        result = gpu::warpReduce(result, op);

        // First thread stores the result in memory accessible to the entire
        // block
        if (0 == lane) {
            scratch[0] = result;
        }
    }

    // Others wait until first warp is done
    __syncthreads();

    // Everyone returns the same value from scratch memory
    return scratch[0];
}

template <typename T>
__device__ bool keepSearching(const typename SparseMatrix<T>::View &m,
                              BitVector x, std::span<T> y,
                              std::span<T> scratch) {
    // do the computations, then recudction to scratch, then warp reduction to
    // one value, then sync, then everyone reads it and returns y[i] < 0.0
    //
    // sign = -2 * candidate.x + 1
    // delta = sign * (2.0 * candidate.energies + sign * self.diagonal)
    // self.i = np.argmin(delta)

    // return delta[self.i] < 0.0
    return false;
}

template <typename T>
__device__ void updateXY(const typename SparseMatrix<T>::View &m, BitVector x,
                         std::span<T> y) {
    // # sq is a sparse array
    // begin = sq.indptr[i]
    // end = sq.indptr[i + 1]

    // row_indices = sq.indices[begin:end]
    // row_data = sq.data[begin:end]

    // # Update energies for rows
    // delta = row_data * (-2 * self.x[i] + 1)
    // self.energies[row_indices] += delta

    // # Flip bit i
    //  this can be done by thread 0, then everyone syncs
    // self.x[i] = abs(self.x[i] - 1)
}

template <typename T, typename Op>
__device__ T updateMinimum(BitVector &x, BitVector &min_x, std::span<T> y,
                           T min_y, std::span<T> scratch, Op op, T initial) {
    auto result = blockReduce(y, scratch, op, initial);
    if (result < min_y) {
        min_y = result;
        // TODO: make sure this works
        std::swap(x, min_x);
    }
    return min_y;
}

template <typename T>
__device__ T blockSearch(const typename SparseMatrix<T>::View &m, BitVector &x,
                         BitVector &min_x, std::span<T> y, T min_y,
                         std::span<T> scratch) {
    generateRandomX(x);
    mx(m, x, y);

    while (keepSearching(m, x, y, scratch)) {
        updateXY(m, x, y);
    }

    return updateMinimum(
        x, mx, y, min_y, scratch, [](auto a, auto b) { return a + b; }, 0);
}

template <typename T>
__device__ void search(const typename SparseMatrix<T>::View &m,
                       std::span<T> block_y, SuperSpan<uint32_t> min_x,
                       std::span<T> min_y, std::span<std::byte> shared_mem,
                       uint32_t num_values_to_search) {
    // TODO can this be made to a testable function?
    // Construct bit vectors for x and minimum x in shared memory
    const auto num_cols = m.shape().second;
    using S = typename BitVector::S;
    const auto len_x = min_x.subspanLength();
    BitVector block_x(
        num_cols,
        std::span<S>(static_cast<S *>(static_cast<void *>(shared_mem.data())),
                     len_x));
    BitVector block_min_x(num_cols, std::span<S>(block_x.span().end(), len_x));

    // TODO: maybe alignment?
    std::span<T> scratch(static_cast<T *>(static_cast<void *>(
                             block_min_x.span().data() + len_x)),
                         gpu::numWarpsInBlock());

    auto minimum_y = static_cast<T>(INFINITY);
    for (uint32_t bid = blockIdx.x; bid < num_values_to_search;
         bid += gridDim.x) {
        minimum_y =
            blockSearch(m, block_x, block_min_x, block_y, minimum_y, scratch);
    }

    // Copy the minimum found by the block from shared memory to global memory
    auto bmx = min_x.subspan(blockIdx.x);
    auto bmx_shared = block_min_x.span();
    for (uint32_t tid = threadIdx.x; tid < bmx.size(); tid += blockDim.x) {
        bmx[tid] = bmx_shared[tid];
    }

    // Update the minimum_y found by the block from register to global memory
    if (0 == threadIdx.x) {
        min_y[blockIdx.x] = minimum_y;
    }
}

template <typename T>
__global__ void searchKernel(const typename SparseMatrix<T>::View &m,
                             SuperSpan<T> y, SuperSpan<uint32_t> min_x,
                             std::span<T> min_y, size_t shared_mem_bytes,
                             uint32_t num_values_to_search) {
    extern __shared__ std::byte shared_mem[];
    auto block_y = y.subspan(blockIdx.x);
    search(m, block_y, min_x, min_y,
           std::span<std::byte>(shared_mem, shared_mem_bytes),
           num_values_to_search);
}

template <typename T>
__global__ void searchKernel(const typename SparseMatrix<T>::View &m,
                             SuperSpan<uint32_t> min_x, std::span<T> min_y,
                             size_t shared_mem_bytes,
                             uint32_t num_values_to_search) {
    extern __shared__ std::byte shared_mem[];

    // Construct y in shared memory
    std::span<T> block_y(static_cast<T *>(static_cast<void *>(shared_mem)),
                         m.shape().second);

    // TODO: maybe align these correctly
    // The remainder of the shared memory is passed on
    const auto ptr = static_cast<std::byte *>(
        static_cast<void *>(block_y.data() + block_y.size()));
    const auto bytes = shared_mem_bytes - block_y.size_bytes();

    search(m, block_y, min_x, min_y, std::span<std::byte>(ptr, bytes),
           num_values_to_search);
}

namespace testing {
template <typename T>
__device__ void matrixProductPerBlock(const typename SparseMatrix<T>::View &m,
                                      std::span<T> x, std::span<T> y) {
    // This function produces one matrix product per block
    // If multiple blocks call this, they should each provide a unique y vector
    assert(m.shape().first == y.size());
    assert(m.shape().second == x.size());

    const uint32_t wid = gpu::warpID();
    const uint32_t wstride = gpu::numWarpsInBlock();
    const uint32_t lane = gpu::laneID();
    const uint32_t lanestride =
        gpu::WARP_SIZE < blockDim.x ? gpu::WARP_SIZE : blockDim.x;

    for (auto row = wid; row < m.shape().first; row += wstride) {
        const auto row_indices = m.rowIndices(row);
        const auto row_data = m.rowData(row);
        auto sum = static_cast<T>(0);
        for (auto i = lane; i < row_indices.size(); i += lanestride) {
            sum += row_data[i] * x[row_indices[i]];
        }

        sum = gpu::warpReduce(sum, [](auto a, auto b) { return a + b; });
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

template <typename T>
__device__ void matrixProduct(const typename SparseMatrix<T>::View &m,
                              std::span<T> x, std::span<T> y) {
    // X and Y contain many vectors
    const uint32_t length = m.shape().second;
    const uint32_t bid = blockIdx.x;
    const uint32_t bstride = gridDim.x;
    for (uint32_t i = bid; i < x.size() / length; i += bstride) {
        const auto begin = i * length;
        const auto end = begin + length;

        const std::span<T> block_x(x.data() + begin, x.data() + end);
        const std::span<T> block_y(y.data() + begin, y.data() + end);

        matrixProductPerBlock(m, block_x, block_y);
    }
}

template <typename T>
__global__ void mp(typename SparseMatrix<T>::View m, std::span<T> x,
                   std::span<T> y) {
    matrixProduct(m, x, y);
}

void testMatrixProduct() {
    using T = float;
    constexpr static auto n = 4;
    std::vector<uint32_t> indptr{0, 2, 3, 4, 5};
    std::vector<uint32_t> indices{0, 2, 1, 2, 3};
    std::vector<T> data{11.0f, 13.0f, 22.0f, 33.0f, 44.0f};

    const SparseMatrix mat(indptr, indices, data);

    std::vector<T> x(n, 1.0f);
    std::vector<T> y(n, 0.0f);

    auto mem_x = gpu::allocate(sizeof(T) * x.size());
    auto mem_y = gpu::allocate(sizeof(T) * y.size());

    std::span<T> x_d(static_cast<T *>(mem_x), x.size());
    std::span<T> y_d(static_cast<T *>(mem_y), y.size());

    gpu::memcpy(x_d.data(), x.data(), x_d.size_bytes());
    gpu::memcpy(y_d.data(), y.data(), y_d.size_bytes());

    LAUNCH_KERNEL(mp, 1, 64, 0, 0, "", mat.view, x_d, y_d);

    gpu::memcpy(y.data(), y_d.data(), y_d.size_bytes());

    assert(y[0] == 24.0f);
    assert(y[1] == 22.0f);
    assert(y[2] == 33.0f);
    assert(y[3] == 44.0f);

    gpu::free(mem_x);
    gpu::free(mem_y);
}

template <typename T, typename Op>
__global__ void warpReductionKernel(std::span<T> values, Op op) {
    auto value = threadIdx.x < values.size() ? values[threadIdx.x] : 0;
    value = gpu::warpReduce(value, op);
    if (threadIdx.x == 0) {
        values[0] = value;
    }
}

void testSparseMatrix() {
    constexpr static auto n = 4;
    std::vector<uint32_t> indptr{0, 2, 3, 4, 5};
    std::vector<uint32_t> indices{0, 2, 1, 2, 3};
    std::vector<float> data{11.0f, 13.0f, 22.0f, 33.0f, 44.0f};

    const SparseMatrix mat(indptr, indices, data);
    const auto diagonal = mat.view.diagonal;

    assert(diagonal[0] == 11.0f);
    assert(diagonal[1] == 22.0f);
    assert(diagonal[2] == 33.0f);
    assert(diagonal[3] == 44.0f);

    const auto dense = mat.view.toDense();
    auto i = 0;
    for (auto d : dense) {
        const auto row = i / n + 1;
        const auto col = i % n + 1;
        if (row == col) {
            assert(d == static_cast<float>(row * 11.0f));
        } else if (row == 1 && col == 3) {
            assert(d == 13.0f);
        } else {
            assert(d == 0.0f);
        }
        std::cout << d << std::endl;
        i++;
    }

    gpu::synchronize();
}

void testWarpReduction() {
    const uint32_t n = gpu::warpSize();
    auto mem = gpu::allocate(sizeof(double) * n);

    auto launchWarpReduction =
        []<typename T, typename Op>(std::span<T> values,
                                    const std::vector<T> &h_values, Op op) {
            gpu::memcpy(values.data(), h_values.data(), values.size_bytes());

            LAUNCH_KERNEL(warpReductionKernel, 1, values.size(), 0, 0, "",
                          values, op);

            T result = 0;
            gpu::memcpy(&result, values.data(), sizeof(T));
            T h_result = std::reduce(h_values.cbegin() + 1, h_values.cend(),
                                     h_values[0], op);

            std::cout << result << ", " << h_result << std::endl;

            assert(result == h_result);
        };

    launchWarpReduction(
        std::span(static_cast<uint32_t *>(mem), n),
        [&n]() {
            std::vector<uint32_t> values(n);
            std::iota(values.begin(), values.end(), 0);
            return values;
        }(),
        [](auto a, auto b) { return a + b; });

    launchWarpReduction(
        std::span(static_cast<float *>(mem), n),
        [&n]() {
            std::vector<float> values(n);
            std::generate(values.begin(), values.end(), [&n, i = 0]() mutable {
                return -static_cast<float>(n) / 2.0f + (i++);
            });
            return values;
        }(),
        [](auto a, auto b) { return a < b ? a : b; });

    gpu::free(mem);
}

} // namespace testing

int main() {
    std::printf("Test matrix product\n");
    std::fflush(stdout);
    testing::testMatrixProduct();

    std::printf("test sparse matrix\n");
    std::fflush(stdout);
    testing::testSparseMatrix();

    std::printf("test warp reduction\n");
    std::fflush(stdout);
    testing::testWarpReduction();

    gpu::synchronize();
}
