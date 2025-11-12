#include <cassert>
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

template <typename T, typename Op> __device__ T warpReduction(T value, Op op) {
    uint32_t n = ::warpSize >> 1u;
    while (n > 0) {
        value = op(value, __shfl_down(value, n));
        n >>= 1;
    }

    return value;
}
} // namespace gpu

// This only supporst square matrices
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

template <typename T>
__device__ void matrixProductPerBlock(const typename SparseMatrix<T>::View &m,
                                      std::span<T> x, std::span<T> y) {
    // This function produces one matrix product per block
    // If multiple blocks call this, they should each provide a unique y vector
    assert(m.shape().first == y.size());
    assert(m.shape().second == x.size());

    const uint32_t warp_size_base_2 = 5 + (warpSize >> 6);
    const uint32_t wid = threadIdx.x >> warp_size_base_2;
    const uint32_t wstride = max(blockDim.x >> warp_size_base_2, 1);
    const uint32_t lane = threadIdx.x & (warpSize - 1);
    const uint32_t lanestride = warpSize < blockDim.x ? warpSize : blockDim.x;

    for (auto row = wid; row < m.shape().first; row += wstride) {
        const auto row_indices = m.rowIndices(row);
        const auto row_data = m.rowData(row);
        auto sum = static_cast<T>(0);
        for (auto i = lane; i < row_indices.size(); i += lanestride) {
            sum += row_data[i] * x[row_indices[i]];
        }

        sum = gpu::warpReduction(sum, [](auto a, auto b) { return a + b; });
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
    value = gpu::warpReduction(value, op);
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

template <typename T>
__device__ void mx(const typename SparseMatrix<T>::View &m,
                   std::span<uint32_t> x, std::span<T> y) {
    constexpr static uint32_t values_per_x = 8 * sizeof(decltype(x[0]));
    // This assumes 32 or 64 bit values
    constexpr static uint32_t x_size_base_2 = 5 + (values_per_x >> 6);

    assert(m.shape().first == y.size());
    assert(m.shape().second == values_per_x * x.size());

    const uint32_t warp_size_base_2 = 5 + (warpSize >> 6);
    const uint32_t wid = threadIdx.x >> warp_size_base_2;
    const uint32_t wstride = max(blockDim.x >> warp_size_base_2, 1);
    const uint32_t lane = threadIdx.x & (warpSize - 1);
    const uint32_t lanestride = warpSize < blockDim.x ? warpSize : blockDim.x;

    for (auto row = wid; row < m.shape().first; row += wstride) {
        const auto row_indices = m.rowIndices(row);
        const auto row_data = m.rowData(row);
        auto sum = static_cast<T>(0);
        for (auto i = lane; i < row_indices.size(); i += lanestride) {
            // j is the column of the matrix
            const auto j = row_indices[i];
            // x_id is the index to the x vector, with multiple bits per x
            const auto x_id = j >> x_size_base_2;
            // bit_id contains the correct bit from the fetched x value
            const auto bit_id = j & (values_per_x - 1);
            // Finally, perform a product between the matrix value at (row, j)
            // and the bit j in the bit vector
            sum += row_data[i] * (x[x_id] >> bit_id) & 1;
        }

        sum = gpu::warpReduction(sum, [](auto a, auto b) { return a + b; });
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

__device__ void generateRandomX(std::span<uint32_t> x) {
    // TODO
}

template <typename T>
__device__ bool keepSearching(const typename SparseMatrix<T>::View &m,
                              std::span<uint32_t> x, std::span<T> y,
                              std::span<uint32_t> scratch) {
    // do the computations, then recudction to scratch, then warp reduction to
    // one value, then sync, then everyone reads it and returns y[i] < 0.0
    //
    // sign = -2 * candidate.x + 1
    // delta = sign * (2.0 * candidate.energies + sign * self.diagonal)
    // self.i = np.argmin(delta)

    // return delta[self.i] < 0.0
}

template <typename T>
__device__ void updateXY(const typename SparseMatrix<T>::View &m,
                         std::span<uint32_t> x, std::span<T> y) {
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

template <typename T>
__global__ void search(typename SparseMatrix<T>::View m, std::span<uint32_t> x,
                       std::span<uint32_t> min_x, std::span<T> y,
                       std::span<T> min_y, uint32_t num_values_to_search) {
    // TODO: needs dynamic memory of sizeof(uint32_t) * blockDim.x / warpSize
    extern __shared__ uint32_t scratch[];

    const auto num_rows = m.shape().first;
    const auto num_cols = m.shape().second;

    constexpr static uint32_t values_per_x = 8 * sizeof(decltype(x[0]));
    assert(values_per_x * x.size() == num_cols * gridDim.x);
    assert(values_per_x * min_x.size() == num_cols * gridDim.x);
    assert(y.size() == num_cols * gridDim.x);
    // Only one value per block in min_y
    assert(min_y.size() == gridDim.x);

    const auto begin = blockIdx.x * num_cols;
    const auto end = begin + num_cols;

    // Make spans for each block
    std::span<uint32_t> bx(x.data() + begin, x.data() + end);
    std::span<uint32_t> bmx(min_x.data() + begin, min_x.data() + end);
    std::span<T> by(y.data() + begin, y.data() + end);

    for (uint32_t bid = blockIdx.x; bid < num_values_to_search;
         bid += gridDim.x) {
        generateRandomX(bx);
        mx(m, bx, by);

        // The previous does a warp reduction, here we'll work on the reduced
        // values
        __syncthreads();

        while (
            keepSearching(m, bx, by, std::span<uint32_t>(scratch, warpSize))) {
            updateXY(m, bx, by);
        }
        // We've reached a local minimum
        // Reduce the y to a single value
        // Then compare to minimum_y by the entire block
        // if smaller, every thread swaps bx to bmx, so the x is stored in the
        // other if smaller, thread 0 updates min_y[blockIdx.x] to the computed
        // y
    }

    // We've done the computation for this block,
    // next we may need to do a final copy:
    // if bx is the same pointer as x, we do no copy, as the minimum will be in
    // bmx otherwise, we'll do a copy from bx to bmx
}

int main() {
    std::printf("Test matrix product\n");
    std::fflush(stdout);
    testMatrixProduct();

    std::printf("test sparse matrix\n");
    std::fflush(stdout);
    testSparseMatrix();

    std::printf("test warp reduction\n");
    std::fflush(stdout);
    testWarpReduction();

    gpu::synchronize();
}
