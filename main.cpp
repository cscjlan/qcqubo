#include <cassert>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <memory>
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

template <typename F, typename... Args>
__global__ void loop_kernel(F f, int nx, int ny, Args... args) {
    const auto tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const auto tidy = threadIdx.y + blockIdx.y * blockDim.y;
    const auto stridex = blockDim.x * gridDim.x;
    const auto stridey = blockDim.y * gridDim.y;

    const auto num_values = nx * ny;
    for (auto y = tidy; y < ny; y += stridey) {
        const auto offset = y * nx;
        for (auto x = tidx; x < nx; x += stridex) {
            f(offset + x, num_values, args...);
        }
    }
}

void hip_errchk(hipError_t result, const char *file, int32_t line);

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
} // namespace gpu

struct SparseMatrix {
    std::unique_ptr<uint8_t, decltype(&gpu::free)> memory;
    std::span<uint32_t> indptr;
    std::span<uint32_t> indices;
    std::span<float> data;

    static_assert(sizeof(decltype(indptr[0])) == sizeof(decltype(indices[0])),
                  "All types must be equal size");
    static_assert(sizeof(decltype(data[0])) == sizeof(decltype(indices[0])),
                  "All types must be equal size");

    SparseMatrix(const std::vector<uint32_t> &indptr_in,
                 const std::vector<uint32_t> &indices_in,
                 const std::vector<float> &data_in)
        : memory(static_cast<uint8_t *>(gpu::allocate(
                     (indptr_in.size() + indices_in.size() + data_in.size()) *
                     sizeof(decltype(data[0])))),
                 gpu::free) {
        uint8_t *ptr = memory.get();

        auto setSpan = [&ptr](const auto &vec) {
            auto span = std::span<std::remove_cvref_t<decltype(vec[0])>>(
                static_cast<std::remove_cvref_t<decltype(vec[0])> *>(
                    static_cast<void *>(ptr)),
                vec.size());

            uint32_t bytes = span.size_bytes();
            gpu::memcpy(span.data(), vec.data(), bytes);
            ptr += bytes;

            return span;
        };

        indptr = setSpan(indptr_in);
        indices = setSpan(indices_in);
        data = setSpan(data_in);
    }

    __host__ __device__ std::span<uint32_t> rowIndices(uint32_t r) const {
        assert(r + 1 < indptr.size());
        const auto begin = indptr[r];
        const auto end = indptr[r + 1];
        return std::span<uint32_t>(&indices[begin], end - begin);
    }

    __host__ __device__ std::span<float> rowData(uint32_t r) const {
        assert(r + 1 < indptr.size());
        const auto begin = indptr[r];
        const auto end = indptr[r + 1];
        return std::span<float>(&data[begin], end - begin);
    }

    __host__ __device__ std::pair<uint32_t, uint32_t> shape() const {
        return std::make_pair(indptr.size() - 1, indptr.size() - 1);
    }
};

__device__ float warpReduction(float value) {
    // TODO test this
    uint32_t i = 1;
    uint32_t n = warpSize >> i;
    while (n > 1) {
        value += __shfl_down(value, n);
        n = warpSize >> ++i;
    }

    return value;
}

__global__ void testMatrixProduct(SparseMatrix matrix, std::span<float> x,
                                  std::span<float> y) {
    const auto dim = matrix.shape().first;
    const auto begin = dim * blockIdx.x;
    const auto end = begin + dim;

    const std::span<float> block_x(&x[begin], &x[end]);
    const std::span<float> block_y(&y[begin], &y[end]);

    const auto tid = threadIdx.x;
    const auto stride = blockDim.x;

    for (auto row = 0; row < matrix.indptr.size() - 1; row++) {
        const auto indices = matrix.rowIndices(row);
        const auto data = matrix.rowData(row);
        auto sum = 0.0f;
        for (auto i = tid; i < indices.size(); i += stride) {
            const auto j = indices[i];
            sum += block_x[j] * data[i];
        }

        // Row has been summed by the block, perform a reduction
        sum = warpReduction(sum);
        // Now thread 0 has the result computed by the warp
        // TODO thread 0 should write to shared memory (based on warp id)
        // then syncthreads, then first warp performs the reduction again
        // and then thread 0 of warp 0 writes the value to the blockx[row]
    }
}

__global__ void testWarpReduction() {
    const float value = warpReduction(threadIdx.x);
    if (threadIdx.x == 0) {
        printf("%f\n", value);
    }
}

void testSparseMatrix() {
    std::vector<uint32_t> indptr{0, 2, 3, 4, 5};
    std::vector<uint32_t> indices{0, 2, 1, 2, 3};
    std::vector<float> data{11.0f, 13.0f, 22.0f, 33.0f, 44.0f};

    const SparseMatrix mat(indptr, indices, data);

    for (auto i = 0; i < indptr.size(); i++) {
        assert(indptr[i] == mat.indptr[i]);
    }

    for (auto i = 0; i < indices.size(); i++) {
        assert(indices[i] == mat.indices[i]);
    }

    for (auto i = 0; i < data.size(); i++) {
        assert(data[i] == mat.data[i]);
    }

    gpu::synchronize();
}

int main() {
    testSparseMatrix();
    LAUNCH_KERNEL(testWarpReduction, 1, 64, 0, 0, "");
    gpu::synchronize();
}
