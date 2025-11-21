#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

#define LAUNCH_KERNEL(kernel, ...)                                             \
    gpu::launch_kernel(#kernel, __FILE__, __LINE__, kernel, __VA_ARGS__)
#define HIP_ERRCHK(result) gpu::hip_errchk(result, __FILE__, __LINE__)

// Here we have generic GPU related boilerplate
namespace gpu {
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
static constexpr uint32_t WARP_SIZE = 32ul;
static constexpr uint32_t WARP_SIZE_BASE_TWO = 5ul;
#elif defined(__HIPCC__)
static constexpr uint32_t WARP_SIZE = 64ul;
static constexpr uint32_t WARP_SIZE_BASE_TWO = 6ul;
#endif

__forceinline__ __device__ uint32_t nblocks() { return gridDim.x; }
__forceinline__ __device__ uint32_t nthreads() { return blockDim.x; }
__forceinline__ __device__ uint32_t nwarps() {
    return max(nthreads() >> WARP_SIZE_BASE_TWO, 1);
}
__forceinline__ __device__ uint32_t nlanes() {
    return min(nthreads(), WARP_SIZE);
}

__forceinline__ __device__ uint32_t blockid() { return blockIdx.x; }
__forceinline__ __device__ uint32_t threadid() { return threadIdx.x; }
__forceinline__ __device__ uint32_t warpid() {
    return threadIdx.x >> WARP_SIZE_BASE_TWO;
}
__forceinline__ __device__ uint32_t laneid() {
    return threadIdx.x & (WARP_SIZE - 1);
}

__forceinline__ __device__ void syncthreads() { __syncthreads(); }

// These are for testing the code serially on the CPU
__forceinline__ __host__ uint32_t nblocks() { return 1; }
__forceinline__ __host__ uint32_t nthreads() { return 1; }
__forceinline__ __host__ uint32_t nwarps() { return 1; }
__forceinline__ __host__ uint32_t nlanes() { return 1; }

__forceinline__ __host__ uint32_t blockid() { return 0; }
__forceinline__ __host__ uint32_t threadid() { return 0; }
__forceinline__ __host__ uint32_t warpid() { return 0; }
__forceinline__ __host__ uint32_t laneid() { return 0; }

__forceinline__ __host__ void syncthreads() {}

inline void hip_errchk(hipError_t result, const char *file, int32_t line) {
    if (result != hipSuccess) {
        std::printf("\n\n%s in %s at line %d\n", hipGetErrorString(result),
                    file, line);
        exit(EXIT_FAILURE);
    }
}

inline void eventDestroy(hipEvent_t *event) {
    HIP_ERRCHK(hipEventDestroy(*event));
    delete event;
}

inline void streamDestroy(hipStream_t *stream) {
    HIP_ERRCHK(hipStreamDestroy(*stream));
    delete stream;
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

inline void memset(void *dst, uint32_t value, size_t num_bytes) {
    HIP_ERRCHK(hipMemset(dst, value, num_bytes));
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

// The single threads serial host code just passes the value back
template <typename T, typename Op> __host__ T warpReduce(T value, Op) {
    return value;
}

template <typename T, typename U, typename Op>
__device__ std::pair<T, U> warpArgSearch(T t, U u, Op op) {
    // We want the value of u from the lane that has the minimum value of t
    const uint32_t lane = gpu::laneid();
    uint32_t minlane = lane;
    uint32_t delta = gpu::WARP_SIZE >> 1u;
    while (delta > 0ul) {
        auto interpolate = [](auto a, auto b, auto s) {
            return s * a + (1 - s) * b;
        };
        const auto s = static_cast<T>(lane + delta < gpu::nlanes());
        // The lanes that are not active (seem to) have zeros in them.
        // If we're doing a minimum/maximum search with strictly
        // positive/negative values, we'll end up with a false min/max of zero
        // taken from an inactive lane, unless we check for that.
        auto val = interpolate(__shfl_down(t, delta), t, s);
        auto ind = interpolate(__shfl_down(minlane, delta), minlane, s);

        std::tie(t, minlane) = op(val, t, ind, minlane);
        delta >>= 1u;
    }

    // Get the u from the lane with the minimum to lane 0
    u = __shfl_down(u, minlane);
    return std::make_pair(t, u);
};

template <typename T, typename U, typename Op>
__host__ std::pair<T, U> warpArgSearch(T t, U u, Op) {
    // Host with one thread just passes the values
    return std::make_pair(t, u);
};

struct GPUMem {
    static void *allocate(size_t num_bytes) { return gpu::allocate(num_bytes); }
    static void free(void *ptr) { gpu::free(ptr); }
    static void memcpy(void *dst, const void *src, size_t num_bytes) {
        gpu::memcpy(dst, src, num_bytes);
    }
};
} // namespace gpu
