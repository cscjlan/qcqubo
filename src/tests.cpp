#include <cstdio>
#include <iostream>
#include <numeric>

#include "gpu.hpp"
#include "qubo.hpp"

namespace testing {
template <typename T>
__device__ void matrixProductPerBlock(const typename SparseMatrix<T>::View &m,
                                      std::span<T> x, std::span<T> y) {
    // This function produces one matrix product per block
    // If multiple blocks call this, they should each provide a unique y vector
    assert(m.shape().first == y.size());
    assert(m.shape().second == x.size());

    const uint32_t wid = gpu::warpid();
    const uint32_t wstride = gpu::nwarps();
    const uint32_t lane = gpu::laneid();
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
    return 0;
}
