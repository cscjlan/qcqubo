#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>

#include "gpu.hpp"
#include "qubo.hpp"

namespace testing {
struct CPUMem {
    static void *allocate(size_t num_bytes) { return std::malloc(num_bytes); }

    static void free(void *ptr) { std::free(ptr); }

    static void memcpy(void *dst, const void *src, size_t num_bytes) {
        std::memcpy(dst, src, num_bytes);
    }
};

SparseMatrix<float, CPUMem> fromDense(const std::vector<float> &dense,
                                      uint32_t num_rows) {
    assert(dense.size() == num_rows * num_rows);
    std::vector<uint32_t> indptr{};
    std::vector<uint32_t> indices{};
    std::vector<float> data{};
    const uint32_t num_cols = num_rows;

    indptr.push_back(0);
    for (auto row = 0; row < num_rows; row++) {
        for (auto col = 0; col < num_cols; col++) {
            const auto index = row * num_cols + col;
            const auto value = dense[index];
            if (value != 0.0f) {
                data.push_back(value);
                indices.push_back(col);
            }
        }
        const auto end = data.size();
        indptr.push_back(end);
    }

    assert(indptr.size() == num_rows + 1);

    return SparseMatrix<float, CPUMem>(indptr, indices, data);
}

void testFromToDense() {
    std::printf("Testing fromDense\n");
    static constexpr uint32_t num_rows = 4;

    std::vector<float> dense{
        11.0f, 0.0f,  0.0f,  0.0f, 21.0f, 22.0f, 0.0f, 0.0f,
        0.0f,  32.0f, 33.0f, 0.0f, 0.0f,  0.0f,  0.0f, 44.0f,
    };

    const auto sparse = fromDense(dense, num_rows);
    const auto m = sparse.getView();

    assert(m.shape().first == num_rows);
    assert(m.shape().second == num_rows);

    auto row = m.diagonal;
    assert(row[0] == 11.0f);
    assert(row[1] == 22.0f);
    assert(row[2] == 33.0f);
    assert(row[3] == 44.0f);

    row = m.rowData(0);
    assert(row[0] == 11.0f);
    assert(row.size() == 1);

    auto ind = m.rowIndices(0);
    assert(ind[0] == 0);
    assert(ind.size() == 1);

    row = m.rowData(1);
    assert(row[0] == 21.0f);
    assert(row[1] == 22.0f);
    assert(row.size() == 2);

    ind = m.rowIndices(1);
    assert(ind[0] == 0);
    assert(ind[1] == 1);
    assert(ind.size() == 2);

    row = m.rowData(2);
    assert(row[0] == 32.0f);
    assert(row[1] == 33.0f);
    assert(row.size() == 2);

    ind = m.rowIndices(2);
    assert(ind[0] == 1);
    assert(ind[1] == 2);
    assert(ind.size() == 2);

    row = m.rowData(3);
    assert(row[0] == 44.0f);
    assert(row.size() == 1);

    ind = m.rowIndices(3);
    assert(ind[0] == 3);
    assert(ind.size() == 1);

    const auto td = m.toDense();
    assert(td.size() == dense.size());
    for (auto i = 0; i < td.size(); i++) {
        assert(td[i] == dense[i]);
    }
}

void testBitVector() {
    std::printf("Testing bitVector\n");
    uint32_t len = 100;
    const auto reql = BitVector::requiredLength(len);
    assert(reql == len / 32 + 1);
    std::vector<uint32_t> data(reql, 0);

    BitVector bv(len, std::span(data));
    for (auto i = 0; i < bv.size(); i++) {
        assert(bv[i] == 0u);
    }

    bv.flip(10);
    assert(bv[9] == 0u);
    assert(bv[10] == 1u);
    assert(bv[11] == 0u);
    bv.flip(10);

    for (auto i = 0; i < bv.size(); i++) {
        bv.flip(i);
    }

    // The last may not be entirely ones, as the length of the bit vector might
    // not be divisible by 32
    auto last = bv.span().back();
    assert(last == (~0u) >> (32 - (bv.size() % 32)));
    for (auto v = bv.span().begin(); v != bv.span().end() - 1; v++) {
        assert(*v == ~0u);
    }
}

void testSuperspan() {
    std::printf("Testing superspan\n");

    const uint32_t num_subspans = 20;
    const uint32_t subspan_len = 58;
    const auto reql =
        SuperSpan<double>::requiredLength(num_subspans, subspan_len);
    std::vector<double> data(reql, 0);
    auto ss = SuperSpan<double>(std::span(data), num_subspans, subspan_len);

    assert(ss.subspanLength() == subspan_len);

    for (auto &v : ss.subspan(0)) {
        v = 1.0;
    }

    for (auto &v : ss.subspan(1)) {
        v = 2.0;
    }

    // There's padding between the subspans that should be zero
    for (auto i = 0; i < 128; i++) {
        if (i < subspan_len) {
            assert(data[i] == 1.0);
        } else if (i < 64) {
            assert(data[i] == 0.0);
        } else if (i < subspan_len + 64) {
            assert(data[i] == 2.0);
        } else {
            assert(data[i] == 0.0);
        }
    }
}

void testSuperspan2() {
    std::printf("Testing superspan2\n");

    const uint32_t num_subspans = 5;
    const uint32_t subspan_len = 96;
    const auto reql =
        SuperSpan<double>::requiredLength(num_subspans, subspan_len);
    std::vector<double> data(reql, 0);
    auto ss = SuperSpan<double>(std::span(data), num_subspans, subspan_len);

    assert(ss.subspanLength() == subspan_len);

    for (auto &v : ss.subspan(0)) {
        v = 1.0;
    }

    for (auto &v : ss.subspan(1)) {
        v = 2.0;
    }

    // There shouldn't be any padding between the subspans
    for (auto i = 0; i < 192; i++) {
        if (i < subspan_len) {
            assert(data[i] == 1.0);
        } else {
            assert(data[i] == 2.0);
        }
    }
    assert(data[192] == 0.0);
}

/*
void testMatrixProduct() {
    using T = float;
    constexpr static auto n = 4;
    std::vector<uint32_t> indptr{0, 2, 3, 4, 5};
    std::vector<uint32_t> indices{0, 2, 1, 2, 3};
    std::vector<T> data{11.0f, 13.0f, 22.0f, 33.0f, 44.0f};

    const SparseMatrix<T, CPUMem> mat(indptr, indices, data);

    std::vector<T> x(n, 1.0f);
    std::vector<T> y(n, 0.0f);

    auto mem_x = gpu::allocate(sizeof(T) * x.size());
    auto mem_y = gpu::allocate(sizeof(T) * y.size());

    std::span<T> x_d(static_cast<T *>(mem_x), x.size());
    std::span<T> y_d(static_cast<T *>(mem_y), y.size());

    gpu::memcpy(x_d.data(), x.data(), x_d.size_bytes());
    gpu::memcpy(y_d.data(), y.data(), y_d.size_bytes());

    // LAUNCH_KERNEL(mp, 1, 64, 0, 0, "", mat.view, x_d, y_d);

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
*/

} // namespace testing

int main() {
    testing::testFromToDense();
    testing::testBitVector();
    testing::testSuperspan();
    testing::testSuperspan2();

    // testing::testMatrixProduct();

    // testing::testSparseMatrix();

    // testing::testWarpReduction();

    // gpu::synchronize();

    return 0;
}
