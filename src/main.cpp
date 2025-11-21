#include "json.hpp"
#include "qubo.hpp"
#include <fstream>
#include <hiprand/hiprand_kernel.h>
#include <sstream>

__global__ void setupRandKernel(hiprandState *states, uint32_t gpu_id,
                                uint32_t seed) {
    // Add all threads by earlier GPUs to the subsequence
    const uint32_t subsequence = gpu::threadid() +
                                 gpu::blockid() * gpu::nthreads() +
                                 gpu::nthreads() * gpu::nblocks() * gpu_id;

    const uint32_t tid = gpu::threadid() + gpu::blockid() * gpu::nthreads();
    hiprand_init(seed, subsequence, 0, &states[tid]);
}

template <typename T, typename M>
SparseMatrix<T, M> matrixFromFile(const std::string &fname) {
    auto getLines = [](std::istream &input) {
        std::vector<std::vector<std::string>> lines;

        for (std::string line; std::getline(input, line);) {
            std::vector<std::string> result;
            std::stringstream lineStream(line);

            for (std::string cell; std::getline(lineStream, cell, ',');) {
                result.push_back(cell);
            }
            lines.push_back(result);
        }

        return lines;
    };

    auto readFileToVector = [&getLines](const std::string &fname) {
        if (std::ifstream is{fname}) {
            return getLines(is);
        } else {
            std::printf("Failed to open file '%s'\n", fname.c_str());
        }
        std::exit(EXIT_FAILURE);
    };

    auto lines = readFileToVector(fname);
    const bool dense_format =
        (lines.size() != 3) && (lines.size() == lines[0].size());

    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<float> data;

    if (not dense_format) {
        for (auto str : lines[0]) {
            indptr.push_back(std::stoul(str));
        }

        for (auto str : lines[1]) {
            indices.push_back(std::stoul(str));
        }

        for (auto str : lines[2]) {
            data.push_back(std::stof(str));
        }
    } else {
        const uint32_t num_rows = lines.size();
        const uint32_t num_cols = num_rows;

        indptr.push_back(0);
        for (auto row = 0; row < num_rows; row++) {
            for (auto col = 0; col < num_cols; col++) {
                const auto value = std::stof(lines[row][col]);
                if (value != 0.0) {
                    data.push_back(value);
                    indices.push_back(col);
                }
            }
            const auto end = data.size();
            indptr.push_back(end);
        }
    }

    return SparseMatrix<T, M>(indptr, indices, data);
}

std::pair<nlohmann::json, uint32_t> handleInput(int argc, char **argv) {
    if (argc < 3) {
        std::printf(
            "Give an input.json and gpu ID as input: '%s input.json gpuID'\n",
            argv[0]);
    }
    const auto fname = argv[1];
    nlohmann::json input_json;
    std::fstream file(fname, std::ios::in);

    if (file.is_open()) {
        file >> input_json;
    } else {
        std::exit(EXIT_FAILURE);
    }
    const uint32_t gpu_id = std::stoul(argv[2]);

    return std::make_pair(input_json, gpu_id);
}

void outputCandidate(const std::vector<uint32_t> &candidate,
                     const std::string &fname) {
    std::fstream file(fname, std::ios::out);
    if (file.is_open()) {
        for (auto i = 0; i < candidate.size() - 1; i++) {
            file << candidate[i] << ",";
        }
        file << candidate.back();
    } else {
        std::printf("Could not open '%s' for output\n", fname.c_str());
    }
}

/*
 * TODO:
 * constant from script
 * */
int main(int argc, char **argv) {
    const auto [input, gpu_id] = handleInput(argc, argv);

    static constexpr size_t num_blocks = 1024ul;
    const uint32_t seed = input["seed"].get<uint32_t>();
    const size_t num_to_search = input["num_to_search"].get<size_t>();

    auto mat = matrixFromFile<float, gpu::GPUMem>(
        input["matrix_filename"].get<std::string>());
    const size_t num_threads = computeNumThreads(mat.getView().shape().first);

    std::unique_ptr<hiprandState, decltype(&gpu::free)> memory(
        static_cast<hiprandState *>(
            gpu::allocate(sizeof(hiprandState) * num_threads * num_blocks)),
        gpu::free);
    auto *states = memory.get();

    auto generator = [states](auto *sd) {
        const uint32_t tid = gpu::threadid() + gpu::blockid() * gpu::nblocks();
        hiprandState localState = states[tid];

        const auto n = BitVector::requiredLength(sd->len_data);
        auto data = sd->candidate.data();

        for (uint32_t i = gpu::threadid(); i < n; i += gpu::nthreads()) {
            data[i] = hiprand(&localState);
        }

        states[tid] = localState;
        gpu::syncthreads();
    };

    Qubo qubo(std::move(mat), num_blocks, num_to_search, generator);

    LAUNCH_KERNEL(setupRandKernel, num_blocks, num_threads, 0, 0, "", states,
                  gpu_id, seed);

    qubo.search();

    std::printf("minimum: %f, %f candidates/s\n", qubo.getMinimum(),
                qubo.searchesPerSecond());

    const auto bc = qubo.getBits();
    outputCandidate(bc, input["output_filename"].get<std::string>());

    return 0;
}
