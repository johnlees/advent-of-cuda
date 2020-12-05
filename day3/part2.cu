#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>
#include <cstdint>

#include <cub/cub.cuh>

#include "../cuda.cuh"

const size_t blockSize = 32;

struct CustomProd
{
    template <typename T, typename U>
    __device__ __forceinline__
    U operator()(const T &a, const U &b) const {
        T prod = a * b;
        return prod;
    }
};

std::vector<char> prepare_array(std::vector<std::string>& forest) {
    std::vector<char> char_array(forest.front().size() * forest.size());
    for (size_t row_idx = 0; row_idx < forest.size(); row_idx++) {
        for (size_t tree_idx = 0; tree_idx < forest.front().size(); tree_idx++) {
            char_array[row_idx + tree_idx * forest.size()] = forest[row_idx][tree_idx];
        }
    }
    return char_array;
}

template <typename T>
void array_to_device(T*& array_ptr, std::vector<T>& data) {
    CUDA_CALL(cudaMalloc((void** )&array_ptr, data.size() * sizeof(T)));
    CUDA_CALL(cudaMemcpy(array_ptr, data.data(), data.size() * sizeof(T),
                         cudaMemcpyDefault));
}

__global__
void count_trees(char* forest, unsigned int* trees,
                 size_t length, size_t width,
                 int* width_skips, int* length_skips,
                 size_t n_routes) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < (length * n_routes)) {
        int route_idx = __float2int_rz(__fdividef(index, length) + 0.001f);
        int row_idx = index % length;
        if (row_idx % length_skips[route_idx] == 0) { // Loading width/length skips into __shared__ would be faster
            int col = ((row_idx / length_skips[route_idx]) * width_skips[route_idx]) % width;
            if (forest[row_idx + col * length] == '#') {
                atomicInc(trees + route_idx, length);
            }
        }
    }
    __syncwarp();
}

int main() {
    cudaDeviceReset();

    std::string line;
    std::ifstream infile("inputs/day3.data");

    // Read input
    std::vector<std::string> forest;
    if (infile.is_open()) {
      while (std::getline(infile, line)) {
        forest.push_back(line);
      }
      infile.close();
    }

    // Restride passwords as array
    std::vector<char> forest_strided = prepare_array(forest);
    size_t input_len = forest.size();
    size_t input_width = forest.front().size();

    // Set routes
    std::vector<int> down = {1, 1, 1, 1, 2};
    std::vector<int> across = {1, 3, 5, 7, 1};
    size_t n_routes = down.size();

    // Copy input to device
    char* forest_d = nullptr;
    array_to_device(forest_d, forest_strided);
    int* down_d = nullptr;
    int* across_d = nullptr;
    array_to_device(down_d, down);
    array_to_device(across_d, across);

    // Allocate space to store output (assume all valid)
    unsigned int* trees_d;
    CUDA_CALL(cudaMalloc((void** )&trees_d, n_routes * sizeof(unsigned int)));
    CUDA_CALL(cudaMemset(trees_d, 0, n_routes * sizeof(unsigned int)));

    // Check for trees
    size_t blockCount = ((input_len * n_routes) + blockSize - 1) / blockSize;
    count_trees<<<blockCount, blockSize>>>(forest_d,
                                           trees_d,
                                           input_len,
                                           input_width,
                                           across_d,
                                           down_d,
                                           n_routes);
    cudaDeviceSynchronize();

    std::vector<unsigned int> tree_cnt(n_routes);
    CUDA_CALL(cudaMemcpy(tree_cnt.data(), trees_d, n_routes * sizeof(unsigned int), cudaMemcpyDefault));
    for (auto it = tree_cnt.begin(); it != tree_cnt.end(); ++it) {
        std::cout << *it << std::endl;
    }

    // Use device reduce prod to get the answer
    uint64_t *d_out;
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(uint64_t)));

    // Allocate temp storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    CustomProd prod_op;
    cub::DeviceReduce::Reduce(d_temp_storage,
                           temp_storage_bytes,
                           trees_d, d_out,
                           n_routes,
                           prod_op,
                           1);
    CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceReduce::Reduce(d_temp_storage,
        temp_storage_bytes,
        trees_d, d_out,
        n_routes,
        prod_op,
        1);

    // Free device memory
    CUDA_CALL(cudaFree(forest_d));
    CUDA_CALL(cudaFree(trees_d));
    CUDA_CALL(cudaFree(down_d));
    CUDA_CALL(cudaFree(across_d));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_temp_storage));

    return 0;
}
