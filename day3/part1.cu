#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

#include <cub/cub.cuh>

#include "../cuda.cuh"

const size_t blockSize = 32;

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
void count_trees(char* forest, char* trees,
                 size_t length, size_t width) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < length) {
        int col = (index * 3) % width;
        if (forest[index + col * length] == '#') {
            trees[index] = 1;
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

    // Copy input to device
    char* forest_d = nullptr;
    array_to_device(forest_d, forest_strided);

    // Allocate space to store output (assume all valid)
    char* trees_d;
    CUDA_CALL(cudaMalloc((void** )&trees_d, input_len * sizeof(char)));
    CUDA_CALL(cudaMemset(trees_d, 0, input_len * sizeof(char)));

    // Check for invalid passwords
    size_t blockCount = (input_len + blockSize - 1) / blockSize;
    count_trees<<<blockCount, blockSize>>>(forest_d,
                                           trees_d,
                                           input_len,
                                           input_width);
    cudaDeviceSynchronize();

    // Use device reduce sum to get the answer
    int *d_out;
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(int)));

    // Allocate temp storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes,
                           trees_d, d_out,
                           input_len);
    CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sum
    cub::DeviceReduce::Sum(d_temp_storage,
                           temp_storage_bytes,
                           trees_d, d_out,
                           input_len);

    // Get and print answer
    int total_trees;
    CUDA_CALL(cudaMemcpy(&total_trees, d_out, sizeof(int),
                        cudaMemcpyDefault));
    std::cout << total_trees << std::endl;

    // Free device memory
    CUDA_CALL(cudaFree(forest_d));
    CUDA_CALL(cudaFree(trees_d));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_temp_storage));

    return 0;
}
