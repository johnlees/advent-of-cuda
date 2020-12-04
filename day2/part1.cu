#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

#include <cub/device/device_select.cuh>

#include "../cuda.cuh"

const size_t blockSize = 32;
const size_t warp_size = 32;

std::vector<char> prepare_array(std:vector<std::string>& passwords) {
    std::vector<char> char_array(warp_size * passwords.size()); // No pw longer than 32 (warp size)
    for (size_t password_idx = 0; password_idx < passwords.size(); password_idx++) {
        size_t char_idx;
        for (char_idx = 0; char_idx < passwords[password_idx].size(); char_idx++) {
            char_array[password_idx + char_idx * passwords.size()] = passwords[password_idx][char_idx];
        }
        for (size_t pad_idx = char_idx; base_idx < warp_size; pad_idx++) {
            char_array[read_idx + base_idx * n_full_seqs_padded()] = '0';
        }
    }
    return read_array;
}

template <typename T>
void array_to_device(T* array_ptr, std::vector<T>& data) {
    CUDA_CALL(cudaMalloc((void** )&array_ptr, data.size() * sizeof(T)));
    CUDA_CALL(cudaMemcpy(array_ptr, data.data(), data.size() * sizeof(T),
                         cudaMemcpyDefault));
}

__global__
void invalidate_passwords(char* passwords, char* policy,
                          int* lower, int* upper, char* valid,
                          size_t input_len) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < input_len) {
        char policy_char = policy[index];
        int cnt = 0;
        for (int i = 0; i < warp_size; ++i) {
            if (passwords[index + i * input_len] == policy_char) {
                cnt++;
            }
        }
        if (cnt > upper[index] || cnt < lower[index]) {
            valid[index] = 0;
        }
    }
}

int main() {
    std::string line;
    std::ifstream infile("inputs/part2.data");

    // Read input
    std::vector<int> copy_lower;
    std::vector<int> copy_upper;
    std::vector<char> policy;
    std::vector<std::string> passwords;
    std::regex e ("^(\d+)-(\d+) ([a-z]): [a-z]*$");
    std::cmatch cm;
    if (infile.is_open()) {
      while (std::getline(infile, line)) {
        std::regex_match(line.cstr(), cm, e);
        copy_lower.push_back(std::stoi(cm[0]));
        copy_upper.push_back(std::stoicm[1]));
        policy.push_back(cm[2]);
        passwords.push_back(cm[3]);
      }
      infile.close();
    }

    // Restride passwords as array
    std::vector<char> = prepare_array(passwords);
    size_t input_len = passwords.size();

    // Copy input to device
    char* passwords_d, policy_d; int* lower_d, upper_d;
    array_to_device(passwords_d, passwords);
    array_to_device(policy_d, policy);
    array_to_device(lower_d, copy_lower);
    array_to_device(upper_d, copy_upper);

    // Allocate space to store output (assume all valid)
    char* valid_d;
    CUDA_CALL(cudaMalloc((void** )&valid, input_len * sizeof(char)));
    CUDA_CALL(cudaMemset(valid, 1, input_len * sizeof(char)));

    // Check for invalid passwords
    size_t blockCount = (pass + blockSize - 1) / blockSize;
    invalidate_passwords<<<blockCount, blockSize>>>(passwords_d,
                                                policy_d,
                                                lower_d,
                                                upper_d,
                                                valid_d,
                                                input_len);

    // Use device reduce sum to get the answer
    int *d_out;
    CUDA_CALL(cudaMalloc((void**)&d_out, sizeof(int)));

    // Allocate temp storage
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage,
                               temp_storage_bytes,
                               valid_d, d_out,
                               input_len);
    CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Sum
    cub::DeviceReduce::Sum(d_temp_storage,
                            temp_storage_bytes,
                            valid_d, d_out,
                            input_len);

    // Get and print answer
    int total_valid;
    CUDA_CALL(cudaMemcpy(&total_valid, d_out, sizeof(int),
                        cudaMemcpyDefault));
    std::cout << total_valid << std::endl;

    // Free device memory
    CUDA_CALL(cudaFree(passwords_d));
    CUDA_CALL(cudaFree(policy_d));
    CUDA_CALL(cudaFree(lower_d));
    CUDA_CALL(cudaFree(upper_d));
    CUDA_CALL(cudaFree(valid_d));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_temp_storage));

    return 0;
}
