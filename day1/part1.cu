#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cub/device/device_select.cuh>

#include "../cuda.cuh"

const size_t blockSize = 32;

// Functions to convert index position to/from squareform to condensed form
__device__
int calc_row_idx(const int k, const int n) {
	// __ll2float_rn() casts long long to float, rounding to nearest
	return n - 2 - floor(__fsqrt_rn(__int2float_rz(-8*k + 4*n*(n-1)-7))/2 - 0.5);
}

__device__
int calc_col_idx(const int k, const int i, const int n) {
	return k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2;
}

// Square dist matrix kernel which stores sum == 2020 in one array, multiple in another
__global__
void add_and_multiply(int* expenses_d, char* sums, int* prods, size_t length, size_t pairs) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < pairs) {
		int i, j;
		i = calc_row_idx(index, length);
    j = calc_col_idx(index, i, length);
    *(sums + index) = (*(expenses_d + i) + *(expenses_d + j)) == 2020;
    *(prods + index) = *(expenses_d + i) * *(expenses_d + j);
    }
}

int main() {
    std::string line;
    std::ifstream infile("inputs/day1.data");

    // Read input
    std::vector<int> expenses;
    if (infile.is_open()) {
      while (std::getline(infile, line)) {
        expenses.push_back(std::stoi(line));
      }
      infile.close();
    }

    // Copy input to device
    int* expenses_d;
    CUDA_CALL(cudaMalloc((void** )&expenses_d, expenses.size() * sizeof(int)));
    CUDA_CALL(cudaMemcpy(expenses_d, expenses.data(), expenses.size() * sizeof(int),
                         cudaMemcpyDefault));

    // Allocate space to store output
    size_t n_pairs = 0.5 * expenses.size() * (expenses.size() - 1);
    char* sums;
    CUDA_CALL(cudaMalloc((void** )&sums, n_pairs * sizeof(char)));
    CUDA_CALL(cudaMemset(sums, 0, n_pairs * sizeof(char)));
    int* prods;
    CUDA_CALL(cudaMalloc((void** )&prods, n_pairs * sizeof(int)));
    CUDA_CALL(cudaMemset(prods, 0, n_pairs * sizeof(int)));

    // Calculate sums and products
    size_t blockCount = (n_pairs + blockSize - 1) / blockSize;
    add_and_multiply<<<blockCount, blockSize>>>(expenses_d,
                                                sums,
                                                prods,
                                                expenses.size(),
                                                n_pairs);

    // Use device select to get the answer
    int *d_out;
    CUDA_CALL(cudaMalloc((void**)&d_out, n_pairs * sizeof(int)));
    int *d_num_selected_out;
    CUDA_CALL(cudaMalloc((void**)&d_num_selected_out, sizeof(int)));
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Allocate temp storage
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               prods, sums, d_out,
                               d_num_selected_out, n_pairs);
    CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run selection
    cub::DeviceSelect::Flagged(d_temp_storage,
                               temp_storage_bytes,
                               prods, sums, d_out,
                               d_num_selected_out, n_pairs);

    // Get and print answer
    int n_selected;
    CUDA_CALL(cudaMemcpy(&n_selected, d_num_selected_out, sizeof(int),
                        cudaMemcpyDefault));
    std::vector<int> answer(n_selected);
    CUDA_CALL(cudaMemcpy(answer.data(), d_out, n_selected * sizeof(int),
                         cudaMemcpyDefault));
    for (auto it = answer.begin(); it != answer.end(); ++it) {
        std::cout << *it << std::endl;
    }

    // Free device memory
    CUDA_CALL(cudaFree(expenses_d));
    CUDA_CALL(cudaFree(sums));
    CUDA_CALL(cudaFree(prods));
    CUDA_CALL(cudaFree(d_out));
    CUDA_CALL(cudaFree(d_num_selected_out));
    CUDA_CALL(cudaFree(d_temp_storage));

    return 0;
}