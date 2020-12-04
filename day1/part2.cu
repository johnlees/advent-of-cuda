#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <cub/device/device_select.cuh>

#include "../cuda.cuh"

const size_t blockSize = 32;

// Square dist matrix kernel which stores sum == 2020 in one array, multiple in another
__global__
void add_and_multiply(int* expenses_d, char* sums, int* prods, size_t length, size_t triples) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
  int count = 0;
  int i, j, k;
  // This is horrible
  if (index < triples) {
    for (i = 0; i < length - 2; ++i) {
      for (j = i + 1; j < length - 1; ++j) {
        for (k = j + 1; k < length; k++) {
          if (count == index) {
            break;
          } else {
            count++;
          }
        }
        if (count == index) {
          break;
        }
      }
      if (count == index) {
        break;
      }
    }
    __syncwarp();

    *(sums + index) = (*(expenses_d + i) + *(expenses_d + j) + *(expenses_d + k)) == 2020;
    *(prods + index) = *(expenses_d + i) * *(expenses_d + j) * *(expenses_d + k);
    //printf("idx:%d i:%d j:%d k:%d val1:%d val2:%d val3:%d sum:%d prod:%d\n",
    // index, i, j, k, *(expenses_d + i), *(expenses_d + j), *(expenses_d + k), *(sums + index), *(prods + index));
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
  // nC3
  size_t n_triples = expenses.size() * (expenses.size() - 1) * (expenses.size() - 2) / 6;
  char* sums;
  CUDA_CALL(cudaMalloc((void** )&sums, n_triples * sizeof(char)));
  CUDA_CALL(cudaMemset(sums, 0, n_triples * sizeof(char)));
  int* prods;
  CUDA_CALL(cudaMalloc((void** )&prods, n_triples * sizeof(int)));
  CUDA_CALL(cudaMemset(prods, 0, n_triples * sizeof(int)));

  // Calculate sums and products
  size_t blockCount = (n_triples + blockSize - 1) / blockSize;
  add_and_multiply<<<blockCount, blockSize>>>(expenses_d,
                                              sums,
                                              prods,
                                              expenses.size(),
                                              n_triples);

  // Use device select to get the answer
  int *d_out;
  CUDA_CALL(cudaMalloc((void**)&d_out, n_triples * sizeof(int)));
  int *d_num_selected_out;
  CUDA_CALL(cudaMalloc((void**)&d_num_selected_out, sizeof(int)));
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Allocate temp storage
  cub::DeviceSelect::Flagged(d_temp_storage,
                              temp_storage_bytes,
                              prods, sums, d_out,
                              d_num_selected_out, n_triples);
  CUDA_CALL(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  // Run selection
  cub::DeviceSelect::Flagged(d_temp_storage,
                              temp_storage_bytes,
                              prods, sums, d_out,
                              d_num_selected_out, n_triples);

  // Get and print answer
  int n_selected;
  CUDA_CALL(cudaMemcpy(&n_selected, d_num_selected_out, sizeof(int),
                      cudaMemcpyDefault));
  std::vector<int> answer(n_selected);
  CUDA_CALL(cudaMemcpy(answer.data(), d_out, n_selected * sizeof(int),
                        cudaMemcpyDefault));
  for (auto it = answer.begin(); it != answer.end(); ++it) {
    if (*it > 1) {
      std::cout << *it << std::endl;
    }
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