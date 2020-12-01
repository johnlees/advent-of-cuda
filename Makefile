CXXFLAGS+=-Wall -Wextra -std=c++14 -O3
CUDAFLAGS +=-Xptxas -dlcm=ca -L/usr/local/cuda-11.0/lib64 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75

day1-part1:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) -x cu -c day1/part1.cu -o $@
