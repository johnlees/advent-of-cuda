CUDAFLAGS +=-std=c++14 -O3 -Xcompiler -Wall -Xptxas -dlcm=ca -L/usr/local/cuda-11.0/lib64 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_75,code=sm_75

DAYS=day1-part1 day1-part2 day2-part1

all: $(DAYS)

day1-part1:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) day1/part1.cu -o $@

day1-part2:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) day1/part2.cu -o $@

day2-part1:
	nvcc $(CUDAFLAGS) $(CPPFLAGS) day2/part1.cu -o $@

clean:
	$(RM) *.o ~* $(DAYS)
