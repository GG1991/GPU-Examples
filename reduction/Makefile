all: reduction

reduction: main.cu
	nvcc -O3 main.cu -o $@ -std=c++11

clean:
	rm -f reduction
