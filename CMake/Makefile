all: main-cpu main-cuda

main-cpu:
	echo "Compiling for CPU"
	g++ -c main.cpp -o main.o
	g++ -c func.cpp -o func.o
	g++ main.o func.o -o main-cpu -lcudart
	echo

main-cuda:
	echo "Compiling for CUDA"
	nvcc -dc -x cu -c main.cpp -o main.cpp.o -DCUDA
	nvcc -dc -x cu -c func.cu -o func.cu.o -DCUDA
	nvcc -dc -x cu -c func.cpp -o func.cpp.o -DCUDA
	nvcc main.cpp.o func.cu.o func.cpp.o -o main-cuda -lcudart -lcuda
	echo

clean:
	rm -f main-cpu main-cuda *.o
