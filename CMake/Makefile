all: main-cpu main-cuda

main-cpu:
	g++ -c main.cpp -o main.o
	g++ -c func.cpp -o func.o
	g++ main.o func.o -o main-cpu

main-cuda:
	g++ -c main.cpp -o main-cu.o -DCUDA
	nvcc -x cu -c func.cu -o func-cu.o
	g++ main-cu.o func-cu.o -o main-cuda -lcudart -lcuda

clean:
	rm -f main-cpu main-cuda *.o
