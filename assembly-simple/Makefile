all: main-cpu main-gpu

main-cpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o main-cpu -I . -std=c++11

main-gpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o main-gpu -I . -std=c++11 -DGPU

clean:
	rm -f main-opt main-debug

#pgi: pgi-cpu pgi-gpu
#
#pgi-cpu:
#	pgc++ -O3 -std=c++11 main.cpp -o main-pgi-cpu
#
#pgi-gpu:
#	pgc++ -acc -O3 -std=c++11 main.cpp -o main-pgi-gpu
