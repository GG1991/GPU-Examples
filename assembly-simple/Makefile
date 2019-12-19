all: main-debug main-opt

main-debug: main.cpp ell.cpp assembly.cu
	nvcc main.cpp ell.cpp assembly.cu -o main-debug -I . -std=c++11

main-opt: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o main-opt -I . -std=c++11

clean:
	rm -f main-opt main-debug

#pgi: pgi-cpu pgi-gpu
#
#pgi-cpu:
#	pgc++ -O3 -std=c++11 main.cpp -o main-pgi-cpu
#
#pgi-gpu:
#	pgc++ -acc -O3 -std=c++11 main.cpp -o main-pgi-gpu
