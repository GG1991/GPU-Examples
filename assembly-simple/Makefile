all: gcc-debug gcc-opt

gcc: gcc-debug gcc-opt

gcc-debug:
	g++ -ggdb -fsanitize=address main.cpp ell.cpp -o main-gcc-debug -I .

gcc-opt:
	g++ -O3 main.cpp ell.cpp -o main-gcc-opt -I .

#pgi: pgi-cpu pgi-gpu
#
#pgi-cpu:
#	pgc++ -O3 -std=c++11 main.cpp -o main-pgi-cpu
#
#pgi-gpu:
#	pgc++ -acc -O3 -std=c++11 main.cpp -o main-pgi-gpu
