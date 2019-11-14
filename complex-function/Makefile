all: gcc-debug gcc-opt

gcc-debug:
	g++ -ggdb -fsanitize=address main.cpp -o main-gcc-debug

gcc-opt:
	g++ -O3 main.cpp -o main-gcc-opt

pgi-cpu:
	pgic++ -O3 main.cpp -o main-pgi-cpu

pgi-gpu:
	pgic++ -acc -O3 main.cpp -o main-pgi-gpu
