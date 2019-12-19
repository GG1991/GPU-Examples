all: main-cpu main-gpu main-cpugpu

main-cpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o main-cpu -I . -std=c++11 -DCPU

main-gpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o main-gpu -I . -std=c++11 -DGPU

main-cpugpu: main.cpp ell.cpp assembly.cu
	nvcc main.cpp ell.cpp assembly.cu -o main-cpugpu -I . -std=c++11 -DGPU -DCPU

clean:
	rm -f main-cpu main-gpu main-cpugpu
