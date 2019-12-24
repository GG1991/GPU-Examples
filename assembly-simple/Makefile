all: main-cpu main-gpu main-cpu-gpu 




main-cpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPU
	#nvcc -g main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPU

main-gpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU
	#nvcc -g main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU

main-cpu-gpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU -DCPU

clean:
	rm -f main-cpu main-gpu main-cpu-gpu 
