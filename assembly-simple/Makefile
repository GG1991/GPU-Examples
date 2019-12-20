all: main-cpu main-cpunew main-gpu main-cpu-gpu main-cpu-cpunew 




main-cpu: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPU
	#nvcc -g main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPU

main-cpunew: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPUNEW
	#nvcc -g main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPUNEW

main-gpu: main.cpp ell.cpp assembly.cu
	#nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU
	nvcc -g main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU




main-cpu-cpunew: main.cpp ell.cpp assembly.cu
	nvcc -O3 main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DCPU -DCPUNEW

main-cpu-gpu: main.cpp ell.cpp assembly.cu
	nvcc main.cpp ell.cpp assembly.cu -o $@ -I . -std=c++11 -DGPU -DCPU

clean:
	rm -f main-cpu main-cpunew main-gpu main-cpu-gpu main-cpu-cpunew
