all: mvp-cpu mvp-cuda mvp-acc mvp-cpu-acc mvp-cpu-cuda


mvp-cpu: main.cpp ell.cpp ell.cu
	nvcc -O3 $^ -o $@ -I . -std=c++11 -DCPU 
	#nvcc -g $< -o $@ -I . -std=c++11 -DCPU

mvp-acc: main.cpp ell.cpp 
	pgc++ -O3 $^ -o $@ -I . -std=c++11 -DOPENACC -acc -Minfo=accel
	#pgc++ -g $^ -o $@ -I . -std=c++11 -DOPENACC -acc

mvp-cuda: main.cpp ell.cpp ell.cu
	nvcc -O3 $^ -o $@ -I . -std=c++11 -DCUDA -x cu
	#nvcc -g $< -o $@ -I . -std=c++11 -DCUDA

mvp-cpu-cuda: main.cpp ell.cpp ell.cu
	nvcc -O3 $^ -o $@ -I . -std=c++11 -DCPU -DCUDA -x cu

mvp-cpu-acc: main.cpp ell.cpp 
	pgc++ -O3 $^ -o $@ -I . -std=c++11 -DCPU -DOPENACC -acc -Minfo=accel


clean:
	rm -f mvp-cpu mvp-cuda mvp-acc mvp-cpu-cuda mvp-cpu-acc  
