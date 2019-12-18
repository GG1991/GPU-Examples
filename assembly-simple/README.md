# Complex Function

This example corresponds to a simulation of the assembly of a ELLPACK format
matrix typical of a Finite Element problem. Basically the code fills a large
vector - that represents a large matrix - adding partial sums of small vectors -
which represent small matrices.

## Prerequisites

* GCC compiler (> 8.0.0)
* PGI compiler (> 19.4)

## Compile

To compile for GCC check that `g++` is visible in your environment.

```
make gcc-debug
make gcc-opt
make gcc  (to compile both)
```

To compile for PGI CPU and GPU versiones check that `pgc++` is visible in your environment.

```
make pgi-cpu
make pgi-gpu
make pgi
```

## Compilation and Run

To execute just put the name a positive integer number as argument (recommended
`1 < n < 120`). For example in a Power9 machine with V100 Nvidia GPU:

```
$ ./main-pgi-cpu 100
time_init = 3 ms
time_assembly = 21175 ms

$ ./main-pgi-gpu 100
time_init = 3 ms
time_assembly = 3750 ms
```

## Authors

* **Guido Giuntoli**
