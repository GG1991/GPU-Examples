# Complex Function

This example corresponds to a simulation of the assembly of a ELLPACK
format matrix typical of a Finite Element problem. The code fills a
large vector - representing a large matrix - with partial small
vectors sum - representing small matrices.

## Prerequisites

* GCC compiler (> 8.0.0)
* CUDA Toolkit (> 9.0)

## Compile

To compile check that the CUDA wrapper `nvcc` is visible in your
environment.

```
make 
```

## Run

To execute just put the name a positive integer number as argument
(recommended `1 < n < 120`). In a Power9 machine with V100
Nvidia GPU:

```
$ ./main-cpu 100
...
time_init = 384 ms
time_assembly = 16065 ms

$ ./main-gpu 100
...
time_init = 377 ms
time_assembly = 1534 ms
```

## Authors

* **Guido Giuntoli**
