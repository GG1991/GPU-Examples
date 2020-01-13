#pragma once

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

#ifdef CUDA
int function_cuda();
#else
int function();
#endif

CUDA_HOSTDEV int small(int a);
