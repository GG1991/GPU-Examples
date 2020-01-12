#pragma once

#ifdef CUDA
int function_cuda();
#else
int function();
#endif
