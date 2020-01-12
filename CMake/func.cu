
#include "func.hpp"
#include <cstdio>

__global__
void kernel()
{
	printf("Calling from function_cuda\n");
}

int function_cuda()
{
	kernel<<<1, 5>>>();
	return 0;
}

