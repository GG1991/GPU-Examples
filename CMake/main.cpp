#include "func.hpp"

int main()
{
	/* 
	 * <function> can be normal CPU 
	 * or CUDA implementation 
	 */
#ifdef CUDA
	function_cuda(); 
#else
	function(); 
#endif
	return 0;
}
