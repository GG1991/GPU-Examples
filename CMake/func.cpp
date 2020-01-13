
#include <iostream>
#include "func.hpp"

using namespace std;

CUDA_HOSTDEV
int small(int a) {
	return a + 1;
}

int function()
{
	small(2);
	cout << "Hello World! from function" << endl;
	return 0;
}
