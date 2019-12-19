
#pragma once

#define NPE 8
#define NVOI 6
#define DIM 3

struct CUDA_vars {
	int nex, ney, nez;
	int nx, ny, nz;
	double bmat_cache[NPE][NVOI][NPE * DIM];
};

