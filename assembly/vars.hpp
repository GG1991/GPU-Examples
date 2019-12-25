
#pragma once

#include "ell.hpp"

#define NPE 8
#define NVOI 6
#define NVOI2 (NVOI * NVOI)
#define DIM 3
#define NPEDIM (NPE * DIM)
#define NPEDIM2 (NPEDIM * NPEDIM)

#define glo_elem(ex,ey,ez)   ((ez) * (nex) * (ney) + (ey) * (nex) + (ex))

struct CUDA_vars {

	int nex, ney, nez;
	int nx, ny, nz;
	double bmat_cache[NPE][NVOI][NPE * DIM];

};

void assembly_mat(ell_matrix *A, const double *u, CUDA_vars *CUDA_vars_h);

void assembly_mat_gpu(ell_matrix *A, const double *u, CUDA_vars *CUDA_vars_h);

