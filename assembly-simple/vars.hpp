
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

extern struct CUDA_vars CUDA_vars_h;

void get_strain(const double *u, int gp, double *strain_gp,
		int ex, int ey, int ez);

void get_elem_displ(const double *u, double elem_disp[NPE * DIM], const int nx, const int ny, const int nz, int ex, int ey, int ez);

void get_elem_nodes(int n[NPE], const int nx, const int ny, const int nz, int ex, int ey, int ez);

void get_ctan(const double *eps, double *ctan, const double *history_params);

void assembly_mat_new_cpu(ell_matrix *A, const double *u);
void assembly_mat_new_gpu(ell_matrix *A, const double *u);
