
#include "ell.hpp"
#include "vars.hpp"

#include "cuda_profiler_api.h"

#define cudaCheckError() { \
         cudaError_t e=cudaGetLastError(); \
	 if(e!=cudaSuccess) { \
		    printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
		    exit(0); \
		  } \
}

#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

#ifdef CPU
void get_ctan(const double *eps, double *ctan, const double *history_params)
{
	const double lambda = 1.0e1;
	const double mu = 1.3e5;

	memset(ctan, 0, 6 * 6 * sizeof(double));

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			ctan[i * 6 + j] += lambda;

	for (int i = 0; i < 3; ++i)
		ctan[i * 6 + i] += 2 * mu;

	for (int i = 3; i < 6; ++i)
		ctan[i * 6 + i] = mu;
}


void get_elem_nodes(int n[NPE], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	const int nxny = ny * nx;
	const int n0 = ez * nxny + ey * nx + ex;
	n[0] = n0;
	n[1] = n0 + 1;
	n[2] = n0 + nx + 1;
	n[3] = n0 + nx;

	if (DIM == 3) {
		n[4] = n[0] + nxny;
		n[5] = n[1] + nxny;
		n[6] = n[2] + nxny;
		n[7] = n[3] + nxny;
	}
}


void get_elem_displ(const double *u, double elem_disp[NPE * DIM], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	int n[NPE];
	get_elem_nodes(n, nx, ny, nz, ex, ey, ez);

	for (int i = 0 ; i < NPE; ++i) {
		for (int d = 0; d < DIM; ++d) {
			elem_disp[i * DIM + d] = u[n[i] * DIM + d];
		}
	}
}


void get_strain(const double *u, int gp, double *strain_gp,
	       	CUDA_vars *CUDA_vars_h,
		int ex, int ey, int ez)
{
	double elem_disp[NPE * DIM];
	const int nx = CUDA_vars_h->nx;
	const int ny = CUDA_vars_h->ny;
	const int nz = CUDA_vars_h->nz;
	get_elem_displ(u, elem_disp, nx, ny, nz, ex, ey, ez);

	for (int i = 0; i < NVOI; ++i) {
		strain_gp[i] = 0;
	}

	for (int v = 0; v < NVOI; ++v) {
		for (int i = 0; i < NPE * DIM; ++i){
			strain_gp[v] += CUDA_vars_h->bmat_cache[gp][v][i] * elem_disp[i];
		}
	}
}


void get_elem_mat(const double *u, double Ae[NPE * DIM * NPE * DIM],
	       	  CUDA_vars *CUDA_vars_h,
		  int ex, int ey, int ez)
{
	const double wg = 0.25;
	double ctan[NVOI][NVOI];
	constexpr int npedim = NPE * DIM;
	constexpr int npedim2 = npedim * npedim;

	double TAe[npedim2] = { 0.0 };

	for (int gp = 0; gp < NPE; ++gp) {

		double eps[6];
		get_strain(u, gp, eps, CUDA_vars_h, ex, ey, ez);

		get_ctan(eps, (double *)ctan, nullptr);

		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i][k] * CUDA_vars_h->bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_h->bmat_cache[gp][m][i];
				for (int j = 0; j < npedim; ++j)
					TAe[inpedim + j] += bmatmi * cxb[m][j];
			}
		}
	}
	memcpy(Ae, TAe, npedim2 * sizeof(double));
}

void get_elem_mats_cpu(double *Ae_arr, const double *ctan_arr,
		       CUDA_vars *CUDA_vars_h)
{
	const double wg = 0.25;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nex = CUDA_vars_h->nex;
	const int ney = CUDA_vars_h->ney;
	const int nez = CUDA_vars_h->nez;

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
	double TAe[npedim2] = { 0.0 };
	for (int gp = 0; gp < NPE; ++gp) {

const double *ctan = &ctan_arr[glo_elem(ex,ey,ez) * NPE * NVOI2
			+ gp * NVOI2];
		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i * NVOI + k] 
						* CUDA_vars_h->bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_h->bmat_cache[gp][m][i];
				for (int j = 0; j < npedim; ++j)
					TAe[inpedim + j] += bmatmi * cxb[m][j];
			}
		}
	}
	double *Ae = &Ae_arr[glo_elem(ex,ey,ez) * NPEDIM2];
	memcpy(Ae, TAe, NPEDIM2 * sizeof(double));
			}
		}
	}
}


void assembly_mat(ell_matrix *A, const double *u, CUDA_vars *CUDA_vars_h)
{
	ell_set_zero_mat(A);

	const int nex = CUDA_vars_h->nex;
	const int ney = CUDA_vars_h->ney;
	const int nez = CUDA_vars_h->nez;

	double Ae[NPE * DIM * NPE * DIM];
	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				get_elem_mat(u, Ae, CUDA_vars_h, ex, ey, ez);
				ell_add_3D(A, ex, ey, ez, Ae);
			}
		}
	}
	//ell_set_bc_3D(A);
}
#endif


#ifdef GPU
__device__
void get_ctan_d(const double *eps, double *ctan, const double *history_params)
{
	const double lambda = 1.0e1;
	const double mu = 1.3e5;

	memset(ctan, 0, 6 * 6 * sizeof(double));

	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			ctan[i * 6 + j] += lambda;

	for (int i = 0; i < 3; ++i)
		ctan[i * 6 + i] += 2 * mu;

	for (int i = 3; i < 6; ++i)
		ctan[i * 6 + i] = mu;
}


__device__
void get_elem_nodes_d(int n[NPE], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	const int nxny = ny * nx;
	const int n0 = ez * nxny + ey * nx + ex;
	n[0] = n0;
	n[1] = n0 + 1;
	n[2] = n0 + nx + 1;
	n[3] = n0 + nx;

	if (DIM == 3) {
		n[4] = n[0] + nxny;
		n[5] = n[1] + nxny;
		n[6] = n[2] + nxny;
		n[7] = n[3] + nxny;
	}
}


__device__
void get_elem_displ_d(const double *u, double elem_disp[NPE * DIM], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	int n[NPE];
	get_elem_nodes_d(n, nx, ny, nz, ex, ey, ez);

	for (int i = 0 ; i < NPE; ++i) {
		for (int d = 0; d < DIM; ++d) {
			elem_disp[i * DIM + d] = u[n[i] * DIM + d];
		}
	}
}


__device__
void get_strain_d(const double *u, int gp, double *strain_gp,
	       	CUDA_vars *CUDA_vars_d,
		int ex, int ey, int ez)
{
	const int nx = CUDA_vars_d->nx;
	const int ny = CUDA_vars_d->ny;
	const int nz = CUDA_vars_d->nz;
	double elem_disp[NPE * DIM];
	get_elem_displ_d(u, elem_disp, nx, ny, nz, ex, ey, ez);

	for (int i = 0; i < NVOI; ++i) {
		strain_gp[i] = 0;
	}

	for (int v = 0; v < NVOI; ++v) {
		for (int i = 0; i < NPE * DIM; ++i){
			strain_gp[v] += CUDA_vars_d->bmat_cache[gp][v][i] * elem_disp[i];
		}
	}
}

__device__
void ell_add_3D_gpu(ell_matrix *m, double *vals_d, int ex, int ey, int ez, const double Ae[NPEDIM2])
{
	// assembly Ae in 3D structured grid representation
	// nFields : number of scalar components on each node

	const int nx = m->n[0];
	const int ny = m->n[1];
	const int nfield = m->nfield;
	const int npe = 8;
	const int nnz = m->nnz;
	const int cols_row[8][8] = {
		{ 13, 14, 17, 16, 22, 23, 26, 25 },
		{ 12, 13, 16, 15, 21, 22, 25, 24 },
		{ 9,  10, 13, 12, 18, 19, 22, 21 },
		{ 10, 11, 14, 13, 19, 20, 23, 22 },
		{ 4,  5,  8,  7,  13, 14, 17, 16 },
		{ 3,  4,  7,  6,  12, 13, 16, 15 },
		{ 0,  1,  4,  3,  9,  10, 13, 12 },
		{ 1,  2,  5,  4,  10, 11, 14, 13 } };

	const int nxny = nx * ny;
	const int n0 = ez * nxny + ey * nx + ex;
	const int n1 = n0 + 1;
	const int n2 = n0 + nx + 1;
	const int n3 = n0 + nx;

	const int ix_glo[8] = {	n0, n1, n2, n3,
		n0 + nxny,
		n1 + nxny,
		n2 + nxny,
		n3 + nxny };

	const int nnz_nfield = nfield * nnz;
	const int npe_nfield = npe * nfield;
	const int npe_nfield2 = npe * nfield * nfield;

	for (int fi = 0; fi < nfield; ++fi)
		for (int fj = 0; fj < nfield; ++fj)
			for (int i = 0; i < npe; ++i)
				for (int j = 0; j < npe; ++j){
					vals_d[ix_glo[i] * nnz_nfield + cols_row[i][j] * nfield + fi * nnz + fj] +=
						Ae[i * npe_nfield2 + fi * npe_nfield + j * nfield + fj];
					__syncthreads();
				}

}


__global__
void get_elem_mats_gpu(double *Ae_arr, const double *u,
		       	 CUDA_vars *CUDA_vars_d)
{
	const double wg = 0.25;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nex = CUDA_vars_d->nex;
	const int ney = CUDA_vars_d->ney;
	const int nez = CUDA_vars_d->nez;

	int ex_t = threadIdx.x + blockDim.x * blockIdx.x;
	int ey_t = threadIdx.y + blockDim.y * blockIdx.y;
	int ez_t = threadIdx.z + blockDim.z * blockIdx.z;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	int stride_z = blockDim.z * gridDim.z;

	for (int ex = ex_t; ex < nex; ex += stride_x) {
		for (int ey = ey_t; ey < ney; ey += stride_y) {
			for (int ez = ez_t; ez < nez; ez += stride_z) {
	double TAe[npedim2] = { 0.0 };
	for (int gp = 0; gp < NPE; ++gp) {

		double eps[6];
		double ctan[NVOI2];
		get_strain_d(u, gp, eps, CUDA_vars_d, ex, ey, ez);
		get_ctan_d(eps, ctan, nullptr);
		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i * NVOI + k] 
						* CUDA_vars_d->bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_d->bmat_cache[gp][m][i];
				for (int j = 0; j < npedim; ++j)
					TAe[inpedim + j] += bmatmi * cxb[m][j];
			}
		}
	}
	double *Ae = &Ae_arr[glo_elem(ex,ey,ez) * NPEDIM2];
	memcpy(Ae, TAe, npedim2 * sizeof(double));
			}
		}
	}
}

__global__
void assembly_elem_mats_gpu(ell_matrix *A_d, double *vals_d, const double *u,
		       	 CUDA_vars *CUDA_vars_d)
{
	const double wg = 0.25;
	const int npedim = NPE * DIM;
	const int nex = CUDA_vars_d->nex;
	const int ney = CUDA_vars_d->ney;
	const int nez = CUDA_vars_d->nez;

	int ex_t = threadIdx.x + blockDim.x * blockIdx.x;
	int ey_t = threadIdx.y + blockDim.y * blockIdx.y;
	int ez_t = threadIdx.z + blockDim.z * blockIdx.z;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	int stride_z = blockDim.z * gridDim.z;

	for (int ex = ex_t; ex < nex; ex += stride_x) {
		for (int ey = ey_t; ey < ney; ey += stride_y) {
			for (int ez = ez_t; ez < nez; ez += stride_z) {
	double TAe[NPEDIM2] = { 0.0 };
	for (int gp = 0; gp < NPE; ++gp) {

		double eps[6];
		double ctan[NVOI2];
		get_strain_d(u, gp, eps, CUDA_vars_d, ex, ey, ez);
		get_ctan_d(eps, ctan, nullptr);
		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i * NVOI + k] 
						* CUDA_vars_d->bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_d->bmat_cache[gp][m][i];
				for (int j = 0; j < npedim; ++j)
					TAe[inpedim + j] += bmatmi * cxb[m][j];
			}
		}
	}
	//double *Ae = &Ae_arr[glo_elem(ex,ey,ez) * NPEDIM2];
	//memcpy(Ae, TAe, npedim2 * sizeof(double));
				ell_add_3D_gpu(A_d, vals_d, ex, ey, ez, TAe);
			}
		}
	}
}

void assembly_mat_gpu(ell_matrix *A, const double *u,
		      CUDA_vars *CUDA_vars_h)
{
	auto time_1 = high_resolution_clock::now();
	cudaProfilerStart();
	ell_set_zero_mat(A);
	CUDA_vars *CUDA_vars_d;

	cudaMalloc((void **)&CUDA_vars_d, sizeof(CUDA_vars));
	cudaMemcpy(CUDA_vars_d, CUDA_vars_h, sizeof(CUDA_vars),
		   cudaMemcpyHostToDevice);

	const int nex = CUDA_vars_h->nex;
	const int ney = CUDA_vars_h->ney;
	const int nez = CUDA_vars_h->nez;
	const int ne = nex * ney * nez;
	const int nx = CUDA_vars_h->nx;
	const int ny = CUDA_vars_h->ny;
	const int nz = CUDA_vars_h->nz;
	const int nn = nx * ny * nz;

	double *Ae_arr = new double[ne * NPEDIM2];

	double *u_d;
	double *Ae_arr_d;

	cudaMalloc((void**)&Ae_arr_d, ne * NPEDIM2 * sizeof(double));
	cudaMalloc((void**)&u_d, nn * DIM * sizeof(double));
	cudaMemcpy(u_d, u,
		   nn * DIM * sizeof(double), 
		   cudaMemcpyHostToDevice);

	dim3 grid(15, 15, 15);
	dim3 block(4, 4, 4);
	get_elem_mats_gpu<<<grid, block>>>(Ae_arr_d, u_d, 
					   CUDA_vars_h);
        cudaCheckError();
	cudaMemcpy(Ae_arr, Ae_arr_d, 
		   ne * NPEDIM2 * sizeof(double),
		   cudaMemcpyDeviceToHost);

	cudaFree(Ae_arr_d);
	cudaFree(u_d);
	cudaFree(CUDA_vars_d);

	auto time_2 = high_resolution_clock::now();

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
	double *Ae = &Ae_arr[glo_elem(ex,ey,ez) * NPEDIM2];
				ell_add_3D(A, ex, ey, ez, Ae);
			}
		}
	}

	delete [] Ae_arr;

	//ell_set_bc_3D(A);
	cudaProfilerStop();
	auto time_3 = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "assembly 1 = " << duration.count() << " ms" << endl;
	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "assembly 2 = " << duration.count() << " ms" << endl;
}

void assembly_mat_gpu_2(ell_matrix *A, const double *u,
		        CUDA_vars *CUDA_vars_h)
{
	auto time_1 = high_resolution_clock::now();
	cudaProfilerStart();
	ell_set_zero_mat(A);
	CUDA_vars *CUDA_vars_d;

	cudaMalloc((void **)&CUDA_vars_d, sizeof(CUDA_vars));
	cudaMemcpy(CUDA_vars_d, CUDA_vars_h, sizeof(CUDA_vars),
		   cudaMemcpyHostToDevice);

	const int nx = CUDA_vars_h->nx;
	const int ny = CUDA_vars_h->ny;
	const int nz = CUDA_vars_h->nz;
	const int nn = nx * ny * nz;

	double *u_d;
	double *vals_d;
	ell_matrix *A_d;

	cudaMalloc((void**)&A_d, sizeof(ell_matrix));
	cudaMalloc((void**)&vals_d, A->nnz * A->nrow * sizeof(double));
	cudaMalloc((void**)&u_d, nn * DIM * sizeof(double));
	cudaMemcpy(u_d, u,
		   nn * DIM * sizeof(double), 
		   cudaMemcpyHostToDevice);
	cudaMemcpy(A_d, A,
		   sizeof(ell_matrix), 
		   cudaMemcpyHostToDevice);
	cudaMemset(vals_d, 0, A->nnz * A->nrow * sizeof(double));

	dim3 grid(15, 15, 15);
	dim3 block(4, 4, 4);
	assembly_elem_mats_gpu<<<grid, block>>>(A_d, vals_d, u_d, 
					   CUDA_vars_h);
        cudaCheckError();
	cudaMemcpy(A->vals, vals_d, 
		   A->nrow * A->nnz * sizeof(double),
		   cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(vals_d);
	cudaFree(u_d);
	cudaFree(CUDA_vars_d);

	auto time_2 = high_resolution_clock::now();

	//ell_set_bc_3D(A);
	cudaProfilerStop();
	auto time_3 = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "assembly 1 = " << duration.count() << " ms" << endl;
	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "assembly 2 = " << duration.count() << " ms" << endl;
}
#endif

#ifdef CPU
void assembly_mat_new_cpu(ell_matrix *A, const double *u, 
			  CUDA_vars *CUDA_vars_h)
{
	ell_set_zero_mat(A);

	const int nex = CUDA_vars_h->nex;
	const int ney = CUDA_vars_h->ney;
	const int nez = CUDA_vars_h->nez;
	const int ne = nex * ney * nez;

	double *ctan_arr = new double[ne * NPE * NVOI2];

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				for (int gp = 0; gp < NPE; ++gp) {
					double eps[6];
	double *ctan = &ctan_arr[glo_elem(ex,ey,ez) * NPE * NVOI2 
		+ gp * NVOI2];
					get_strain(u, gp, eps, CUDA_vars_h, ex, ey, ez);
					get_ctan(eps, ctan, nullptr);
				}
			}
		}
	}

	double *Ae_arr = new double[ne * NPEDIM2];

	get_elem_mats_cpu(Ae_arr, ctan_arr, CUDA_vars_h);

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
	double *Ae = &Ae_arr[glo_elem(ex,ey,ez) * NPEDIM2];
				ell_add_3D(A, ex, ey, ez, Ae);
			}
		}
	}

	delete [] ctan_arr;
	delete [] Ae_arr;

	//ell_set_bc_3D(A);
}
#endif
