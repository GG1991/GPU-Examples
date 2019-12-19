
#include "ell.hpp"
#include "vars.hpp"

__device__ struct CUDA_vars CUDA_vars_d;


void get_elem_mats_cpu(double *Ae_arr, const double *ctan_arr)
{
	const double wg = 0.25;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nex = CUDA_vars_h.nex;
	const int ney = CUDA_vars_h.ney;
	const int nez = CUDA_vars_h.nez;

	double TAe[npedim2] = { 0.0 };

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
	for (int gp = 0; gp < NPE; ++gp) {

		const double *ctan = &ctan_arr[glo_elem(ex,ey,ez) * NPE * NVOI2
			+ gp * NVOI2];
		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i * NVOI + k] 
						* CUDA_vars_h.bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_h.bmat_cache[gp][m][i];
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
void get_elem_mats_gpu(double *Ae_arr, const double *ctan_arr)
{
	const double wg = 0.25;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nex = CUDA_vars_d.nex;
	const int ney = CUDA_vars_d.ney;
	const int nez = CUDA_vars_d.nez;

	double TAe[npedim2] = { 0.0 };

	int ex_t = threadIdx.x + blockDim.x * blockIdx.x;
	int ey_t = threadIdx.y + blockDim.y * blockIdx.y;
	int ez_t = threadIdx.z + blockDim.z * blockIdx.z;
	int stride_x = blockDim.x * gridDim.x;
	int stride_y = blockDim.y * gridDim.y;
	int stride_z = blockDim.z * gridDim.z;

	for (int ex = ex_t; ex < nex; ex += stride_x) {
		for (int ey = ey_t; ey < ney; ey += stride_y) {
			for (int ez = ez_t; ez < nez; ez += stride_z) {
	for (int gp = 0; gp < NPE; ++gp) {

		const double *ctan = &ctan_arr[glo_elem(ex,ey,ez) * NPE * NVOI2
			+ gp * NVOI2];
		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i * NVOI + k] 
						* CUDA_vars_d.bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = CUDA_vars_d.bmat_cache[gp][m][i];
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


void assembly_mat_gpu(ell_matrix *A, const double *u)
{
	ell_set_zero_mat(A);

	cudaMemcpy(&CUDA_vars_h, &CUDA_vars_d,
		       	sizeof(CUDA_vars), cudaMemcpyHostToDevice);

	const int nex = CUDA_vars_h.nex;
	const int ney = CUDA_vars_h.ney;
	const int nez = CUDA_vars_h.nez;
	const int ne = nex * ney * nez;

	double *ctan_arr = new double[ne * NPE * NVOI2];

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				for (int gp = 0; gp < NPE; ++gp) {
					double eps[6];
	double *ctan = &ctan_arr[glo_elem(ex,ey,ez) * NPE * NVOI2 + gp * NVOI2];
					get_strain(u, gp, eps, ex, ey, ez);
					get_ctan(eps, ctan, nullptr);
				}
			}
		}
	}

	double *Ae_arr = new double[ne * NPEDIM2];

	double *Ae_arr_d;
	cudaMalloc(&Ae_arr_d, ne * NPEDIM2 * sizeof(double));

	dim3 grid(4,4,4);
	dim3 block(64,64,64);
	//get_elem_mats_gpu<<<grid, block>>>(Ae_arr_d, ctan_arr);
	get_elem_mats_cpu(Ae_arr_d, ctan_arr);

	//cudaMemcpy(&Ae_arr, &Ae_arr_d, 
	//		ne * NPEDIM2 * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(Ae_arr_d);

	double Ae[NPE * DIM * NPE * DIM];
	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				ell_add_3D(A, ex, ey, ez, Ae);
			}
		}
	}

	delete [] ctan_arr;
	delete [] Ae_arr;

	ell_set_bc_3D(A);
}
