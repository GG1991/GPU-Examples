
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


void assembly_mat_gpu(ell_matrix *A, const double *u)
{
	ell_set_zero_mat(A);
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

	//get_elem_mats_d<<<1, 1>>>(u, Ae, ex, ey, ez);
	get_elem_mats_cpu(Ae_arr, ctan_arr); 

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
