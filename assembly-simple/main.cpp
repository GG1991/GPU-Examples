
#include <iostream>
#include <chrono>
#include <bits/stdc++.h>

#include "vars.hpp"
#include "ell.hpp"

using namespace std;
using namespace std::chrono;

int nex, ney, nez;
int nx, ny, nz;

double bmat_cache[NPE][NVOI][NPE * DIM];

struct CUDA_vars CUDA_vars_h;

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
		int ex, int ey, int ez)
{
	double elem_disp[NPE * DIM];
	get_elem_displ(u, elem_disp, nx, ny, nz, ex, ey, ez);

	for (int i = 0; i < NVOI; ++i) {
		strain_gp[i] = 0;
	}

	for (int v = 0; v < NVOI; ++v) {
		for (int i = 0; i < NPE * DIM; ++i){
			strain_gp[v] += bmat_cache[gp][v][i] * elem_disp[i];
		}
	}
}


void get_elem_mat(const double *u, double Ae[NPE * DIM * NPE * DIM],
		  int ex, int ey, int ez)
{
	const double wg = 0.25;
	double ctan[NVOI][NVOI];
	constexpr int npedim = NPE * DIM;
	constexpr int npedim2 = npedim * npedim;

	double TAe[npedim2] = { 0.0 };

	for (int gp = 0; gp < NPE; ++gp) {

		double eps[6];
		get_strain(u, gp, eps, ex, ey, ez);

		get_ctan(eps, (double *)ctan, nullptr);

		double cxb[NVOI][npedim];

		for (int i = 0; i < NVOI; ++i) {
			for (int j = 0; j < npedim; ++j) {
				double tmp = 0.0;
				for (int k = 0; k < NVOI; ++k)
					tmp += ctan[i][k] * bmat_cache[gp][k][j];
				cxb[i][j] = tmp * wg;
			}
		}

		for (int m = 0; m < NVOI; ++m) {
			for (int i = 0; i < npedim; ++i) {
				const int inpedim = i * npedim;
				const double bmatmi = bmat_cache[gp][m][i];
				for (int j = 0; j < npedim; ++j)
					TAe[inpedim + j] += bmatmi * cxb[m][j];
			}
		}
	}
	memcpy(Ae, TAe, npedim2 * sizeof(double));
}


void assembly_mat(ell_matrix *A, const double *u)
{
	ell_set_zero_mat(A);

	double Ae[NPE * DIM * NPE * DIM];
	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				get_elem_mat(u, Ae, ex, ey, ez);
				ell_add_3D(A, ex, ey, ez, Ae);
			}
		}
	}
	ell_set_bc_3D(A);
}


int main(int argc, char **argv)
{
	auto time_1 = high_resolution_clock::now();

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " n" << endl;
		return(1);
	}

	const int n = atoi(argv[1]);

	//const int nex = n;
	//const int ney = n;
	//const int nez = n;
	nex = n;
	ney = n;
	nez = n;

	const int nx = nex + 1;
	const int ny = ney + 1;
	const int nz = nez + 1;
	const int nxny = nx * ny;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nndim = nx * ny * nz * DIM;

	CUDA_vars_h.nex = nex;
	CUDA_vars_h.ney = ney;
	CUDA_vars_h.nez = nez;

	double * u = new double[nndim];

	ell_matrix A;  // Matrix
	const int ns[DIM] = { nx, ny, nz };
	ell_init(&A, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);

	auto time_2 = high_resolution_clock::now();

	//assembly_mat(&A, u);
	assembly_mat_gpu(&A, u);

	auto time_3 = high_resolution_clock::now();

	delete [] u;
	ell_free(&A);

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "time_init = " << duration.count() << " ms" << endl;

	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "time_assembly = " << duration.count() << " ms" << endl;

	return 0;
}
