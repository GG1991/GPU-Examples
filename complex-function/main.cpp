
#include <iostream>
#include <chrono>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

const int npe = 8;
const int ngp = 8;
const int nvoi = 6;
const int dim = 3;

double bmat_cache[ngp][nvoi][npe * dim];

//#pragma acc routine seq
void get_elem_nodes(int n[npe], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	const int nxny = ny * nx;
	const int n0 = ez * nxny + ey * nx + ex;
	n[0] = n0;
	n[1] = n0 + 1;
	n[2] = n0 + nx + 1;
	n[3] = n0 + nx;

	if (dim == 3) {
		n[4] = n[0] + nxny;
		n[5] = n[1] + nxny;
		n[6] = n[2] + nxny;
		n[7] = n[3] + nxny;
	}
}

//#pragma acc routine seq
void get_elem_displ(const double *u, double elem_disp[npe * dim], const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	int n[npe];
	get_elem_nodes(n, nx, ny, nz, ex, ey, ez);

	for (int i = 0 ; i < npe; ++i) {
		for (int d = 0; d < dim; ++d) {
			elem_disp[i * dim + d] = u[n[i] * dim + d];
		}
	}
}

//#pragma acc routine seq
void get_strain(const double *u, int gp, double *strain_gp, const int nx, const int ny, const int nz, int ex, int ey, int ez)
{
	double elem_disp[npe * dim];
	get_elem_displ(u, elem_disp, nx, ny, nz, ex, ey, ez);

	for (int i = 0; i < nvoi; ++i) {
		strain_gp[i] = 0;
	}

	for (int v = 0; v < nvoi; ++v) {
		for (int i = 0; i < npe * dim; ++i){
			strain_gp[v] += bmat_cache[gp][v][i] * elem_disp[i];
		}
	}
}

int main(int argc, char **argv)
{
	auto time_1 = high_resolution_clock::now();

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " n" << endl;
		return(1);
	}

	const int n = atoi(argv[1]);

	const int nex = n;
	const int ney = n;
	const int nez = n;

	const int nx = nex - 1;
	const int ny = ney - 1;
	const int nz = nez - 1;
	const int nxny = nx * ny;

	const int nndim = nx * ny * nz * dim;

	for (int gp = 0; gp < ngp; ++gp) {
		for (int v = 0; v < nvoi; ++v) {
			for (int i = 0; i < npe; ++i) {
				bmat_cache[gp][v][i] = 1.0;
			}
		}
	}

	double* eps = new double[nex * ney * nez * npe * nvoi];
	double* u = new double[nndim];

	for (int i = 0; i < nndim; ++i) {
		u[i] = 1.0;
	}

	auto time_2 = high_resolution_clock::now();

	//#pragma acc parallel loop copy(eps[:nex*ney*nez*npe*6]) copyin(u[:nndim])
	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				for (int gp = 0; gp < npe; ++gp) {
					get_strain(u, gp, &eps[ex*ney*nez*npe*nvoi+ey*nez*npe*nvoi+ez*npe*nvoi+gp*nvoi], nx, ny, nz, ex, ey, ez);
				}
			}
		}
	}

	double *ctan = new double[nex * ney * nez * npe * nvoi * nvoi];
	int *ix_glo = new int[nex * ney * nez * npe];

	for (int ex = 0; ex < nex; ++ex) {
		for (int ey = 0; ey < ney; ++ey) {
			for (int ez = 0; ez < nez; ++ez) {
				const int e = glo_elem(ex, ey, ez);
				const material_t *material = get_material(e);
				for (int gp = 0; gp < npe; ++gp) {
					const double *vars = (vars_old) ? &vars_old[intvar_ix(e, gp, 0)] : nullptr;
					material->get_ctan(&eps[ex*ney*nez*npe*6+ey*nez*npe*6+ez*npe*6+gp*6],
						&ctan[ex*ney*nez*npe*nvoi*nvoi+ey*nez*npe*nvoi*nvoi+ez*npe*nvoi*nvoi+gp*nvoi*nvoi],
						vars);
				}
				const int n0 = ez * nxny + ey * nx + ex;
				const int n1 = n0 + 1;
				const int n2 = n0 + nx + 1;
				const int n3 = n0 + nx;

				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 0] = n0;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 1] = n1;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 2] = n2;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 3] = n3;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 4] = n0 + nxny;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 5] = n1 + nxny;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 6] = n2 + nxny;
				ix_glo[ex*ney*nez*8 + ey*nez*8 + ez*8 + 7] = n3 + nxny;
			}
		}
	}

//#pragma acc parallel loop gang vector copyin(ctan[:nex*ney*nez*npe*nvoi*nvoi], \
//					     bmat[:npe*nvoi*npedim], \
//					     A[:1], A->nrow, A->nnz, cols_row[:8][:8],\
//					     ix_glo[:nex*ney*nez*8]) \
//	copy(A->vals[:A->nrow * A->nnz])
	for (int ex = 0; ex < nex*ney*nez; ++ex) {

		double Ae[npedim2];
		for(int i = 0; i < npedim2; ++i){
			Ae[i] = 0;
		}

		for (int gp = 0; gp < npe; ++gp) {
			double cxb[nvoi][npedim];
			for (int i = 0; i < nvoi; ++i) {
				for (int j = 0; j < npedim; ++j) {
					double tmp = 0.0;
					for (int k = 0; k < nvoi; ++k){
						tmp += ctan[ex*npe*nvoi*nvoi+gp*nvoi*nvoi+i*nvoi+k] * bmat[gp*nvoi*npedim+k*npedim+j];
					}
					cxb[i][j] = tmp * wg;
				}
			}
			for (int m = 0; m < nvoi; ++m) {
				for (int i = 0; i < npedim; ++i) {
					const int inpedim = i * npedim;
					const double bmatmi = bmat[gp*nvoi*npedim+m*npedim+i];
					for (int j = 0; j < npedim; ++j){
						Ae[inpedim + j] += bmatmi * cxb[m][j];
					}
				}
			}
		}

		for (int fi = 0; fi < nfield; ++fi){
			for (int fj = 0; fj < nfield; ++fj){
				for (int i = 0; i < npe; ++i){
					for (int j = 0; j < npe; ++j){
//#pragma acc atomic update
						A->vals[ix_glo[ex*8+i] * nnz_nfield + cols_row[i][j] * nfield + fi * nnz + fj] += Ae[i * npe_nfield2 + fi * npe_nfield + j * nfield + fj];
					}
				}
			}
		}
	}

	auto time_3 = high_resolution_clock::now();

	delete eps;
	delete u;

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "time_init = " << duration.count() << " ms" << endl;

	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "time_init = " << duration.count() << " ms" << endl;

	return 0;
}
