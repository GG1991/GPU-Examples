
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

	auto time_3 = high_resolution_clock::now();

    delete eps;
    delete u;

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "time_init = " << duration.count() << " ms" << endl;

	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "time_init = " << duration.count() << " ms" << endl;

    return 0;
}
