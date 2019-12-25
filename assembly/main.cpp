
#include <iostream>
#include <chrono>
#include <bits/stdc++.h>

#include "vars.hpp"
#include "ell.hpp"

using namespace std;
using namespace std::chrono;

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

	const int nx = nex + 1;
	const int ny = ney + 1;
	const int nz = nez + 1;
	const int nxny = nx * ny;
	const int npedim = NPE * DIM;
	const int npedim2 = npedim * npedim;
	const int nndim = nx * ny * nz * DIM;

	double * u = new double[nndim];

	const int ns[DIM] = { nx, ny, nz };

	struct Params params_h;

	for (int i = 0; i < NPE; ++i) {
		for (int j = 0; j < NVOI; ++j) {
			for (int k = 0; k < NPE * DIM; ++k) {
				params_h.bmat_cache[i][j][k] = 1.0;
			}
		}
	}

	params_h.nex = nex;
	params_h.ney = ney;
	params_h.nez = nez;

#ifdef CPU
	ell_matrix A_cpu;  // Matrix
	ell_init(&A_cpu, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
#endif
#ifdef GPU
	ell_matrix A_gpu;  // Matrix
	ell_init(&A_gpu, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
#endif

	auto time_2 = high_resolution_clock::now();

#ifdef CPU
	cout << "CPU case" << endl;
	assembly_mat(&A_cpu, u, &params_h);
	for (int i = 0; i < 0 + A_cpu.nnz; ++i)
		cout << A_cpu.vals[i] << " ";
	cout << endl;
#endif

#ifdef GPU
	cout << "GPU case" << endl;
	assembly_mat_gpu(&A_gpu, u, &params_h);
	for (int i = 0; i < 0 + A_gpu.nnz; ++i)
		cout << A_gpu.vals[i] << " ";
	cout << endl;
#endif

	auto time_3 = high_resolution_clock::now();

#ifdef CPU
#ifdef GPU
	cout << endl << endl;
	cout << "Evaluating assert CPU vs GPU" << endl;
	int ell_comp = ell_compare(&A_cpu, &A_gpu);
	cout << " ell_compare: " << ell_comp << endl;
	for (int i = ell_comp; i < ell_comp + A_gpu.nnz; ++i)
		cout << A_cpu.vals[i] << " ";
	cout << endl << endl;
	for (int i = ell_comp; i < ell_comp + A_gpu.nnz; ++i)
		cout << A_gpu.vals[i] << " ";
	cout << endl;
        assert (!ell_compare(&A_cpu, &A_gpu));
#endif
#endif

	delete [] u;

#ifdef CPU
	ell_free(&A_cpu);
#endif
#ifdef GPU
	ell_free(&A_gpu);
#endif

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "time_init = " << duration.count() << " ms" << endl;

	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "time_assembly = " << duration.count() << " ms" << endl;

	return 0;
}
