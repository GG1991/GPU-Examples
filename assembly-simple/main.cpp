
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




int main(int argc, char **argv)
{
	auto time_1 = high_resolution_clock::now();

	if (argc < 2) {
		cerr << "Usage: " << argv[0] << " n" << endl;
		return(1);
	}

	const int n = atoi(argv[1]);

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

	double * u = new double[nndim];

	const int ns[DIM] = { nx, ny, nz };

	for (int i = 0; i < NPE; ++i) {
		for (int j = 0; j < NVOI; ++j) {
			for (int k = 0; k < NPE * DIM; ++k) {
				bmat_cache[i][j][k] = 1.0;
			}
		}
	}

	struct CUDA_vars CUDA_vars_h;

	CUDA_vars_h.nex = nex;
	CUDA_vars_h.ney = ney;
	CUDA_vars_h.nez = nez;
	memcpy(&CUDA_vars_h.bmat_cache, bmat_cache,
	       NPE * NVOI * NPE * DIM * sizeof(double));


#ifdef CPU
	ell_matrix A_cpu;  // Matrix
	ell_init(&A_cpu, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
#endif
#ifdef CPUNEW
	ell_matrix A_cpunew;  // Matrix
	ell_init(&A_cpunew, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
#endif
#ifdef GPU
	ell_matrix A_gpu;  // Matrix
	ell_init(&A_gpu, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
#endif

	auto time_2 = high_resolution_clock::now();

#ifdef CPU
	cout << "CPU case" << endl;
	assembly_mat(&A_cpu, u, &CUDA_vars_h);
	for (int i = 0; i < 0 + A_cpu.nnz; ++i)
		cout << A_cpu.vals[i] << " ";
	cout << endl;
#endif
#ifdef CPUNEW
	cout << "CPU NEW case" << endl;
	assembly_mat_new_cpu(&A_cpunew, u, &CUDA_vars_h);
	for (int i = 0; i < 0 + A_cpunew.nnz; ++i)
		cout << A_cpunew.vals[i] << " ";
	cout << endl;
#endif
#ifdef GPU
	cout << "GPU case" << endl;
	assembly_mat_gpu(&A_gpu, u, &CUDA_vars_h); // works fine
	assembly_mat_gpu_2(&A_gpu, u, &CUDA_vars_h);
	for (int i = 0; i < 0 + A_gpu.nnz; ++i)
		cout << A_gpu.vals[i] << " ";
	cout << endl;
#endif

	auto time_3 = high_resolution_clock::now();

#ifdef CPU
#ifdef CPUNEW
	cout << "Evaluating assert CPU vs CPUNEW" << endl;
	cout << " ell_compare: " << ell_compare(&A_cpu, &A_cpunew) << endl;
        assert (!ell_compare(&A_cpu, &A_cpunew));
        //assert (0);
#endif
#endif

#ifdef CPU
#ifdef GPU
	cout << "Evaluating assert CPU vs GPU" << endl;
	cout << " ell_compare: " << ell_compare(&A_cpu, &A_gpu) << endl;
        assert (!ell_compare(&A_cpu, &A_gpu));
        //assert (0);
#endif
#endif

	delete [] u;

#ifdef CPU
	ell_free(&A_cpu);
#endif
#ifdef CPUNEW
	ell_free(&A_cpunew);
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
