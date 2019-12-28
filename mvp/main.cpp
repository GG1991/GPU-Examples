
#include <iostream>
#include <chrono>
#include <bits/stdc++.h>

#include "vars.hpp"
#include "ell.hpp"

using namespace std;
using namespace std::chrono;

#ifdef CUDA
__global__
void ell_mvp_kernel(const ell_matrix *m_d, const double *vals_d,
		    const int *cols_d, const double *x_d, double *y_d);
#endif

int vec_compare(const double *v1, const double *v2, const int n)
{
	// 0 if are equal, 1 if not.

	for (int i = 0; i < n; ++i) {
		if (fabs(v1[i] - v2[i]) > 1.0e-5) return i;
	}
	return 0;
}

#define REPETITIONS 100

void ell_set_values(ell_matrix * m)
{
	for (int row = 0; row < m->nrow; ++row) {
		for (int col = 0; col < m->nnz; ++col) {
			m->vals[row * m->nnz + col] = (col % 2) ? 2 : 3;
		}
	}
}

void arr_set_values(double *x, const int n)
{
	for (int i = 0; i < n; ++i) {
		x[i] = (i % 2) ? 13.6 : 11.1;
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

	const int ns[DIM] = { n, n, n };

#ifdef CPU
	ell_matrix A_cpu;  // Matrix
	ell_init(&A_cpu, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
	ell_set_values(&A_cpu);
	double *x_cpu = (double *)malloc(A_cpu.nrow * sizeof(double));
	double *y_cpu = (double *)malloc(A_cpu.nrow * sizeof(double));
	arr_set_values(x_cpu, A_cpu.nrow);
#endif
#ifdef CUDA
	ell_matrix A_cuda;  // Matrix
	ell_init(&A_cuda, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
	ell_set_values(&A_cuda);
	double *x_cuda = (double *)malloc(A_cuda.nrow * sizeof(double));
	double *y_cuda = (double *)malloc(A_cuda.nrow * sizeof(double));
	arr_set_values(x_cuda, A_cuda.nrow);
#endif
#ifdef OPENACC
	ell_matrix A_acc;  // Matrix
	ell_init(&A_acc, DIM, DIM, ns, CG_ABS_TOL, CG_REL_TOL, CG_MAX_ITS);
	ell_set_values(&A_acc);
	double *x_acc = (double *)malloc(A_acc.nrow * sizeof(double));
	double *y_acc = (double *)malloc(A_acc.nrow * sizeof(double));
	arr_set_values(x_acc, A_acc.nrow);
#endif


	auto time_2 = high_resolution_clock::now();
#ifdef CPU
	cout << "CPU case" << endl;
	for (int i = 0; i < REPETITIONS; ++i) {
		ell_mvp(&A_cpu, x_cpu, y_cpu);
	}
	for (int i = 0; i < 10; ++i)
		cout << y_cpu[i] << " ";
	cout << endl;
#endif

#ifdef CUDA
	cout << "GPU case" << endl;
	ell_matrix *m_d;
	double *vals_d, *x_d, *y_d;
	int *cols_d;
	//const int grid = 1000;
	//const int block = 1024;
	dim3 grid(50000, 1, 1);
	dim3 block(2, 128, 1);
	cudaMalloc((void **)&m_d, sizeof(ell_matrix));
	cudaMalloc((void **)&vals_d, A_cuda.nrow * A_cuda.nnz * sizeof(double));
	cudaMalloc((void **)&cols_d, A_cuda.nrow * A_cuda.nnz * sizeof(int));
	cudaMalloc((void **)&x_d, A_cuda.nrow * sizeof(double));
	cudaMalloc((void **)&y_d, A_cuda.nrow * sizeof(double));

	cudaMemcpy(m_d, &A_cuda, sizeof(ell_matrix), cudaMemcpyHostToDevice);
cudaMemcpy(vals_d, A_cuda.vals, A_cuda.nrow * A_cuda.nnz*sizeof(double),cudaMemcpyHostToDevice);
cudaMemcpy(cols_d, A_cuda.cols, A_cuda.nrow * A_cuda.nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(x_d, x_cuda, A_cuda.nrow *sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < REPETITIONS; ++i) {
		ell_mvp_kernel<<<grid, block>>>(m_d, vals_d, cols_d, x_d, y_d);
	}

	cudaMemcpy(y_cuda, y_d, A_cuda.nrow * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(x_d);
	cudaFree(y_d);
	cudaFree(m_d);
	cudaFree(vals_d);
	cudaFree(cols_d);

	for (int i = 0; i < 10; ++i)
		cout << y_cuda[i] << " ";
	cout << endl;
#endif

#ifdef OPENACC
	cout << "OpenACC case" << endl;
#pragma acc enter data copyin(A_acc)
#pragma acc enter data copyin(A_acc.cols[:A_acc.nrow * A_acc.nnz])
#pragma acc enter data copyin(A_acc.vals[:A_acc.nrow * A_acc.nnz])
#pragma acc enter data copyin(x_acc[:A_acc.nrow], y_acc[:A_acc.nrow])
	for (int i = 0; i < REPETITIONS; ++i) {
		ell_mvp_acc(&A_acc, x_acc, y_acc);
	}
#pragma acc exit data copyout(A_acc.cols[:A_acc.nrow * A_acc.nnz])
#pragma acc exit data copyout(A_acc.vals[:A_acc.nrow * A_acc.nnz])
#pragma acc exit data delete(A_acc)
#pragma acc exit data copyout(y_acc[:A_acc.nrow])
	for (int i = 0; i < 10; ++i)
		cout << y_acc[i] << " ";
	cout << endl;
#endif
	auto time_3 = high_resolution_clock::now();


#ifdef CPU
#ifdef OPENACC
	cout << endl << endl;
	cout << "CPU vs OPENACC" << endl;
	int vec_comp = vec_compare(y_cpu, y_acc, A_cpu.nrow);
	cout << "vec_comp: " << vec_comp << endl;
	for (int i = vec_comp; i < vec_comp + 10; ++i)
		cout << y_cpu[i] << " ";
	cout << endl << endl;
	for (int i = vec_comp; i < vec_comp + 10; ++i)
		cout << y_acc[i] << " ";
	cout << endl;
        assert (!vec_comp);
#endif
#endif
#ifdef CPU
#ifdef CUDA
	cout << endl << endl;
	cout << "CPU vs CUDA" << endl;
	int vec_comp = vec_compare(y_cpu, y_cuda, A_cpu.nrow);
	cout << "vec_comp: " << vec_comp << endl;
	for (int i = vec_comp; i < vec_comp + 10; ++i)
		cout << y_cpu[i] << " ";
	cout << endl << endl;
	for (int i = vec_comp; i < vec_comp + 10; ++i)
		cout << y_cuda[i] << " ";
	cout << endl;
        assert (!vec_comp);
#endif
#endif

#ifdef CPU
	ell_free(&A_cpu);
#endif
#ifdef GPU
	ell_free(&A_cuda);
#endif
#ifdef OPENACC
	ell_free(&A_acc);
#endif

	auto duration = duration_cast<milliseconds>(time_2 - time_1);
	cout << "time_init = " << duration.count() << " ms" << endl;

	duration = duration_cast<milliseconds>(time_3 - time_2);
	cout << "time_solver = " << duration.count() << " ms" << endl;

	return 0;
}
