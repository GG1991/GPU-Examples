#include <iostream>

using namespace std;

class material_t {

	protected:
	double a;

	public:
	__host__ __device__ virtual double get_a() const = 0;
	__host__ __device__ virtual void set_a() = 0;
};


class material_1_t : public material_t {

	public:
	__host__ __device__ double get_a() const { return a; };
	__host__ __device__ void set_a() { a = 1.0; };

};


class material_2_t : public material_t {

	public:
	__host__ __device__ double get_a() const { return a; };
	__host__ __device__ void set_a() { a = 2.0; };

};


__device__ material_t *material_d;

__global__
void kernel(double *a_d)
{
	material_d = new material_1_t();

	material_d->set_a();
	*a_d = material_d->get_a(); 
}


int main()
{
	material_t *material_1_h = new material_1_t();
	material_t *material_1_d;
	double *a_d, a_h;

	material_1_h->set_a();
 
	cudaMalloc((void **)&material_1_d, sizeof(material_1_t));
	cudaMalloc((void **)&a_d, sizeof(double));
	cudaMemcpy(material_1_d, material_1_h, sizeof(material_1_t), cudaMemcpyHostToDevice);

	kernel<<<1, 1>>>(a_d);

	cudaMemcpy(&a_h, a_d, sizeof(double), cudaMemcpyDeviceToHost);

	cout << "a_h = " << material_1_h->get_a() << endl;
	cout << "a_d = " << a_h << endl;

	cudaFree(material_1_d);
	cudaFree(a_d);

	delete material_1_h;
	return 0;
}
