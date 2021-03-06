//Juliana Brown 20010601
//Machine Problem 3: Matrix Multiplication

#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "time.h"
#include <iostream>
#include <fstream>
#include <iomanip>


#define BLOCK_SIZE 4

//CPU matrix multiplication A and B are multiplied to give C
__host__ void cpuMultiplication(float *A, float *B, float *C, int dim) {
	for (int i = 0; i <dim; ++i)
	{
		for (int j = 0; j < dim; ++j)
		{
			float tmp = 0.0;
			for (int h = 0; h < dim; ++h)
			{
				tmp += A[i * dim + h] * B[h * dim + j];
			}
			C[i * dim + j] = tmp;
		}
	}
}

//fills matrix with random variables 
__host__ int fill(float **Lmatrix, float **Rmatrix, int LdimX, int LdimY, int RdimX, int RdimY) {

	int sqr_dim_X, sqr_dim_Y, size;

	sqr_dim_X = RdimX;
	if (LdimX > RdimX) {
		sqr_dim_X = LdimX;
	}

	sqr_dim_Y = RdimY;
	if (LdimY > RdimY) {
		sqr_dim_Y = LdimY;
	}

	size = sqr_dim_Y;
	if (sqr_dim_X > sqr_dim_Y) {
		size = sqr_dim_X;
	}

	int temp = size / BLOCK_SIZE + (size % BLOCK_SIZE == 0 ? 0 : 1);
	size = temp * BLOCK_SIZE;

	size_t pt_size = size * size * sizeof(float);

	*Lmatrix = (float *)malloc(pt_size);
	*Rmatrix = (float *)malloc(pt_size);

	memset(*Lmatrix, 0, pt_size);
	memset(*Rmatrix, 0, pt_size);

	for (int i = 0; i < LdimX; i++) {
		for (int j = 0; j < LdimY; j++) {
			int dummy = size * i + j;
			(*Lmatrix)[dummy] = sinf(dummy);
		}
	}
	for (int i = 0; i < RdimX; i++) {
		for (int j = 0; j < RdimY; j++) {
			int dummy = size * i + j;
			(*Rmatrix)[dummy] = cosf(dummy);
		}
	}
	return size;
}

//Matrix multiplications: based on multiplication example in 
__global__ void multiply(float *M, float *N, float *out, int dim) {

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	if (row < dim && col < dim)
	{
		float outVal = 0;

		// each thread computes one element of the block sub-matrix
		for (int k = 0; k < dim; ++k)
			outVal += M[row*dim + k] * N[k*dim + col];
		out[row*dim + col] = outVal;
	}
}


// main routine that executes on the host
int main(void)
{
	//size of the vectors to be processed  and matrix dimensions
	int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y, Left_vector_size, Right_vector_size;

	float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU;  // Pointer to host & device arrays

	printf("Enter m n n k :\n");

	scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y); // input matrix dimensions are taken

	int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y);

	size_t vector_size;
	vector_size = dim*dim * sizeof(float);

	Res_h = (float *)malloc(vector_size); // Allocate array on host for result
	CPU = (float *)malloc(vector_size);// Allocate array on host for CPU_matrix_multiplication result

	cudaMalloc((void **)&Left_Vector_d, vector_size);     // Allocate array on device for LHS operand
	cudaMalloc((void **)&Right_Vector_d, vector_size);   // Allocate array on device for RHS operand but this is vector 1xN
	cudaMalloc((void **)&Res_d, vector_size);     // Allocate array on device for result

	clock_t beg = clock();

	cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice);

	clock_t done = clock();
	double timeHD = (double)1000 * (done - beg) / CLOCKS_PER_SEC;

	printf("Host to Device Time: %f ms\n", timeHD);


	dim3 Block_dim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 Grid_dim(dim / BLOCK_SIZE, dim / BLOCK_SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//kernel function 
	multiply << < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Retrieve result from device and store it in host array
	clock_t beg2 = clock();

	cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

	clock_t done2 = clock();
	double timeDH = (double)1000 * (done2 - beg2) / CLOCKS_PER_SEC;
	printf("Device to Host Time: %f ms\n", timeDH);

	clock_t begin = clock();

	//matrix multiplication on cpu
	cpuMultiplication(Left_Vector_h, Right_Vector_h, CPU, dim);

	clock_t end = clock();
	double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	printf("GPU time= %f ms\n", et);
	printf("CPU time= %lf ms\n", time_spent);


	//check for equal answers
	bool success = true;
	for (int i = 0; i< Left_matrix_x && success; i++) {
		for (int j = 0; j < Right_matrix_y && success; j++) {
			if (abs(Res_h[i*dim + j] - CPU[i*dim + j]) > 0.001)
			{
				success = false;
				printf("NOT EQUAL\n");
			}
		}
	}
	if (success)
	{
		std::cout << "Test Passed!" << std::endl;
	}
	else
	{
		std::cout << "Test not Passed!" << std::endl;
	}


	free(Left_Vector_h);
	free(Right_Vector_h);
	free(Res_h);
	free(CPU);
	cudaFree(Left_Vector_d);
	cudaFree(Right_Vector_d);
	cudaFree(Res_d);
}
