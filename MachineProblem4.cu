//Juliana Brown 20010601
//Machine Problem 4 Tiled Matrix Multiplication (Square Matrix)
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
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define TILE_WIDTH 2

__global__ void tiledMatrixMult(float* M, float* N, float* P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; 
	int by = blockIdx.y;
	int tx = threadIdx.x; 
	int ty = threadIdx.y;
	int Row = by * TILE_WIDTH + ty; // Identify the row index and column index
	int Col = bx * TILE_WIDTH + tx; // of the P element to work on
	float Pvalue = 0;
	
	// Loop over the M and N tiles required to compute the P element
	for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
		// Collaborative loading of M and N tiles into shared memory
		Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
		Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];
		
		__syncthreads();
		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];
		__synchthreads();
		
	}
	P[Row*Width + Col] = Pvalue; // All threads write to their P element
	
}


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

//fill matrix with random values
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

	int temp = size / TILE_WIDTH + (size % TILE_WIDTH == 0 ? 0 : 1);
	size = temp * TILE_WIDTH;

	size_t pt_size = size * size * sizeof(float);

	*Lmatrix = (float *)malloc(pt_size);
	*Rmatrix = (float *)malloc(pt_size);

	memset(*Lmatrix, 0, pt_size);
	memset(*Rmatrix, 0, pt_size);

	for (int i = 0; i < LdimX; i++) {
		for (int j = 0; j < LdimY; j++) {
			int temp = size * i + j;
			(*Lmatrix)[temp] = sinf(temp);
		}
	}
	for (int i = 0; i < RdimX; i++) {
		for (int j = 0; j < RdimY; j++) {
			int temp = size * i + j;
			(*Rmatrix)[temp] = cosf(temp);
		}
	}
	return size;
}

int main(void)
{
	//size of the vectors to be processed  and matrix dimensions
	int Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y;

	float *Left_Vector_h, *Right_Vector_h, *Left_Vector_d, *Right_Vector_d, *Res_h, *Res_d, *CPU; 

	printf("Enter m n n k :\n");

	scanf("%d %d %d %d", &Left_matrix_x, &Left_matrix_y, &Right_matrix_x, &Right_matrix_y);

	int dim = fill(&Left_Vector_h, &Right_Vector_h, Left_matrix_x, Left_matrix_y, Right_matrix_x, Right_matrix_y);

	size_t vector_size;
	vector_size = dim*dim * sizeof(float);

	Res_h = (float *)malloc(vector_size); 
	CPU = (float *)malloc(vector_size);

	cudaMalloc((void **)&Left_Vector_d, vector_size);  
	cudaMalloc((void **)&Right_Vector_d, vector_size);   
	cudaMalloc((void **)&Res_d, vector_size);     

	cudaMemcpy(Left_Vector_d, Left_Vector_h, vector_size, cudaMemcpyHostToDevice);
	cudaMemcpy(Right_Vector_d, Right_Vector_h, vector_size, cudaMemcpyHostToDevice);

	//as specified in lecture slides
	dim3 Block_dim(TILE_WIDTH, TILE_WIDTH);
	dim3 Grid_dim(dim / TILE_WIDTH, dim / TILE_WIDTH);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//kernel function 
	tiledMatrixMult << < Grid_dim, Block_dim >> > (Left_Vector_d, Right_Vector_d, Res_d, dim);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float et;
	cudaEventElapsedTime(&et, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(Res_h, Res_d, vector_size, cudaMemcpyDeviceToHost);

	clock_t begin = clock();

	//matrix multiplication on cpu
	cpuMultiplication(Left_Vector_h, Right_Vector_h, CPU, dim);

	clock_t end = clock();
	double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	printf("GPU time= %f ms\n", et);
	printf("CPU time= %lf ms\n", time_spent);
	
	cudaDeviceProp dp;
	printf("  Total amount of shared memory per block: %zu bytes\n",
		dp.sharedMemPerBlock);
	
	

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
