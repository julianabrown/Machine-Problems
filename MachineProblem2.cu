//Juliana Brown 20010601

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "math.h"
#include "time.h"

#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>

//Machine Problem 2: Matrix Addition


//Part 2: Each thread produces one matrix output element
__global__ void MatrixAddition(float *outMatrix, const float *matrixA, float *matrixB, int dimensions){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < dimensions && col < dimensions)
	{
		int uniqueIdx = row * dimensions + col;
		outMatrix[uniqueIdx] = matrixA[uniqueIdx] + matrixB[uniqueIdx];
	}
}


//Part 3: Each thread produces one matrix output row
__global__ void MatrixAdditionRow(float *outMatrix, const float *matrixA, float *matrixB, int dimensions)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < dimensions)
	{
		int rowStartIdx = row * dimensions;

		for (int colIdx = 0; colIdx < dimensions; ++colIdx)
		{
			int curIdx = rowStartIdx + colIdx;
			outMatrix[curIdx] = matrixA[curIdx] + matrixB[curIdx];
		}
	}
}

//Part 4: Each thread produces one matrix output collumn

__global__ void MatrixAdditionCol(float *outMatrix, const float *matrixA, float *matrixB, int dimensions)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < dimensions)
	{
		for (int rowIdx = 0; rowIdx < dimensions; ++rowIdx)
		{
			int curIdx = rowIdx * dimensions + col;
			outMatrix[curIdx] = matrixA[curIdx] + matrixB[curIdx];
		}
	}
}

//Host Code
int main(int argv, int* argc) {

	int size = 500;
	printf("Enter size of Matrix(M:");
	scanf("%d", &size);

	//16 threads per block - specified in assignment instructions
	dim3 threadsPerBlock(16, 16); 
	dim3 blocksPerGrid((size/ 16), (size/16));

	const size_t d_size = sizeof(float) * size_t(size*size);

	float ms;
	float avems = 0.0;
	cudaEvent_t start, end;

	// Initialize host matrices
	clock_t h_alloctime = clock();
	float *h_matA = (float*)malloc(size*size * sizeof(float));
	float *h_matB = (float*)malloc(size*size * sizeof(float));
	float *h_matC = (float*)malloc(size*size * sizeof(float));
	
	//randomArray(h_matA, h_matB, size);

	// Initialize device matrices
	float *d_A, *d_B, *d_C;

	clock_t d_alloctime = clock();
	cudaMalloc((void **)&d_A, d_size);
	cudaMalloc((void **)&d_B, d_size);
	cudaMalloc((void **)&d_C, d_size);
	
	cudaMemcpy(d_A, h_matA, d_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_matB, d_size, cudaMemcpyHostToDevice);

	for (int i = 0; i<10; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		MatrixAddition << < blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, size);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		avems += ms;
		cudaMemcpy(h_matC, d_C, d_size, cudaMemcpyDeviceToHost);

		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf(" Kernel execution time for addition by element: %.2fms.\n\n", avems / 10.0);


	avems = 0.0;
	for (int i = 0; i<10; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		MatrixAdditionRow << < blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, size);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		avems += ms;
		cudaMemcpy(h_matC, d_C, d_size, cudaMemcpyDeviceToHost);
		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf(" Kernel execution time for addition by row: %.2fms.\n\n", avems / 10.0);

	avems = 0;
	for (int i = 0; i<10; i++) {
		cudaEventCreate(&start);
		cudaEventCreate(&end);
		cudaEventRecord(start, 0);

		MatrixAdditionCol << < blocksPerGrid, threadsPerBlock >> >(d_A, d_B, d_C, size);


		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		cudaEventElapsedTime(&ms, start, end);

		avems += ms;
		cudaMemcpy(h_matC, d_C, d_size, cudaMemcpyDeviceToHost);

		cudaEventDestroy(start);
		cudaEventDestroy(end);
	}
	printf(" Kernel execution time for addition by collumn: %.2fms.\n", avems / 10.0);


	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_matA);
	free(h_matB);
	free(h_matC);

	return 0;
}


