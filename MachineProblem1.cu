//Juliana Brown 
//Student Number: 20010601

#include "cuda_runtime.h"
#include <iostream>
#include <memory>
#include <string>
#include <cuda.h>
#include <stdio.h>

//Machine Problem 1: Code identifies number and type of CUDA devices on GPU servers,
//clock rate, streaming multiprocessors, cores, warp sizes ect.


int main(int argc, char **argv) {
	
	// number of GPU devices the support CUDA
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0) {
		printf("There are no devices that support CUDA\n");
	}
	else {
		printf("Detected %d CUDA Capable devices\n", deviceCount);
	}

	int dev; 

	for (dev = 0; dev < deviceCount; ++dev) {
		cudaSetDevice(dev);
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, dev);

		//device type
		printf("\ nDevice Name %d: \"%s\"\n", dev, dp.name);

		//clock rate
		printf(
			"  GPU Max Clock rate: %.0f MHz (%0.2f GHz)\n",
			dp.clockRate * 1e-3f, dp.clockRate * 1e-6f);
		
		//number of streaming multiprocessors & Cores

		printf("  Multiprocessors: %2d \n", dp.multiProcessorCount);

		//number of cores
		//printf(" CUDA Cores %d \n"
			//_ConvertSMVer2Cores(dp.major, dp.minor) *dp.multiProcessorCount);
		int cores = 0; 
		int mp = dp.multiProcessorCount;

		switch (dp.major) {
		case 2:
			if (dp.minor == 1) cores = mp * 48;
			else cores = mp * 32; 
			break;
		case 3: 
			cores = mp * 192;
			break;
		case 5: // Maxwell
			cores = mp * 128;
			break;
		case 6: // Pascal
			if ((dp.minor == 1) || (dp.minor == 2)) cores = mp * 128;
			else if (dp.minor == 0) cores = mp * 64;
			break;
		case 7: // Volta and Turing
			if ((dp.minor == 0) || (dp.minor == 5)) cores = mp * 64;
			break;
		
		}
	printf("  Number of cores is: %d\n", cores);

		//warp size
		printf("  Warp size: %d\n", 
			dp.warpSize);

		//amount of global memory
		printf("  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(dp.totalGlobalMem / 1048576.0f),
			(unsigned long long)dp.totalGlobalMem);

		//amount of constant memory
		printf("  Total amount of constant memory: %zu bytes\n",
			dp.totalConstMem);

		//amount of shared memory per block
		printf("  Total amount of shared memory per block: %zu bytes\n",
			dp.sharedMemPerBlock);

		//amount of registers available per block
		printf("  Total number of registers available per block: %d\n",
			dp.regsPerBlock);

		//maximum number of threads per block
		printf("  Total number of threads available per block : %d\n",
			dp.maxThreadsPerBlock);
		
		//maximum dimension of each block
		printf("  Max dimension of a block (x,y,z): (%d, %d, %d)\n",
			dp.maxThreadsDim[0], dp.maxThreadsDim[1],
			dp.maxThreadsDim[2]);

		//max size of dimension of a grid
		printf("  Max dimension of a grid size (x,y,z): (%d, %d, %d)\n",
			dp.maxGridSize[0], dp.maxGridSize[1],
			dp.maxGridSize[2]);

	}

}

