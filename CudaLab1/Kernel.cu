#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int const BLOCK_SIZE = 32;

typedef unsigned char uchar;

__global__ void ParallelBlackAndWhite_kernel(uchar *inputMatrix, int pixelIncrementation, uchar *outputMatrix) {
	int noCol = blockIdx.x  * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y  * blockDim.y + threadIdx.y;
	int dim = blockDim.x * gridDim.x;
	int cudaIndex = noRow * dim + noCol;
	outputMatrix[cudaIndex] = inputMatrix[cudaIndex] + pixelIncrementation;
}

extern "C" void ParallelBlackAndWhite(uchar *inputMatrixPointer, int pixelIncrementation, uchar *outputMatrixPointer, dim3 matrixDimension)
{
	uchar *inputMatrixGrid, *outputMatrixGrid;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	// Allouer l'espace memoire des 2 matrices sur la carte GPU 
	size_t memSize = matrixDimension.x * matrixDimension.y * sizeof(uchar);
	cudaMalloc(&inputMatrixGrid, memSize);
	cudaMalloc(&outputMatrixGrid, memSize);

	// Copier de la matrice A dans la memoire du GPU
	cudaMemcpy(inputMatrixGrid, inputMatrixPointer, memSize, cudaMemcpyHostToDevice);

	// Partir le kernel
	ParallelBlackAndWhite_kernel<<<dimGrid, dimBlock>>>(inputMatrixGrid, pixelIncrementation, outputMatrixGrid);

	// Transfert de la matrice résultat 
	cudaMemcpy(outputMatrixPointer, outputMatrixGrid, memSize, cudaMemcpyDeviceToHost);

	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}

