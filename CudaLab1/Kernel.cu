#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))

int const BLOCK_SIZE = 32; // for a 512*512 image

typedef unsigned char uchar;

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void parallelRGBToHSV_kernel(uchar *inputMatrix, uchar *outputMatrix) {
	int noCol = blockIdx.x * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y * blockDim.y + threadIdx.y;
	int dimension = blockDim.x * gridDim.x;
	int index = noRow * dimension * noCol * 3;
	outputMatrix[index] = inputMatrix[index];
}

extern "C" void parallelRGBToHSV(uchar *inputMatrixPointer, uchar *outputMatrixPointer, dim3 matrixDimension) {
	uchar *inputMatrixGrid, *outputMatrixGrid;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	// allocate memory space for both matrix on gpu
	size_t memorySize = matrixDimension.x * matrixDimension.y * sizeof(uchar);
	cudaMalloc(&inputMatrixGrid, memorySize);
	cudaMalloc(&outputMatrixGrid, memorySize);

	// copy matrix(A) to gpu
	cudaMemcpy(inputMatrixGrid, inputMatrixPointer, memorySize, cudaMemcpyHostToDevice);

	// launch kernel >>> create kernel function to convert rgb to hsv
	parallelRGBToHSV_kernel<<<dimGrid, dimBlock>>>(inputMatrixGrid, outputMatrixGrid);

	// transfert result matrix
	cudaMemcpy(outputMatrixPointer, outputMatrixGrid, memorySize, cudaMemcpyDeviceToHost);

	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}
