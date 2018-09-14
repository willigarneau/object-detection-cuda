#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))

using namespace cv;

int const BLOCK_SIZE = 32; // for a 512*512 image

typedef unsigned char uchar;
typedef unsigned int uint;


int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void parallelRGBToHSV_kernel(uchar *inputMatrixImage, uchar *outputMatrixImage, int width, int height) {
	int noCol = blockIdx.x * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y * blockDim.y + threadIdx.y;
	int index = noRow * width + noCol * 3;
	if ((noRow < height)) {

		float blue = inputMatrixImage[index] / 255.0;
		float green = inputMatrixImage[index + 1] / 255.0;
		float red = inputMatrixImage[index + 2] / 255.0;

		float maximum = MAX3(red, green, blue);
		float minimum = MIN3(red, green, blue);

		float delta = maximum - minimum;
		
		//hue calculation
		float hue;
		if (red == maximum) { hue = 60 * (green - blue) / delta; }
		if (green == maximum) { hue = 60 * (blue - red) / delta + 2; }
		if (blue == maximum) { hue = 60 * (red - green) / delta + 4; }
		if (hue < 0) { hue += 360; }
		outputMatrixImage[index] = (uchar)(hue / 2);

		//saturation calculation
		float saturation = (delta / maximum) * 255.0;
		outputMatrixImage[index + 1] = (uchar)saturation;

		//value calculation
		double value = maximum * 255.0;
		outputMatrixImage[index + 2] = (uchar)value;
	}
}

extern "C" void parallelRGBToHSV(Mat *inputImage, Mat *outputImage) {
	uchar *inputMatrixGrid, *outputMatrixGrid;
	// bytes for a row
	uint imgSize = inputImage->rows * inputImage->step1();
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(iDivUp(inputImage->cols, BLOCK_SIZE), iDivUp(inputImage->rows, BLOCK_SIZE));
	// allocate memory space for both matrix on gpu
	cudaMalloc(&inputMatrixGrid, imgSize);
	cudaMalloc(&outputMatrixGrid, imgSize);

	// copy matrix(A) to gpu
	cudaMemcpy(inputMatrixGrid, inputImage->data, imgSize, cudaMemcpyHostToDevice);

	// launch kernel >>> create kernel function to convert rgb to hsv
	parallelRGBToHSV_kernel << <dimGrid, dimBlock >> >(inputMatrixGrid, outputMatrixGrid, inputImage->step1(), inputImage->rows);

	// wait for the kernel to end
	cudaDeviceSynchronize();

	// transfert result matrix to output image
	cudaMemcpy(outputImage->data, outputMatrixGrid, imgSize, cudaMemcpyDeviceToHost);

	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}

extern "C" void parallelBackgroundSubstraction(Mat *inputImage, Mat *outputImage, uchar *backgroundColor,
	bool replaceForeground, uchar *foregroundColor, uchar *treshMin, uchar *treshMax) {

}

