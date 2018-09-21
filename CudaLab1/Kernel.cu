#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc.hpp>

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))

__device__ int Gx[3][3]{ 
	{ -1,0,1 },
	{ -2,0,2 },
	{ -1,0,1 }
};
__device__ int Gy[3][3]{
	{ 1,2,1 },
	{ 0,0,0 },
	{ -1,-2,-1 } 
};

using namespace cv;

int const BLOCK_SIZE = 32; // for a 512*512 image

typedef unsigned char uchar;
typedef unsigned int uint;


int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void parallelObjectDetection_kernel(uchar *inputMatrixImage, uchar *outputMatrixImage, int width, int height, bool replaceBackground,
	int treshMin, int treshMax, int foreground, int background) {
	int noCol = blockIdx.x * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y * blockDim.y + threadIdx.y;
	int index = noRow * width + noCol * 3;

	// RGB to HSV conversion
	if ((noRow < height) && (noCol<width/3)) {

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

		
		//// background substraction
		if (treshMin < outputMatrixImage[index] && outputMatrixImage[index] < treshMax) { // hue is between the given range
			if (replaceBackground)
			{
				outputMatrixImage[index] = foreground;
				outputMatrixImage[index + 1] = foreground;
				outputMatrixImage[index + 2] = foreground;
			}
			else {
				outputMatrixImage[index] = outputMatrixImage[index];
				outputMatrixImage[index + 1] = outputMatrixImage[index + 1];
				outputMatrixImage[index + 2] = outputMatrixImage[index + 2];
			}
		}
		else {
			outputMatrixImage[index] = background;
			outputMatrixImage[index + 1] = background;
			outputMatrixImage[index + 2] = background;
		}
	}
}

__global__ void parallelSobelFilter_kernel(uchar* inputImage, uchar* outputImage, int width, int height)
{
	int gIndex = blockIdx.x  * blockDim.x + threadIdx.x; // global index
	int tIndex = gIndex - width; // top index
	int bIndex = gIndex + width; // bottom index

	if (tIndex < 0 || bIndex>(width*height)) {
		return;
	}

	int gradientX =
		inputImage[tIndex - 1] * Gx[0][0] + // left top
		inputImage[gIndex - 1] * Gx[1][0] + // left middle
		inputImage[bIndex - 1] * Gx[2][0] + // left bottom
		inputImage[tIndex + 1] * Gx[0][2] + // right top
		inputImage[gIndex + 1] * Gx[1][2] + // right middle
		inputImage[bIndex + 1] * Gx[2][2];  // right bottom

	int gradientY = 
		inputImage[tIndex - 1] * Gy[0][0] + // left top
		inputImage[tIndex] * Gy[0][1] + // middle top
		inputImage[bIndex - 1] * Gy[2][0] + // left bottom
		inputImage[tIndex + 1] * Gy[0][2] + // right top
		inputImage[bIndex] * Gy[2][1] + // middle bottom
		inputImage[bIndex + 1] * Gy[2][2];  // right bottom

	gradientX = gradientX * gradientX;
	gradientY = gradientY * gradientY;
	float approxGradient = sqrtf(gradientX + gradientY + 0.0);
	if (approxGradient > 255) { 
		approxGradient = 255;
	}
	outputImage[gIndex] = (uchar)approxGradient;
}

extern "C" cudaError_t parallelObjectDetection(Mat *inputImage, Mat *outputImage, bool replaceBackground, int treshMin, int treshMax, int foreground, int background) {
	cudaError_t status;
	uchar *inputThreshold, *outputThreshold;
	// 1. Define input/output image sizes
	uint imgSize = inputImage->step1() * inputImage->rows;
	// 0. Define block dimension
	dim3 BLOCK_COUNT(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GRID_COUNT(iDivUp(inputImage->cols, BLOCK_SIZE), iDivUp(inputImage->rows, BLOCK_SIZE));
	// 2. Allocate memory space for both matrix on gpu
	status = cudaMalloc(&inputThreshold, imgSize);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}
	status = cudaMalloc(&outputThreshold, imgSize);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}
	// 3. Send matrix(A) to gpu
	status = cudaMemcpy(inputThreshold, inputImage->data, imgSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}
	// 4. Treat matrix in object detection kernel
	parallelObjectDetection_kernel<<<GRID_COUNT, BLOCK_COUNT>>>(inputThreshold, outputThreshold, inputImage->step1(), inputImage->rows, replaceBackground,
		treshMin, treshMax, foreground, background);
	// 5. Wait for the kernel to end
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed");
		goto Error;
	}
	// 6. Transfer result matrix to output image
	status = cudaMemcpy(outputImage->data, outputThreshold, imgSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}
	// 7. Free matrix in memory
	cudaFree(inputThreshold);
	cudaFree(outputThreshold);
Error:
	cudaFree(inputThreshold);
	cudaFree(outputThreshold);

	return status;
}
extern "C" cudaError_t parallelSobelFilter(Mat *inputImage, Mat *outputImage) {
	cudaError_t status;
	uchar *inputSobel, *outputSobel;
	// 0. Define block dimension
	int BLOCK_COUNT = iDivUp((inputImage->cols * inputImage->rows), BLOCK_SIZE);
	// 1. Define input/output image sizes
	uint imgSize = inputImage->rows * inputImage->step1();
	uint gradientSize = inputImage->rows * inputImage->cols * sizeof(uchar);
	// 2. Allocate memory space for both matrix on gpu
	status = cudaMalloc(&inputSobel, imgSize);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}
	cudaMalloc(&outputSobel, gradientSize);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed");
		goto Error;
	}
	// 3. Send matrix(A) to gpu
	status = cudaMemcpy(inputSobel, inputImage->data, imgSize, cudaMemcpyHostToDevice);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}
	// 4. Treat matrix in sobel filter kernel
	parallelSobelFilter_kernel<<<BLOCK_COUNT, BLOCK_SIZE>>>(inputSobel, outputSobel, inputImage->cols, inputImage->rows);
	// 5. Wait for the kernel to end
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize failed");
		goto Error;
	}
	// 6. Transfer result matrix to output image
	status = cudaMemcpy(outputImage->data, outputSobel, gradientSize, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed");
		goto Error;
	}
	// 7. Free matrix in memory
	cudaFree(inputSobel);
	cudaFree(outputSobel);
Error:
	cudaFree(inputSobel);
	cudaFree(outputSobel);

	return status;
}

