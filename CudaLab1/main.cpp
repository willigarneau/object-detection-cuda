/*
	Travail Pratique #1
	Par Willima Garneau
*/
#include "stdafx.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

extern "C" void ParallelBlackAndWhite(uchar *inputMatrixPointer, int pixelIncrementation, uchar *outputMatrixPointer, dim3 matrixDimension);

int main()
{
	// original frame
	Mat originalframe = imread("lena.png");

	imshow("original frame", originalframe);;

	waitKey(0);
	return 0;
}
