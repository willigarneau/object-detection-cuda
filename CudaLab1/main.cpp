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

extern "C" void parallelRGBToHSV(uchar *inputMatrixPointer, uchar *outputMatrixPointer, dim3 matrixDimension);

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))


uchar HueConversion(float blue, float green, float red, float delta, float maximum) {
	uchar h;
	if (red == maximum) { h = 60* (green - blue) / delta; }
	if (green == maximum) { h = 60 * (blue - red) / delta + 120; }
	if (blue == maximum) { h = 60 * (red - green) / delta + 240; }

	if (h < 0) { h += 360; }
	return h;
}

void rgbToHSV(Mat frame) {
	Vec3b hsv;
	for (int rows = 0; rows < frame.rows; rows++) {
		for (int cols = 0; cols < frame.cols; cols++) {
			float blue =  frame.at<Vec3b>(rows, cols)[0] / 255.0; // blue
			float green = frame.at<Vec3b>(rows, cols)[1] / 255.0; // green
			float red = frame.at<Vec3b>(rows, cols)[2] / 255.0; // red

			float maximum = MAX3(red, green, blue);
			float minimum = MIN3(red, green, blue);

			float delta = maximum - minimum;
			uchar h = HueConversion(blue, green, red, delta, maximum);
			hsv[0] = h / 2;
			uchar s = (delta / maximum) * 255;
			hsv[1] = s;
			float v = (maximum) * 255;
			hsv[2] = v;

			frame.at<Vec3b>(rows, cols) = hsv;
		}
	}
}

int main()
{
	// original frame
	Mat originalframe = imread("lena.png");
	Mat cpuConvertedHSVFrame = imread("lena.png");

	rgbToHSV(cpuConvertedHSVFrame);
	Mat inputParallelConvertedFrame = imread("lena.png");
	Mat outputParallelConvertedFrame = imread("lena.png");
	parallelRGBToHSV(inputParallelConvertedFrame.data, outputParallelConvertedFrame.data,
		dim3(inputParallelConvertedFrame.rows, inputParallelConvertedFrame.cols));

	imshow("original frame", originalframe);
	imshow("CPU converted HSV frame", cpuConvertedHSVFrame);
	imshow("GPU converted HSV frame", outputParallelConvertedFrame);

	waitKey(0);
	return 0;
}
