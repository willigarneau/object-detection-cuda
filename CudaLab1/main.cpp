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

extern "C" void parallelRGBToHSV(Mat *inputImage, Mat *outputImage);
extern "C" void parallelBackgroundSubstraction(
	Mat *inputImage,
	Mat *outputImage,
	uchar *backgroundColor,
	bool replaceForeground,
	uchar *foregroundColor,
	uchar *treshMin,
	uchar *treshMax);

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))


float HueConversion(float blue, float green, float red, float delta, float maximum) {
	float h;
	if (red == maximum) { h = 60* (green - blue) / delta; }
	if (green == maximum) { h = 60 * (blue - red) / delta + 2; }
	if (blue == maximum) { h = 60 * (red - green) / delta + 4; }

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

			float hue = HueConversion(blue, green, red, delta, maximum);
			hsv[0] = (uchar)(hue / 2);
			float saturation = (delta / maximum) * 255.0;
			hsv[1] = (uchar)saturation;
			float value = (maximum) * 255.0;
			hsv[2] = (uchar)value;

			frame.at<Vec3b>(rows, cols) = hsv;
		}
	}
}

int main()
{
	// original frame
	Mat originalframe = imread("lena.png");
	Mat cpuConvertedHSVFrame = imread("lena.png");

	// rgb to hsv conversion
	rgbToHSV(cpuConvertedHSVFrame);
	Mat inputHSVImage = imread("scene.png");
	Mat HSVImage = imread("scene.png");
	parallelRGBToHSV(&inputHSVImage, &HSVImage);

	// background substraction
	Mat outputBackgroundSubstraction = HSVImage;
	uchar backgroundColor[3] = { 0, 0, 0 };
	uchar foregroundColor[3] = { 255, 0, 0 };
	uchar treshMin[3] = { 100, 100, 100 };
	uchar treshMax[3] = { 200, 200, 200 };
	parallelBackgroundSubstraction(&HSVImage, &outputBackgroundSubstraction, backgroundColor, true, foregroundColor, treshMin, treshMax);

	imshow("original frame", originalframe);
	imshow("CPU converted HSV frame", cpuConvertedHSVFrame);
	imshow("GPU converted HSV frame", HSVImage);

	waitKey(0);
	return 0;
}
