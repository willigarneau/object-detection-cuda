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

#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))

const int HUE_DEGREE =  512;

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
			hsv[0] = h /2;
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
	Mat originalframe = imread("scene.jpg");

	rgbToHSV(originalframe);


	imshow("original frame", originalframe);;

	waitKey(0);
	return 0;
}
