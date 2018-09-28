/*
	Travail Pratique #1
	Par Willima Garneau
*/
#include "stdafx.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "AxisCommunication.h"

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc.hpp>

using namespace cv;

extern "C" void parallelObjectDetection(Mat *inputImage, Mat *outputImage, bool replaceBackground, int treshMin, int treshMax, int foreground, int background);
extern "C" void parallelSobelFilter(Mat *inputImage, Mat *outputImage);

int minHue = 75, maxHue = 200;
bool replaceBackground = false;

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

void AxisCamera() {
	// Set initial matrix for gpu processing
	Mat inputThresholdImage, outputThresholdImage, outputSobelImage;

	Axis axis("10.128.3.4", "etudiant", "gty970");
	while (true) {
		axis.GetImage(inputThresholdImage);
		axis.GetImage(outputThresholdImage);
		axis.GetImage(outputSobelImage);

		parallelObjectDetection(&inputThresholdImage, &outputThresholdImage, replaceBackground, 75, 200, 255, 0);
		imshow("Original frame", inputThresholdImage);

		if (replaceBackground) {
			// Convert result patrix to grayscale
			cvtColor(outputThresholdImage, outputThresholdImage, CV_RGB2GRAY);
			cvtColor(outputSobelImage, outputSobelImage, CV_RGB2GRAY);

			//// GPU process for applying sobel filter
			parallelSobelFilter(&outputThresholdImage, &outputSobelImage);
			imshow("GPU converted Sobel frame", outputThresholdImage);
		}
		else {
			imshow("GPU converted Threshold frame", outputThresholdImage);
			parallelObjectDetection(&inputThresholdImage, &outputThresholdImage, replaceBackground, 75, 170, 255, 0);
			// Convert result patrix to grayscale
			cvtColor(outputThresholdImage, outputThresholdImage, CV_RGB2GRAY);
			cvtColor(outputSobelImage, outputSobelImage, CV_RGB2GRAY);

			// GPU process for applying sobel filter
			parallelSobelFilter(&outputThresholdImage, &outputSobelImage);
		}

		imshow("Axis PTZ", outputSobelImage);
		waitKey(5);
	}
	axis.ReleaseCam();
}

void staticFrames() {}

int main()
{
	// Convert HSV image on cpu
	Mat cpuConvertedHSVFrame = imread("scene2.png");
	rgbToHSV(cpuConvertedHSVFrame);

	//AxisCamera();

	// Set initial matrix for gpu processing
	Mat inputThresholdImage = imread("scene.png");
	Mat ThresholdImage = imread("scene.png");
	Mat outputSobelImage = imread("scene.png");

	// GPU process to detect object
	parallelObjectDetection(&inputThresholdImage, &ThresholdImage, replaceBackground, minHue, maxHue, 255, 0);

	if (replaceBackground) {
		// Convert result patrix to grayscale
		cvtColor(ThresholdImage, ThresholdImage, CV_RGB2GRAY);
		cvtColor(outputSobelImage, outputSobelImage, CV_RGB2GRAY);

		//// GPU process for applying sobel filter
		parallelSobelFilter(&ThresholdImage, &outputSobelImage);
		createTrackbar("minHue", "GPU converted Threshold frame", &minHue, 200, 0);
		imshow("GPU converted Threshold frame", ThresholdImage);
	}
	else {
		imshow("GPU converted Threshold frame", ThresholdImage);
		parallelObjectDetection(&inputThresholdImage, &ThresholdImage, true, minHue, maxHue, 255, 0);
		// Convert result patrix to grayscale
		cvtColor(ThresholdImage, ThresholdImage, CV_RGB2GRAY);
		cvtColor(outputSobelImage, outputSobelImage, CV_RGB2GRAY);

		//// GPU process for applying sobel filter
		parallelSobelFilter(&ThresholdImage, &outputSobelImage);
	}


	// Show result images
	imshow("CPU converted HSV frame", cpuConvertedHSVFrame);
	imshow("GPU sobel filtered frame", outputSobelImage);

	waitKey(0);
	return 0;
}
