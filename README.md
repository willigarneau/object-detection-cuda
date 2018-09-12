# Motion detection with Cuda and OpenCV
> Implementation of a motion detection solution with OpenCV and Cuda, as part of the Intelligent Industrial System Course.

### Part 1: Convert an RGB color image to HSV, without using an external library.

#### Here's the algorithm :
The R,G,B values are divided by 255 to change the range from 0..255 to 0..1:<br>
R' = R/255<br>
G' = G/255<br>
B' = B/255<br>
Cmax = max(R', G', B')<br>
Cmin = min(R', G', B')<br>
Δ = Cmax - Cmin<br>
Hue calculation:<br>
http://www.rapidtables.com/convert/color/rgb-to-hsv/hue-calc.gif<br>
Saturation calculation:<br>
http://www.rapidtables.com/convert/color/rgb-to-hsv/sat-calc.gif<br>
Value calculation: V = Cmax<br>

The algorithm running on both the CPU and the GPU has been implemented in both cases, to facilitate understanding of the steps to follow when paralleling in Cuda.

```c++
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
```

Also, here is an example of a way to calculate the hue value in c++ :

```c++

uchar HueConversion(float blue, float green, float red, float delta, float maximum) {
	uchar h;
	if (red == maximum) { h = 60* (green - blue) / delta; }
	if (green == maximum) { h = 60 * (blue - red) / delta + 120; }
	if (blue == maximum) { h = 60 * (red - green) / delta + 240; }
	if (h < 0) { h += 360; }
	return h;
}
```
> The Hue value you get needs to be multiplied by 60 to convert it to degrees on the color circle. If Hue becomes negative you need to add 360 to, because a circle has 360 degrees.

###### `MAX`/`MIN` and `MAX3`/`MIN` functions were made to calculate highest and lowest value between 2 or 3 parameters like this :
```c++
#define MIN(a,b)      ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define MIN3(a,b,c)   MIN((a), MIN((b), (c)))
#define MAX3(a,b,c) MAX((a), MAX((b), (c)))
```
