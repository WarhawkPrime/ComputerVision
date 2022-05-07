#pragma once

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

cv::Mat loading_image(const char* filename)
{
	cv::Mat src;

	//loading the image
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);

	//resizing
	cv::Mat resized_src;
	cv::resize(src, resized_src, cv::Size(), 0.2, 0.2);

	//Color to Grey
	cv::Mat grey_src;
	cvtColor(src, grey_src, cv::COLOR_BGR2GRAY);

	//blur
	//blur or smooth an image. multiple possibilities
	//https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
	cv::Mat blur_dst;
	blur(src, blur_dst, cv::Size(3, 3));
}

/*
//now line detection
cv::Mat canny_out, line_out;

//copy edges
cv::cvtColor(dest, canny_out, cv::COLOR_GRAY2BGR);
line_out = canny_out.clone();

//probabilistic hough line transformation
std::vector<cv::Vec4i> lines;
cv::HoughLinesP(dest, lines, 1, CV_PI / 180, 50, 50, 10);

//draw lines
for (size_t i = 0; i < lines.size(); i++)
{
	cv::Vec4i l = lines[i];
	cv::line(line_out, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
}
*/


//colour to grey image
		//cvtColor(src, dst, cv::COLOR_BGR2GRAY);

//gamma correction
/*
cv::Mat lookUpTable(1, 256, CV_8U);
uchar* p = lookUpTable.ptr();

int gamma_cor = 100;
double gamma_value = gamma_cor / 150.0;

for (int i = 0; i < 256; ++i)
	p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_value) * 255.0);
cv::Mat res = dst.clone();
LUT(dst, lookUpTable, res);
imshow("gamma", res);
*/

//contrast
/*
dst.convertTo(dst, -1, 2, 0);
imshow("contrast", dst);
*/

//threshold
/*
cv::Mat thres;
cv::threshold(dst, thres, 90, 255, 2);
imshow("thres", thres);
*/

//adaptive threshold

/*
cv::Mat adThres;
cv::adaptiveThreshold(dst, adThres, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C , cv::THRESH_BINARY , 13, 3);
imshow("adThres", adThres);
*/

// historgramm equalization
/*
cv::Mat he = dst.clone();
cv::equalizeHist(dst, he);
imshow("heq", he);
*/

// canny edge detection
/*
cv::Mat cny = dst.clone();
cv::Canny(dst, cny, 50, 200, 23);
imshow("cnny", cny);
*/




//imshow("dst", dst);


// Wait and Exit