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




//=============================
//loading the image
/*
src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
cv::resize(src, src, cv::Size(), 0.2, 0.2);

imshow("Source", src);
*/


/*
cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE); // Create window
cv::createTrackbar("Operator:\n 0: Opening - 1: Closing  \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations);
cv::createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
	&morph_elem, max_elem,
	Morphology_Operations);
cv::createTrackbar("Kernel size:\n 2n +1", window_name,
	&morph_size, max_kernel_size,
	Morphology_Operations);
Morphology_Operations(0, 0);
*/

/*
//threshold of 115
cv::Mat linedst = src.clone();
//cv::Mat linedst = abs_dst.clone();
cv::Mat det_edges;
cv::Mat dest;
int low_thres = 0;
int high_thres = 140;	//115
int multi = 3;
int kernel = 3;

dest.create(linedst.size(), linedst.type());

cvtColor(linedst, linedst, cv::COLOR_BGR2GRAY);
blur(linedst, det_edges, cv::Size(3, 3));
det_edges.convertTo(det_edges, -1, 1, 50);
Canny(det_edges, det_edges, high_thres, high_thres * multi, kernel);




//Einschub: Bounding Box
cv::RNG rng(12345);
std::vector<std::vector<cv::Point>> contours;
cv::Mat con_in = det_edges.clone();
cv::findContours(con_in, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

std::vector<std::vector<cv::Point> > contours_poly(contours.size());
std::vector<cv::Rect> boundRect(contours.size());
//std::vector<cv::Point2f>centers(contours.size());
//std::vector<float>radius(contours.size());

dest = cv::Scalar::all(0);
linedst.copyTo(dest, det_edges);

for (size_t i = 0; i < contours.size(); i++)
{
	approxPolyDP(contours[i], contours_poly[i], 3, true);

	if (boundingRect(contours_poly[i]).height > 100)
	{
		boundRect[i] = boundingRect(contours_poly[i]);
	}

	//boundRect[i] = boundingRect(contours_poly[i]);
	//minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
}
cv::Mat drawing = cv::Mat::zeros(con_in.size(), CV_8UC3);
for (size_t i = 0; i < contours.size(); i++)
{
	cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	drawContours(drawing, contours_poly, (int)i, color);
	rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);

	if (boundRect[i].height > 0 && boundRect[i].width > 0)
	{
		std::string s = std::to_string(boundRect[i].height);
		std::string h = std::to_string(boundRect[i].width);

		std::cout << s << std::endl;
		std::cout << h << std::endl;

		cv::putText(drawing, s, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
		cv::putText(drawing, h, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
	}

	//circle(drawing, centers[i], (int)radius[i], color, 2);
}
// ==========


imshow("Contours", drawing);
imshow("manuell canny: ", dest);
*/