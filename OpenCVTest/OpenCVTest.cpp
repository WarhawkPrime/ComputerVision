// ComputerVisionP1.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>


void get_src(const char* filename, cv::Mat& src)
{
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
}

void get_src_scaled(const char* filename, cv::Mat& src)
{
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
	cv::resize(src, src, cv::Size(), 0.2, 0.2);
}

/// <summary>
/// calculates the pixel per millimetre. The function assumes that the bigger rect is the object
/// and the smaller one is the reference object. "reference_side_length" measures the real-world
/// measurements for the reference object. With this, it is trivial to calculate the pixel to
/// millimetre ratio. This ratio is returned as a double value.
/// </summary>
/// <param name="width"></param>
/// <param name="height"></param>
/// <param name="width1"></param>
/// <param name="height1"></param>
/// <param name="reference_side_length"></param>
/// <returns></returns>
double calc_px_per_mm(double width, double height, double width1, double height1, double reference_side_length)
{
	double ref_width = 0, ref_height = 0, obj_width = 0, obj_height = 0;

	if (width < width1 && height < height1)
	{
		ref_width = width;
		ref_height = height;

		obj_width = width1;
		obj_height = height1;
	}
	else 
	{
		ref_width = width1;
		ref_height = height1;

		obj_width = width;
		obj_height = height;
	}

	double width_rel = ref_width / reference_side_length;
	double height_rel = ref_height / reference_side_length;

	double px_per_mm = (width_rel + height_rel) / 2;	//px per mm

	return px_per_mm;
}


/// <summary>
/// measures and calculates the sides of a rotated Rect around the object. Returns the ratio of pixel per millimetre
/// </summary>
/// <param name="src"></param>
/// <param name="contours"></param>
/// <param name="reference_side_length"></param>
/// <param name="original"></param>
/// <returns></returns>
double angled_measurement(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
{
	//vectors for the minRects. minRectOnly holds the two relevant rects of the object and reference object
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::RotatedRect> minRectOnly;


	//TODO: instead fix threshold, take only the two largest rects
	for (size_t i = 0; i < contours.size(); i++)
	{
		//minAreaRects finds the area with the minimum area.
		if (cv::minAreaRect(contours[i]).size.height > 105 && cv::minAreaRect(contours[i]).size.width > 105)
		{
			minRect[i] = cv::minAreaRect(contours[i]);

			if (cv::minAreaRect(contours[i]).size.height > 105 && cv::minAreaRect(contours[i]).size.width > 105)
				minRectOnly.push_back(cv::minAreaRect(contours[i]));
		}
	}
	
	double px_per_mm = 0;

	//calculating size with the only 2 relevant minRectOnlys
	if (minRectOnly.size() == 2)
	{
		px_per_mm = calc_px_per_mm(minRectOnly.at(0).size.width, 
									minRectOnly.at(0).size.height, 
									minRectOnly.at(1).size.width, 
									minRectOnly.at(1).size.height, 
									reference_side_length);
	}

	//===== drawing of the contours =====

	//random for colours
	cv::RNG rot_rng(12345);
	cv::Mat drawingR = original.clone();

	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rot_rng.uniform(0, 256), rot_rng.uniform(0, 256), rot_rng.uniform(0, 256));

		drawContours(drawingR, contours, (int)i, color);

		// rotated rectangle
		cv::Point2f rect_points[4];
		minRect[i].points(rect_points);	//bottomleft, topleft, topright, bottomright
		for (int j = 0; j < 4; j++)
		{
			line(drawingR, rect_points[j], rect_points[(j + 1) % 4], color);
		}
	}

	imshow("angled_measurement", drawingR);

	return px_per_mm;
}



/// <summary>
/// measures and calculates the sides of a Rect around the object. Returns the ratio of pixel per millimetre
/// </summary>
/// <param name="src"></param>
/// <param name="contours"></param>
/// <param name="reference_side_length"></param>
/// <param name="original"></param>
/// <returns></returns>
double measurement(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
{
	//vector to hold the approxPolyDP
	std::vector<std::vector<cv::Point> > contours_poly(contours.size());

	//vectors for the minRects. minRectOnly holds the two relevant rects of the object and reference object
	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Rect> boundRectOnly;

	//adding rects to vector. TODO: minareaRect
	for (size_t i = 0; i < contours.size(); i++)
	{
		//approximates curves
		approxPolyDP(contours[i], contours_poly[i], 3, true);

		if (boundingRect(contours_poly[i]).height > 105 && boundingRect(contours_poly[i]).width > 105)
		{
			boundRectOnly.push_back(boundingRect(contours_poly[i]));
			boundRect[i] = boundingRect(contours_poly[i]);
		}
	}

	double px_per_mm = 0;

	//calculating size with the only 2 relevant minRectOnlys
	if (boundRectOnly.size() == 2)
	{
		px_per_mm = calc_px_per_mm(boundRectOnly.at(0).width, 
									boundRectOnly.at(0).height, 
									boundRectOnly.at(1).width, 
									boundRectOnly.at(1).height, 
									reference_side_length);
	}

	//drawing the contours and rects
	//cv::Mat drawing = cv::Mat::zeros(src.size(), CV_8UC3);
	cv::Mat drawing = original.clone();
	cv::RNG rng(12345);

	for (size_t i = 0; i < boundRect.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours_poly, (int)i, color);
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
	}
	
	imshow("measurement", drawing);

	return px_per_mm;
}


/// <summary>
/// calculates the dimensions of the object. The function determines the contour with the biggest area, which is our object
/// From this contour, we can determine the 4 extreme points in the x and y axis. This way, we are not restricted by a Rect.
/// We then calculate the length of the vectors between the 4 points. 
/// We know from the outlines of an imbus, that we need the second and third largest lines for the correct measurements
/// </summary>
/// <param name="contours"></param>
/// <param name="src"></param>
/// <param name="px_to_mm"></param>
void calc_dimensions(std::vector<std::vector<cv::Point>> contours, cv::Mat src, double px_to_mm)
{
	//find moments of contours and biggest contour (our object)
	std::vector<cv::Point> object_contour;
	double area = 0;
	size_t id = 0;

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (cv::contourArea(contours[i]) > area)
		{
			area = cv::contourArea(contours[i]);
			id = i;
		}
	}


	//biggest contour => our object
	object_contour = contours[id];

	//use std::minmax_element to find the extreme points
	auto val_x = std::minmax_element(object_contour.begin(), object_contour.end(), [](cv::Point const& a, cv::Point const& b)
		{
			return a.x < b.x;
		});

	auto val_y = std::minmax_element(object_contour.begin(), object_contour.end(), [](cv::Point const& a, cv::Point const& b)
		{
			return a.y < b.y;
		});


	//create the 4 points
	cv::Point leftMost(val_x.first->x, val_x.first->y);
	cv::Point rightMost(val_x.second->x, val_x.second->y);
	cv::Point topMost(val_y.first->x, val_y.first->y);
	cv::Point bottomMost(val_y.second->x, val_y.second->y);

	//calc Euclidian distances
	double tm_rm_d = cv::norm(topMost - rightMost);
	double rm_bm_d = cv::norm(rightMost - bottomMost);
	double bm_lm_d = cv::norm(bottomMost - leftMost);
	double lm_tm_d = cv::norm(leftMost - topMost);

	//sort lines
	std::vector<double> lengths;
	lengths.push_back(tm_rm_d);
	lengths.push_back(rm_bm_d);
	lengths.push_back(bm_lm_d);
	lengths.push_back(lm_tm_d);

	std::sort(lengths.begin(), lengths.end());

	//the measurements are
	double width = lengths.at(1);
	double length = lengths.at(2);

	std::cout << "WIDTH: " << width/ px_to_mm << std::endl;
	std::cout << "LENGTH: " << length / px_to_mm << std::endl;

	//===== draw dots =====
	cv::RNG rng(12345);
	cv::Scalar color = cv::Scalar(255, 0, 0);

	cv::Mat dots = src.clone();

	cv::circle(dots, leftMost, 10, color, -1);
	cv::circle(dots, rightMost, 10, color, -1);
	cv::circle(dots, topMost, 10, color, -1);
	cv::circle(dots, bottomMost, 10, color, -1);

	imshow("dots: ", dots);
}



/// <summary>
/// reads image, carries out image processing, finds the contours of the objects and calls the functions
/// angled_measurement, measurement and calc_dimensions
/// </summary>
/// <returns></returns>
cv::Mat imbus()
{
	cv::Mat src;
	cv::Mat dst;

	//global scaling
	//desktop
	//const double scaling = 0.20; //105 as threshold for rects

	//laptop
	const double scaling = 0.15;	//80 as Theshold for rects

	//const values 
	const double image_width = 3024;
	const double image_height = 4032;

	//unterlegscheibe reference
	const double reference_side_length = 40;	//in mm

	//maximum value for hsv
	const int max_value = 255;
	
	//variables to filter out a value in hsv
	int low_H = 0;
	int low_S = 0;
	int low_V = 0;
	int high_H = max_value;
	int high_S = max_value;
	int high_V = max_value;

	//file selection
	const char* imbus = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/unterlegscheibe/turned.jpg";

	//reading image
	src = cv::imread(cv::samples::findFile(imbus), cv::IMREAD_COLOR);

	//resizing image to fit the screen. use global scaling:
	cv::resize(src, src, cv::Size(), scaling, scaling);

	//converts bgr to hsv colour space
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV_FULL);

	//filters out the red color
	cv::inRange(hsv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(360, 100, 255), dst);
	imshow("hsv filter: ", dst);

	//finding contours
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(dst, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	imshow("cont: ", dst);
	
	//calculate the rect measuremts and get the ratio
	double px_to_mm_a = angled_measurement(dst, contours, reference_side_length, src);
	double px_to_mm_r = measurement(dst, contours, reference_side_length, src);
	
	//calculate the dimension via the reference and the extrem points of the contours
	calc_dimensions(contours, src, (px_to_mm_a + px_to_mm_r)/2 );

	return dst;
}



int main()
{
	imbus();

	cv::waitKey();
	return 0;
}