// ComputerVisionP1.cpp : This file contains the 'main' function. Program execution begins and ends there.
// const char* default_file = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/ratsche.png";

/*
const char* default_file = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/ratsche.png";
const char* filename = argc >= 2 ? argv[1] : default_file;
// Loads an image
Mat src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);

cv::resize(src, src, cv::Size(), 0.2, 0.2);
*/
#include <iostream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

cv::Mat src;
cv::Mat dst;

	//const char* filename = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/ratschedturned.png";
//const char* filename = "C:/Users/denni/Documents/Uni/Master_SS_2022/CompVision/Praktika/P1/r.jpg";

//const char* filename = "C:/Users/denni/Documents/Uni/Master_SS_2022/CompVision/Praktika/newPics/red.jpg";

//const char* filename = "r.jpg";
//const char* filename = "D:/Documents/Uni/Master/SS_22/ComputerVision/red.jpg";



void get_src(const char* filename, cv::Mat& src)
{
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
}

void get_src_scaled(const char* filename, cv::Mat& src)
{
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
	cv::resize(src, src, cv::Size(), 0.2, 0.2);
}


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

	double px_per_mm = (width_rel + height_rel) / 2;	//px pro mm

	return px_per_mm;
}



void angled_measurement(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
{
	std::vector<cv::RotatedRect> minRect(contours.size());
	std::vector<cv::RotatedRect> minRectOnly;

	
	//TODO: instead fix threshold, take only the two largest rects
	for (size_t i = 0; i < contours.size(); i++)
	{
		if (cv::minAreaRect(contours[i]).size.height > 105 && cv::minAreaRect(contours[i]).size.width > 105)
		{
			minRect[i] = cv::minAreaRect(contours[i]);

			if (cv::minAreaRect(contours[i]).size.height > 105 && cv::minAreaRect(contours[i]).size.width > 105)
				minRectOnly.push_back(cv::minAreaRect(contours[i]));
		}
	
		
	}
	cv::RNG rot_rng(12345);

	//cv::Mat drawingR = cv::Mat::zeros(src.size(), CV_8UC3);
	cv::Mat drawingR = original.clone();

	std::cout << "angled contours size: " << contours.size() << std::endl;
	std::cout << "angled contours size only: " << minRectOnly.size() << std::endl;

	//calculating size:
	if (minRectOnly.size() == 2)
	{

		double px_per_mm = calc_px_per_mm(minRectOnly.at(0).size.width, minRectOnly.at(0).size.height, minRectOnly.at(1).size.width, minRectOnly.at(1).size.height, reference_side_length);
	
		double real_width = minRectOnly.at(0).size.width / px_per_mm;
		double real_height = minRectOnly.at(0).size.height / px_per_mm;
		
		double real_width1 = minRectOnly.at(1).size.width / px_per_mm;
		double real_height1 = minRectOnly.at(1).size.height / px_per_mm;

		std::cout << "real width: " << real_width << std::endl;
		std::cout << "real height: " << real_height << std::endl;

		std::cout << "real width 1: " << real_width1 << std::endl;
		std::cout << "real width 1: " << real_height1 << std::endl;
	}
		


	//drawing
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rot_rng.uniform(0, 256), rot_rng.uniform(0, 256), rot_rng.uniform(0, 256));
		// contour
		drawContours(drawingR, contours, (int)i, color);

		// rotated rectangle
		cv::Point2f rect_points[4];
		minRect[i].points(rect_points);	//bottomleft, topleft, topright, bottomright
		for (int j = 0; j < 4; j++)
		{
			line(drawingR, rect_points[j], rect_points[(j + 1) % 4], color);
		}
	}
	imshow("angled: ", drawingR);
}




void measurement(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
{

	std::vector<std::vector<cv::Point> > contours_poly(contours.size());

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Rect> boundRectOnly;

	//adding rects to vector. TODO: minareaRect
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);

		if (boundingRect(contours_poly[i]).height > 105 && boundingRect(contours_poly[i]).width > 105)
		{
			boundRectOnly.push_back(boundingRect(contours_poly[i]));
			std::cout << "push back" << std::endl;
			boundRect[i] = boundingRect(contours_poly[i]);
		}
	}


	if (boundRectOnly.size() == 2)
	{
		double px_per_mm = calc_px_per_mm(boundRectOnly.at(0).width, boundRectOnly.at(0).height, boundRectOnly.at(1).width, boundRectOnly.at(1).height, reference_side_length);

		double real_width = boundRectOnly.at(0).width / px_per_mm;
		double real_height = boundRectOnly.at(0).height / px_per_mm;

		double real_width1 = boundRectOnly.at(1).width / px_per_mm;
		double real_height1 = boundRectOnly.at(1).height / px_per_mm;

		std::cout << "rec real width: " << real_width << std::endl;
		std::cout << "rec real height: " << real_height << std::endl;

		std::cout << "rec real width 1: " << real_width1 << std::endl;
		std::cout << "rec real width 1: " << real_height1 << std::endl;
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

		/*
		if (boundingRect(contours_poly[i]).height > 30 && boundingRect(contours_poly[i]).width > 30)
		{

			std::string s = std::to_string(boundRect[i].height);
			std::string h = std::to_string(boundRect[i].width);

			std::cout << s << std::endl;
			std::cout << h << std::endl;

			//cv::putText(drawing, s, cv::Point(10, 50 + (i*20)), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
			//cv::putText(drawing, h, cv::Point(10, 100 + (i*20)), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
		}
		*/
	}
	
	//cv::resize(drawing, drawing, cv::Size(), 0.4, 0.4);
	imshow("Contours method:", drawing);
}



cv::Mat imbus()
{
	//global scaling
	const double scaling = 0.20;

	//const values 
	const double image_width = 3024;
	const double image_height = 4032;

	//reference length of dice in millimeter
		//dice	
		//const double reference_side_length = 15;	//in mm

	//unterlegscheibe
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
		//desktop
	//const char* imbus = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/imbus/imb_dice.jpg";
	//const char* imbus = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/imbus/turned_dice.jpg";

	const char* imbus = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/unterlegscheibe/turned.jpg";

		//laptop
	//const char* imbus = "C:/Users/denni/Documents/Uni/Master_SS_2022/CompVision/Praktika/imbus/imbus_dice.jpg";

	//reading image
	src = cv::imread(cv::samples::findFile(imbus), cv::IMREAD_COLOR);

	//resizing image to fit the screen. use global scaling:
	cv::resize(src, src, cv::Size(), scaling, scaling);


	//convert bgr to hsv colour space
	cv::Mat hsv;
	cvtColor(src, hsv, cv::COLOR_BGR2HSV_FULL);

	//filter for specific hsv range (exluding the red background S: red < 60)
	cv::Mat dst;
	//cv::inRange(hsv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(360, 60, 255), dst);

	//new pic:
	cv::inRange(hsv, cv::Scalar(low_H, low_S, low_V), cv::Scalar(360, 85, 255), dst);
		//imshow("hsv filter: ", dst);


	//closing to fill gaps in dice (fault of dice, in an optimal environment this is not needed)
	cv::Mat element = cv::getStructuringElement(0, cv::Size(2* 2.5 +1, 2*2.5+1 ), cv::Point(2.5, 2.5 ));
	cv::morphologyEx(dst, dst, 1, element);
		//imshow("closing:", dst);

	//erosion to adjust the unneccessary effects of the closing morphology
	cv::Mat el = cv::getStructuringElement(0, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1));
	cv::erode(dst, dst, el);
	cv::erode(dst, dst, el);
		//imshow("erosion:", dst);
	

	//finding contours
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(dst, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	//angled measurments
	angled_measurement(dst, contours, reference_side_length, src);

	measurement(dst, contours, reference_side_length, src);

	return dst;
}


/// <summary>
/// Instead of trying to find the contours, make a picture with an empty background. Then, take the object and the difference should be 
/// object we are trying to measure. 
/// Possible Problem: Shadows.
/// justification: in a stationary situation with the same lighting at every time, this is easily achivable
/// </summary>
/// <returns></returns>
cv::Mat BackgroundRemoval()
{
	cv::Mat img;
	return img;
}


int main()
{

	cv::Mat src;
	
	
	imbus();

	//imbus_background_sub(src);

	cv::waitKey();
	return 0;
}