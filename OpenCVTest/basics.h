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



/// <summary>
/// 
/// </summary>
/// <param name="filename"></param>
cv::Mat blur_canny(cv::Mat src)
{
	//src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
	//cv::resize(src, src, cv::Size(), 0.2, 0.2);

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



	/* python
	# perform edge detection + dilation + erosion to close gap
s bt edges
edge_detect = cv2.Canny(gray, 15, 100) #play w/min and max
values to finetune edges (2nd n 3rd params)
edge_detect = cv2.dilate(edge_detect, None, iterations=1)
edge_detect = cv2.erode(edge_detect, None, iterations=1)
	*/



	//Einschub: Bounding Box

	//python rotate
	/*
	int largest_idx = 0;

	for (int idx = 0; idx < contours.size() ; idx++) {

		double a = Imgproc.contourArea(contours.get(idx));  //Find the area of contour

		if (a > largest_area) {

			largest_area = a;

			// rect = Imgproc.boundingRect(contours.get(idx));

			largest_idx = idx;
		}
	}

	MatOfPoint2f new_mat = new MatOfPoint2f( contours.get(largest_idx).toArray() );

	RotatedRect rbox = Imgproc.minAreaRect(new_mat);

	Log.d("rotatedrect_angle", "" + rbox.angle);

	Point points[] = new Point[4];

	rbox.points(points);

	for(int i=0; i<4; ++i){
		Imgproc.line(origMat, points[i], points[(i+1)%4], new Scalar(255,255,255));
	}
	*/




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

		if (boundingRect(contours_poly[i]).height > 500 && boundingRect(contours_poly[i]).width > 90)
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
	return drawing;
	//imshow("Canny", drawing);
}


void l_channel(const char* filename)
{
	//loading the image
	src = cv::imread(cv::samples::findFile(filename), cv::IMREAD_COLOR);
	cv::resize(src, src, cv::Size(), 0.2, 0.2);

	//================
	cv::Mat lab_image;
	cv::cvtColor(src, lab_image, cv::COLOR_BGR2Lab);

	//extract L channel
	std::vector<cv::Mat> lab_planes(3);
	cv::split(lab_image, lab_planes);

	//apply CLAHE to L channel
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	clahe->setClipLimit(4);
	cv::Mat clahe_dst;
	clahe->apply(lab_planes[0], clahe_dst);

	//merge planes back;
	clahe_dst.copyTo(lab_planes[0]);
	cv::merge(lab_planes, lab_image);

	//back to rgb
	cv::Mat image_clahe;
	cv::cvtColor(lab_image, image_clahe, cv::COLOR_Lab2BGR);

	imshow("lab", lab_image);
	imshow("lab after:", image_clahe);
	//src = image_clahe;
	//===============
}



void laplace(cv::Mat src)
{
	cv::Mat lap_src, lap_gray, lap_dst;
	int lap_kern = 3;
	int lap_scale = 1;
	int lap_delta = 0;
	int lap_ddepth = CV_16S;

	lap_src = src.clone();

	// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
	cv::GaussianBlur(lap_src, lap_src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
	cv::cvtColor(lap_src, lap_gray, cv::COLOR_BGR2GRAY); // Convert the image to grayscale
	cv::Mat abs_dst;
	cv::Laplacian(lap_gray, lap_dst, lap_ddepth, lap_kern, lap_scale, lap_delta, cv::BORDER_DEFAULT);
	// converting back to CV_8U
	cv::convertScaleAbs(lap_dst, abs_dst);
	imshow("laplace:", abs_dst);
}


void huegh_lines(cv::Mat src)
{
	//now line detection
	cv::Mat canny_out, line_out;

	//copy edges
	cv::cvtColor(src, canny_out, cv::COLOR_GRAY2BGR);
	line_out = canny_out.clone();

	//probabilistic hough line transformation
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(src, lines, 1, CV_PI / 180, 50, 50, 10);

	//draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(line_out, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}

	imshow("hough", line_out);
}




cv::Mat colour_filter(cv::Mat src)
{
	cv::Mat coloursrc = src.clone();

	//convert bgr to hsv colour space
	cv::Mat hsv;
	cvtColor(coloursrc, hsv, cv::COLOR_BGR2HSV_FULL);

	const int max_value_H = 360 / 2;
	const int max_value = 255;

	int low_H = 0;
	int low_S = 0;
	int low_V = 0;
	int high_H = max_value;
	int high_S = max_value;
	int high_V = max_value;

	//Werte aus Gimp gemessen. Könnte nachträglich auch im Programm stattfinden:
	/*
	H: 360 %	255
	S: 123 %	87,125
	V: 93 % 	65,87

	H: 353 %	250,04
	S: 77 %		54,54
	V: 60 %		42,5
	*/


	/*
	H: 357	252
	S: 90	229
	V: 90	229

	H: 105 1.5	74/1
	S: 2	2
	V: 18	45.9

	*/


	//detect range:
	cv::Mat dst;
	cv::inRange(coloursrc, cv::Scalar(70, low_S, low_V), cv::Scalar(255, 255, 255), dst);

	cv::Mat dest;
	cv::Mat linedst = src.clone();
	int low_thres = 0;
	int high_thres = 140;	//115
	int multi = 3;
	int kernel = 3;
	Canny(dst, dst, high_thres, high_thres * multi, kernel);




	//Einschub: Bounding Box
	cv::RNG rng(12345);
	std::vector<std::vector<cv::Point>> contours;
	cv::Mat con_in = dst.clone();
	cv::findContours(con_in, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point> > contours_poly(contours.size());
	std::vector<cv::Rect> boundRect(contours.size());
	//std::vector<cv::Point2f>centers(contours.size());
	//std::vector<float>radius(contours.size());

	//minarearect

	dest = cv::Scalar::all(0);
	linedst.copyTo(dest, dst);

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);

		if (boundingRect(contours_poly[i]).height > 0 && boundingRect(contours_poly[i]).width > 0)
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

			//calculating mm:
			//Distance to target: 20.3 mm
			//focal length 1,8f
			//Weitwinkelkamera 26mm f1.8
			//3024 x 4032

			/*
			There is only one focal length, F. It is the distance between the focal point and the image plane.

			The numbers you have, are actually not lengths.
			They are calculated by fx=F⋅sx and fy=F⋅sy where sx
			and sy are the size of your image in pixels.
			To get the focal length in mm as you want, simply
			divide fx with the width (in pixels) of your image.
			*/

			//fx = F * sx
			//fx = 1.8mm * 3024 = 5442.2
			//
		}

		//circle(drawing, centers[i], (int)radius[i], color, 2);
	}


	//return dst;
	return drawing;
}



cv::Mat imbus_background_sub(cv::Mat src)
{
	cv::Mat res;
	cv::Mat fg;
	cv::Mat bkg;

	cv::Mat fgMaskMOG2;

	const char* background = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/imbus/background.jpg";
	const char* imbus = "D:/Documents/Uni/Master/SS_22/ComputerVision/P1/ComputerVisionP1/imbus/imbus.jpg";

	fg = cv::imread(cv::samples::findFile(imbus), cv::IMREAD_COLOR);
	bkg = cv::imread(cv::samples::findFile(background), cv::IMREAD_COLOR);

	//===== Background subtraction: =====
	cv::Ptr< cv::BackgroundSubtractor > pBackSub;
	pBackSub = cv::createBackgroundSubtractorMOG2();

	pBackSub->apply(bkg, fgMaskMOG2);
	cv::resize(fgMaskMOG2, fgMaskMOG2, cv::Size(), 0.2, 0.2);
	imshow("mask: ", fgMaskMOG2);

	pBackSub->apply(fg, fgMaskMOG2);
	cv::resize(fgMaskMOG2, fgMaskMOG2, cv::Size(), 0.2, 0.2);
	imshow("mask:", fgMaskMOG2);

	cv::resize(bkg, bkg, cv::Size(), 0.2, 0.2);
	cv::resize(fg, fg, cv::Size(), 0.2, 0.2);

	imshow("foreground:", fg);
	imshow("bakcground:", bkg);

	//cv::resize(src, src, cv::Size(), 0.2, 0.2);
	return res;
}




void imbus_test()
{
	//filter out hsv, use colored background
	/*
	Mat hsv;
	vector<Mat> channels;
	split(hsv, channels);
	channels[0], channels[1], channels[2] will contain your H, S, V respectively.
	*/





	/*
	int morph_elem = 0;
	int morph_size = 0;
	int morph_operator = 0;
	int const max_operator = 4;
	int const max_elem = 2;
	int const max_kernel_size = 21;
	const char* window_name = "Morphology Transformations Demo";


	void Morphology_Operations(int, void*)
	{
		// Since MORPH_X : 2,3,4,5 and 6
		int operation = morph_operator + 2;
		cv::Mat element = cv::getStructuringElement(morph_elem, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
		morphologyEx(src, dst, operation, element);
		imshow(window_name, dst);
	}
	*/





	/*
	std::vector<std::vector<cv::Point> > contours_poly(contours.size());

	std::vector<cv::Rect> boundRect(contours.size());
	std::vector<cv::Rect> boundRectOnly;

	//adding rects to vector. TODO: minareaRect
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);

		if (boundingRect(contours_poly[i]).height > 30 && boundingRect(contours_poly[i]).width > 30)
		{
			boundRectOnly.push_back(boundingRect(contours_poly[i]));
			std::cout << "push back" << std::endl;
			boundRect[i] = boundingRect(contours_poly[i]);
		}
	}

	std::cout << "size of rectonly: " << boundRectOnly.size() << std::endl;

	//calculating size of object via reference:
	cv::Rect reference;
	cv::Rect object;

	//calculating which is the reference, which is the object
	double ref_x = 0, ref_y = 0, obj_x = 0, obj_y = 0;
	for (auto& ele : boundRectOnly)
	{
		if (ref_x == 0 && ref_y == 0)
		{
			ref_x = ele.width;
			ref_y = ele.height;
		}
		else
		{
			if (ele.height < ref_y && ele.width < ref_x)
			{
				obj_x = ref_x;
				obj_y = ref_y;

				ref_x = ele.width;
				ref_y = ele.height;
			}
			else
			{
				obj_x = ele.width;
				obj_y = ele.height;
			}
		}
	}
	*/

	//std::cout << "ref width: " << ref_x << " ref height: " << ref_y << std::endl;

	//get sizes in original pixel count
	//ref_x = ref_x * (1 / scaling);
	//ref_y = ref_y * (1 / scaling);

	//std::cout << "ref width: " << ref_x << " ref height: " << ref_y << std::endl;

	/*
	double width_rel = ref_x / reference_side_length;
	double height_rel = ref_y / reference_side_length;

	double px_per_mm = (width_rel + height_rel) / 2;	//px pro mm

	std::cout << "pixel per mm: " << px_per_mm << std::endl;

	//get sizes in original pixel count
	//obj_x = obj_x * (1 / scaling);
	//obj_y = obj_y * (1 / scaling);
	std::cout << "pbj width: " << obj_x << " obj height: " << obj_y << std::endl;

	obj_x = obj_x / px_per_mm;
	obj_y = obj_y / px_per_mm;

	std::cout << " width  in mm: " << obj_x << std::endl;
	std::cout << " height in mm: " << obj_y << std::endl;

	ref_x = ref_x / px_per_mm;
	ref_y = ref_y / px_per_mm;

	std::cout << " ref width  in mm: " << ref_x << std::endl;
	std::cout << " ref height in mm: " << ref_y << std::endl;

	//cv::resize(src, src, cv::Size(), scaling, scaling);
	imshow("original: ", src);

	//drawing the contours and rects
	cv::Mat drawing = cv::Mat::zeros(dst.size(), CV_8UC3);
	cv::RNG rng(12345);
	*/

	/*
	for (size_t i = 0; i < boundRect.size(); i++ )
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours_poly, (int)i, color);
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);


		if (boundingRect(contours_poly[i]).height > 30 && boundingRect(contours_poly[i]).width > 30)
		{

			std::string s = std::to_string(boundRect[i].height);
			std::string h = std::to_string(boundRect[i].width);

			std::cout << s << std::endl;
			std::cout << h << std::endl;

			//cv::putText(drawing, s, cv::Point(10, 50 + (i*20)), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
			//cv::putText(drawing, h, cv::Point(10, 100 + (i*20)), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
		}

	}

	std::string width_string = "Width: " + std::to_string(obj_x) + " mm";
	std::string height_string = "height: " + std::to_string(obj_y) + " mm";

	cv::putText(src, width_string, cv::Point(10, 50 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
	cv::putText(src, height_string, cv::Point(10, 100 ), cv::FONT_HERSHEY_SIMPLEX, 1.0, CV_RGB(118, 185, 0), 2, cv::LINE_AA, false);
	*/



	//cv::resize(drawing, drawing, cv::Size(), 0.4, 0.4);
	//imshow("Contours", drawing);


	//cvtColor(dst, dst, cv::COLOR_HSV2BGR_FULL);
	//dst = src.clone();
	//cvtColor(dst, dst, cv::COLOR_BGR2GRAY);

	//blur(dst, dst, cv::Size(3,3));

	/*
	cv::Mat dest;
	cv::Mat linedst = dst.clone();
	int low_thres = 0;
	int high_thres = 100;	//115
	int multi = 3;
	int kernel = 3;
	Canny(dst, dest, high_thres, high_thres * multi, kernel);
	//Canny(dst, dest, 100, 100);
	*/

	//cv::resize(src, src, cv::Size(), scaling, scaling);
	//imshow("imbus:", src);

	//cv::resize(dst, dst, cv::Size(), 0.2, 0.2);
	//imshow("hsvFilter:", dst);

	//cv::resize(dest, dest, cv::Size(), 0.2, 0.2);
	//imshow("canny:", dest);
}




/*
static void CannyThreshold(int, void*)
{
	blur(src_gray, detected_edges, cv::Size(3, 3));

	detected_edges.convertTo(detected_edges, -1, 1, 50);

	/*
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* p = lookUpTable.ptr();

	int gamma_cor = 100;
	double gamma_value = gamma_cor / 100.0;

	for (int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma_value) * 255.0);
	//cv::Mat res = dst.clone();
	LUT(detected_edges, lookUpTable, detected_edges);
	//imshow("gamma", res);
	*/
	/*
		Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * ratio, kernel_size);

		//cv::Mat edges = detected_edges.clone();


		dst = cv::Scalar::all(0);
		src.copyTo(dst, detected_edges);
		imshow(window_name, dst);
	}
	*/

	/*
		dst.create(src.size(), src.type());
		cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);
		cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
		cv::createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
		CannyThreshold(0, 0);
		*/
//}

/*
cv::Mat src_gray;
cv::Mat detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 200;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

*/





void hugh_lines(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
{
	//probabilistic hough line transformation
	/*
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(dest, lines, 1, CV_PI / 180, 50, 50, 10);
	*/

	//draw lines
	/*
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(line_out, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}
	*/

	cv::Mat dst = src.clone();
	cv::Mat line_out = original.clone();


	//CV_8U type
	cv::Mat t;

	cv::cvtColor(dst, t, cv::COLOR_GRAY2BGR);
	cv::cvtColor(t, dst, cv::COLOR_BGR2GRAY);

	// canny edge detection
/*
cv::Mat cny = dst.clone();
cv::Canny(dst, cny, 50, 200, 23);
imshow("cnny", cny);
*/

	cv::Mat dest;
	cv::Mat linedst = dst.clone();
	int low_thres = 0;
	int high_thres = 140;	//115
	int multi = 3;
	int kernel = 3;
	Canny(dst, dst, high_thres, high_thres * multi, kernel);



	//cv::cvtColor(dest, dst, cv::COLOR_GRAY2BGR);
	line_out = dst.clone();

	//probabilistic hough line transformation
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(dst, lines, 1, CV_PI / 180, 10, 10, 50);

	//draw lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(line_out, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}


	//cv::cvtColor(dst, canny_out, cv::COLOR_BGR2GRAY);
	//cvCvtColor(img, gray, CV_BGR2GRAY);

	//std::vector<std::vector<cv::Point> > contours_poly(contours.size());


	/*
	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);

	}

	cv::RNG rng(12345);
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(dst, contours_poly, (int)i, color);
	}

	std::cout << "cont size: " << contours_poly.size() << std::endl;
	*/






	imshow("lines: ", dst);

	imshow("lines 2: ", line_out);

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




//points rotatedRect
//calc Euclidian distances
/*
{

cv::Point2f points[4];
minRectOnly.at(1).points(points);

double tm_rm_d = cv::norm(points[2] - points[3]);
double rm_bm_d = cv::norm(points[3] - points[0]);
double bm_lm_d = cv::norm(points[0] - points[1]);
double lm_tm_d = cv::norm(points[1] - points[2]);

std::vector<double> lengths;
lengths.push_back(tm_rm_d);
lengths.push_back(rm_bm_d);
lengths.push_back(bm_lm_d);
lengths.push_back(lm_tm_d);

std::sort(lengths.begin(), lengths.end());

std::cout << "lengths" << std::endl;
for (const auto& l : lengths)
{
	std::cout << "length: " << l << std::endl;
}

double width = lengths.at(1);
double length = lengths.at(2);

std::cout << "WIDTH 1: " << width / px_per_mm << std::endl;
std::cout << "LENGTH 1: " << length / px_per_mm << std::endl;
}
*/



//================================
/*
double measurement(cv::Mat src, std::vector<std::vector<cv::Point>> contours, double reference_side_length, cv::Mat original)
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
			boundRect[i] = boundingRect(contours_poly[i]);
		}
	}

	double px_per_mm = 0;

	if (boundRectOnly.size() == 2)
	{
		px_per_mm = calc_px_per_mm(boundRectOnly.at(0).width, boundRectOnly.at(0).height, boundRectOnly.at(1).width, boundRectOnly.at(1).height, reference_side_length);

		double real_width = boundRectOnly.at(0).width / px_per_mm;
		double real_height = boundRectOnly.at(0).height / px_per_mm;

		double real_width1 = boundRectOnly.at(1).width / px_per_mm;
		double real_height1 = boundRectOnly.at(1).height / px_per_mm;

		//std::cout << "rec real width: " << real_width << std::endl;
		//std::cout << "rec real height: " << real_height << std::endl;

		//std::cout << "rec real width 1: " << real_width1 << std::endl;
		//std::cout << "rec real width 1: " << real_height1 << std::endl;
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
/*
	}

	//cv::resize(drawing, drawing, cv::Size(), 0.4, 0.4);
	imshow("Contours method:", drawing);

	return px_per_mm;
}


*/


/*
	//closing to fill small gaps
	cv::Mat element = cv::getStructuringElement(0, cv::Size(2* 2.5 +1, 2*2.5+1 ), cv::Point(2.5, 2.5 ));
	cv::morphologyEx(dst, dst, 1, element);
		//imshow("closing:", dst);

	//erosion to adjust the unneccessary effects of the closing morphology
	cv::Mat el = cv::getStructuringElement(0, cv::Size(2 * 1 + 1, 2 * 1 + 1), cv::Point(1, 1));
	cv::erode(dst, dst, el);
	cv::erode(dst, dst, el);
		//imshow("erosion:", dst);
	*/