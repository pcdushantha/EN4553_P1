/**
* @function calcHist_Demo.cpp
* @brief Demo code to use the function calcHist
* @author
*/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/**
* @function main
*/

void Histogram(Mat* src, Mat* dst, Mat* yuv);
void Sharpning(Mat* src, Mat* Gimage, Mat* Simage);

int main(int, char** argv)
{
	Mat src, dst,yuv,Gimage,Simage;

	/// Load image
	src = imread(argv[1], 1);

	if (src.empty())
	{
		cout << "No image" << endl;
		waitKey(0);
		return -1;
	}
	//Histogram(&src, &dst, &yuv);
	Sharpning(&src, &Gimage, &Simage);

	return 0;

}


void Histogram(Mat* src, Mat* dst, Mat* yuv) {

	cvtColor(*src, *dst, COLOR_BGR2GRAY);

	cvtColor(*src, *yuv, COLOR_BGR2YUV);

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	vector<Mat> yuv_planes;




	split(*src, bgr_planes);
	//split(yuv, yuv_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	/*
	Mat hist; //new
	int imgCount = 1;
	int dims = 2;
	const int sizes[] = { 256,256,256 };
	const int channels[] = { 0,1,2 };
	float rRange[] = { 0,256 };
	float gRange[] = { 0,256 };
	float bRange[] = { 0,256 };
	const float *ranges[] = { rRange,gRange,bRange };
	Mat mask = Mat();  //new

	calcHist(&src, imgCount, channels, mask, hist, dims, sizes, ranges);
	*/
	Mat b_hist, g_hist, r_hist, hist;

	calcHist(dst, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);//NEW



																					 /// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	//Mat histnew(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));//new
	Mat histnew(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));//new

														  /// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//normalize(hist, hist, 0, histnew.rows, NORM_MINMAX, -1, Mat()); //new
	normalize(hist, hist, 0, histnew.rows, NORM_MINMAX, -1, Mat()); //new

																	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);

		/*	line(histnew, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
		Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
		Scalar(255, 255, 255), 2, 8, 0);  //new
		*/
		line(histnew, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);  //new


	}

	/// Display
	namedWindow("calcHist Demo", WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", histnew);

	namedWindow("GRAY", WINDOW_NORMAL);
	imshow("GRAY", *dst);
	/*
	namedWindow("Y", WINDOW_NORMAL);
	imshow("Y", yuv_planes[0]);
	namedWindow("U", WINDOW_NORMAL);
	imshow("U", yuv_planes[1]);
	namedWindow("V", WINDOW_NORMAL);
	imshow("V", yuv_planes[2]);
	*/

	namedWindow("Original", WINDOW_NORMAL);
	imshow("Original", *src);
	/*
	namedWindow("yuv", WINDOW_NORMAL);
	imshow("yuv", yuv);
	*/
	waitKey(0);

}

void Sharpning(Mat* src, Mat* Gimage, Mat* Simage) {

	GaussianBlur(*src, *Gimage, Size(0, 0), 12);
	addWeighted(*src, 1.5, *Gimage, -0.5, 0, *Simage);

	namedWindow("ORIGINAL", WINDOW_NORMAL);
	imshow("ORIGINAL", *src);

	namedWindow("BLUR", WINDOW_NORMAL);
	imshow("BLUR", *Gimage);
	namedWindow("SHARP", WINDOW_NORMAL);
	imshow("SHARP", *Simage);

	waitKey(0);

}