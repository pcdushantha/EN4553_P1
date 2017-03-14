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
#include"exif.h"
#include"exif.cpp"

using namespace std;
using namespace cv;

/**
* @function main
*/

void Histogram(Mat* src, Mat* dst, Mat* yuv);
void GaussianSharpning(Mat* src, Mat* Gimage, Mat* Simage);
void LaplacianSharpning(Mat* src, Mat* Limage, Mat* dst);
int metadata();

int main(int, char** argv)
{
	Mat src, dst,yuv,Gimage,Simage,Limage;

	/// Load image
	src = imread(argv[1], 1);

	if (src.empty())
	{
		cout << "No image" << endl;
		waitKey(0);
		return -1;
	}
	//Histogram(&src, &dst, &yuv);
	//Sharpning(&src, &Gimage, &Simage);
	//LaplacianSharpning(&src, &Limage, &dst);
	metadata();


	return 0;

}


void Histogram(Mat* src, Mat* dst, Mat* yuv) {

	cvtColor(*src, *dst, COLOR_BGR2GRAY);
	

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	

	split(*src, bgr_planes);
	

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	
	Mat b_hist, g_hist, r_hist, hist;


																					 /// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(dst, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

	Mat histnew(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

														  /// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(hist, hist, 0, histnew.rows, NORM_MINMAX, -1, Mat()); 

																	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0), 1, LINE_AA, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 1, LINE_AA, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255), 1, LINE_AA, 0);

		
		line(histnew, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(0, 0, 0), 1, LINE_AA, 0);  


	}

	/// Display
	namedWindow("calcHist Demo", WINDOW_NORMAL);
	imshow("calcHist Demo", histImage);

	namedWindow("Grayscale", WINDOW_NORMAL);
	imshow("Grayscale", histnew);

	namedWindow("GRAY", WINDOW_NORMAL);
	imshow("GRAY", *dst);
	

	namedWindow("Original", WINDOW_NORMAL);
	imshow("Original", *src);
	
	waitKey(0);

}

void GaussianSharpning(Mat* src, Mat* Gimage, Mat* Simage) {

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

void LaplacianSharpning(Mat* src, Mat* Limage, Mat* dst) {
	
	//cvtColor(*src, *dst, COLOR_BGR2GRAY);
	Laplacian(*src, *Limage, CV_16S, 3, 1, 0);
	convertScaleAbs(*Limage, *Limage);

	addWeighted(*src, 0.8, *Limage, 0.2, 0, *Limage);

	namedWindow("LAPLACIAN", WINDOW_NORMAL);
	imshow("LAPLACIAN", *Limage);
	waitKey(0);
}

int metadata() {

	FILE *fp = fopen("C:/Users/Charith/Documents/Visual Studio 2015/Projects/HistogramTest/data/image.jpg", "rb");
	if (!fp) {
		cout<<"Can't open file.\n"<<endl;
		return -1;
	}
	fseek(fp, 0, SEEK_END);
	unsigned long fsize = ftell(fp);
	rewind(fp);
	unsigned char *buf = new unsigned char[fsize];
	if (fread(buf, 1, fsize, fp) != fsize) {
		cout<<"Can't read file.\n" << endl;
		delete[] buf;
		return -2;
	}
	fclose(fp);

	// Parse EXIF
	easyexif::EXIFInfo result;
	int code = result.parseFrom(buf, fsize);
	delete[] buf;
	if (code) {
		cout<<"Error parsing EXIF: code %d\n"<< code << endl;
		return -3;
	}

	// Dump EXIF information
	cout<<"Camera make          :"<< result.Make.c_str() << endl;
	cout<<"Camera model         :" << result.Model.c_str() << endl;
	cout<<"Software             :" << result.Software.c_str() << endl;
	cout<<"Bits per sample      :" << result.BitsPerSample << endl;
	cout<<"Image width          :"<< result.ImageWidth << endl;
	cout<<"Image height         :" << result.ImageHeight << endl;
	cout<<"Image description    :" << result.ImageDescription.c_str() << endl;
	cout<<"Image orientation    :" << result.Orientation << endl;
	cout<<"Image copyright      :" << result.Copyright.c_str() << endl;
	cout<<"Image date/time      :" << result.DateTime.c_str() << endl;
	cout<<"Original date/time   :" << result.DateTimeOriginal.c_str() << endl;
	cout<<"Digitize date/time   :" << result.DateTimeDigitized.c_str() << endl;
	cout<<"Subsecond time       :" << result.SubSecTimeOriginal.c_str() << endl;
	cout<<"Exposure time        :" <<
		(unsigned)(1.0 / result.ExposureTime) << endl;
	cout<<"F-stop               : " << result.FNumber << endl;
	cout<<"ISO speed            : " << result.ISOSpeedRatings << endl;
	cout<<"Subject distance     : " << result.SubjectDistance << endl;
	cout<<"Exposure bias        : " << result.ExposureBiasValue << endl;
	cout<<"Flash used?          : " << result.Flash << endl;
	cout<<"Metering mode        : " << result.MeteringMode << endl;
	cout<<"Lens focal length    : " << result.FocalLength << endl;
	cout<<"35mm focal length    : " << result.FocalLengthIn35mm << endl;
	cout<<"GPS Latitude         : " <<
		result.GeoLocation.Latitude << result.GeoLocation.LatComponents.degrees <<
		result.GeoLocation.LatComponents.minutes <<
		result.GeoLocation.LatComponents.seconds <<
		result.GeoLocation.LatComponents.direction << endl;
	cout<<"GPS Longitude        : " <<
		result.GeoLocation.Longitude << result.GeoLocation.LonComponents.degrees <<
		result.GeoLocation.LonComponents.minutes <<
		result.GeoLocation.LonComponents.seconds <<
		result.GeoLocation.LonComponents.direction << endl;
	cout<<"GPS Altitude         : " << result.GeoLocation.Altitude << endl;
	cout<<"GPS Precision (DOP)  : " << result.GeoLocation.DOP << endl;
	cout<<"Lens min focal length: " << result.LensInfo.FocalLengthMin << endl;
	cout<<"Lens max focal length: " << result.LensInfo.FocalLengthMax << endl;
	cout<<"Lens f-stop min      : " << result.LensInfo.FStopMin << endl;
	cout<<"Lens f-stop max      : " << result.LensInfo.FStopMax << endl;
	cout<<"Lens make            : " << result.LensInfo.Make.c_str() << endl;
	cout<<"Lens model           : " << result.LensInfo.Model.c_str() << endl;
	cout<<"Focal plane XRes     : " << result.LensInfo.FocalPlaneXResolution << endl;
	cout<<"Focal plane YRes     : " << result.LensInfo.FocalPlaneYResolution << endl;
	waitKey(0);
	return 0;
}