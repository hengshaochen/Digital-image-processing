/*----------------------------------------------------------------------
|   s1003352 陳恒劭 | Image Process Program2 
|   (1) 直方圖均化 Histogram Equalization 
|   (2) Sobel Operator 偵測圖像邊緣
+---------------------------------------------------------------------*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <iostream>

using namespace cv;
using namespace std;

const char* pzOriginalImage = "Original Image";
const char* outImage = "Output Image";

string getImgType(Mat frame);
void histogram_equalization( Mat inputImage, Mat &outputImage );
void sobel_operator( Mat inputImage, Mat &outputImage );
void histogram_equalization_bycv( Mat inputImage, Mat &outputImage );
void sobel_operator_bycv( Mat inputImage, Mat &outputImage );

int main()
{
	Mat inputImage, outputImage;
	
   	char fileName[128];
	int selectFunction = 10;

	// Load the original image
	cout << "Input image filename:";
	cin >> fileName;
    inputImage = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
    
    if( !inputImage.data )  // Check for invalid input
	{
        cout <<  "Could not open or find the image" << endl ;
        system("pause");
        return -1;
    }

	while( selectFunction < 1 || selectFunction > 4 )
	{
		cout << "Select function: \n 1.Histogram Equalization \n 2.Sobel Operator \n 3.Histogram Equalization by cvLibrary \n 4.Sobel Operator by cvLibrary \n ?" ; 
		cin >> selectFunction;
	}

	if( selectFunction == 1 )
		histogram_equalization(inputImage, outputImage);
	else if( selectFunction == 2 )
		sobel_operator(inputImage, outputImage);
	else if( selectFunction == 3 )
		histogram_equalization_bycv (inputImage, outputImage);
	else if( selectFunction == 4 )
		sobel_operator_bycv(inputImage, outputImage);
	
    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    namedWindow( outImage, CV_WINDOW_AUTOSIZE );
    // Create a window for display.

    imshow( pzOriginalImage, inputImage );
	imshow( outImage, outputImage );
    // Show our image inside it.

    waitKey(0);
    // Wait for a keystroke in the window
    return 0;
}

string getImgType(Mat frame)
{
    int imgTypeInt = frame.type();
    int numImgTypes = 28; // 7 base types, with 4 channel options each (C1, ..., C4)
    int enum_ints[] = {CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4, CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};
    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4", "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4", "CV_16UC1",  "CV_16UC2",  "CV_16UC3",  "CV_16UC4", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};
    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

void histogram_equalization( Mat inputImage, Mat &outputImage )
{
	// Generate the histogram
	int histogram[256];

	// initialize all intensity values to 0
    for(int i = 0; i < 256; i++)
    {
        histogram[i] = 0;
    }
    // calculate the no of pixels for each intensity values
    for(int y = 0; y < inputImage.rows; y++)
        for(int x = 0; x < inputImage.cols; x++)
            histogram[(int)inputImage.at<uchar>(y,x)]++;

    // Caluculate the size of image
    int size = inputImage.rows * inputImage.cols;
    float alpha = 255.0/size;
 
 
    // Generate cumulative frequency histogram
    int cumhistogram[256];

	cumhistogram[0] = histogram[0];
    for(int i = 1; i < 256; i++)
    {
        cumhistogram[i] = histogram[i] + cumhistogram[i-1];
    }

    // Scale the histogram
    int Sk[256];
    for(int i = 0; i < 256; i++)
    {
        Sk[i] = cvRound((double)cumhistogram[i] * alpha);
    }
 
    // Generate the equlized image
	outputImage = inputImage.clone();
 
    for(int y = 0; y < inputImage.rows; y++)
        for(int x = 0; x < inputImage.cols; x++)
            outputImage.at<uchar>(y,x) = saturate_cast<uchar>(Sk[inputImage.at<uchar>(y,x)]);

	// ------畫input直方圖 ------ //
	int HistogramBins = 256;
	float HistogramRange1[2]={30,200};
	float *HistogramRange[1]={&HistogramRange1[0]};
	
    IplImage *HistogramImage0;
	CvHistogram *inputHistogram;
    inputHistogram = cvCreateHist(1,&HistogramBins,CV_HIST_ARRAY,HistogramRange);
    HistogramImage0 = cvCreateImage(cvSize(256,300),8,3);
    HistogramImage0->origin=1;

	// mat轉ipimage 
	IplImage* ipImage_input = &IplImage(inputImage);
	
    cvCalcHist(&ipImage_input , inputHistogram);

	for(int i=0;i<HistogramBins;i++)
        cvLine(HistogramImage0,cvPoint(i,0),cvPoint(i,(int)(cvQueryHistValue_1D(inputHistogram,i)/10)),CV_RGB(127,127,127));
    
    cvNamedWindow("Histogram(input)",1);
    cvShowImage("Histogram(input)",HistogramImage0);
	// ------畫input直方圖 ------ //

	// ------畫output直方圖 ------ //
	
    IplImage *HistogramImage1;
	CvHistogram *outputHistogram;
    outputHistogram = cvCreateHist(1,&HistogramBins,CV_HIST_ARRAY,HistogramRange);
    HistogramImage1 = cvCreateImage(cvSize(256,300),8,3);
    HistogramImage1->origin=1;

	// mat轉ipimage 
	IplImage* ipImage_output = &IplImage(outputImage);
	
    cvCalcHist(&ipImage_output , outputHistogram);

	for(int i=0;i<HistogramBins;i++)
        cvLine(HistogramImage1,cvPoint(i,0),cvPoint(i,(int)(cvQueryHistValue_1D(outputHistogram,i)/10)),CV_RGB(127,127,127));
    
    cvNamedWindow("Histogram(output)",1);
    cvShowImage("Histogram(output)",HistogramImage1);
	// ------ 畫output直方圖 ------ //
}

void sobel_operator( Mat inputImage, Mat &outputImage )
{
	int gx, gy, sum;
	outputImage = inputImage.clone();

	for(int y = 0; y < inputImage.rows; y++)
		for(int x = 0; x < inputImage.cols; x++)
			outputImage.at<uchar>(y,x) = 0.0;
 
	for(int y = 1; y < inputImage.rows - 1; y++)
	{
		for(int x = 1; x < inputImage.cols - 1; x++)
		{
			gx = inputImage.at<uchar>(y-1, x-1)+ 2*inputImage.at<uchar>(y, x-1)+ inputImage.at<uchar>(y+1, x-1)- inputImage.at<uchar>(y-1, x+1)- 2*inputImage.at<uchar>(y, x+1)- inputImage.at<uchar>(y+1, x+1);
			gy = inputImage.at<uchar>(y-1, x-1)+ 2*inputImage.at<uchar>(y-1, x)+ inputImage.at<uchar>(y-1, x+1)- inputImage.at<uchar>(y+1, x-1)- 2*inputImage.at<uchar>(y+1, x) -inputImage.at<uchar>(y+1, x+1);
			
			sum = abs(gx) + abs(gy);
			if( sum > 255 ) sum = 255;
			if( sum < 0 ) sum = 0;

			outputImage.at<uchar>(y,x) = sum;
		}
	}
}

void histogram_equalization_bycv( Mat inputImage, Mat &outputImage )
{
	outputImage.create( inputImage.rows, inputImage.cols, inputImage.type() );

	// Apply histogram equalization with the function equalizeHist 
	equalizeHist( inputImage, outputImage );

	// ------畫input直方圖 ------ //
	int HistogramBins = 256;
	float HistogramRange1[2]={30,200};
	float *HistogramRange[1]={&HistogramRange1[0]};
	
    IplImage *HistogramImage0;
	CvHistogram *inputHistogram;
    inputHistogram = cvCreateHist(1,&HistogramBins,CV_HIST_ARRAY,HistogramRange);
    HistogramImage0 = cvCreateImage(cvSize(256,300),8,3);
    HistogramImage0->origin=1;

	// mat轉ipimage 
	IplImage* ipImage_input = &IplImage(inputImage);
	
    cvCalcHist(&ipImage_input , inputHistogram);

	for(int i=0;i<HistogramBins;i++)
        cvLine(HistogramImage0,cvPoint(i,0),cvPoint(i,(int)(cvQueryHistValue_1D(inputHistogram,i)/10)),CV_RGB(127,127,127));
    
    cvNamedWindow("Histogram(input)",1);
    cvShowImage("Histogram(input)",HistogramImage0);
	// ------畫input直方圖 ------ //

	// ------畫output直方圖 ------ //
	
    IplImage *HistogramImage1;
	CvHistogram *outputHistogram;
    outputHistogram = cvCreateHist(1,&HistogramBins,CV_HIST_ARRAY,HistogramRange);
    HistogramImage1 = cvCreateImage(cvSize(256,300),8,3);
    HistogramImage1->origin=1;

	// mat轉ipimage 
	IplImage* ipImage_output = &IplImage(outputImage);
	
    cvCalcHist(&ipImage_output , outputHistogram);

	for(int i=0;i<HistogramBins;i++)
        cvLine(HistogramImage1,cvPoint(i,0),cvPoint(i,(int)(cvQueryHistValue_1D(outputHistogram,i)/10)),CV_RGB(127,127,127));
    
    cvNamedWindow("Histogram(output)",1);
    cvShowImage("Histogram(output)",HistogramImage1);
	// ------ 畫output直方圖 ------ //
}

void sobel_operator_bycv( Mat inputImage, Mat &outputImage )
{
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	outputImage.create( inputImage.rows, inputImage.cols, inputImage.type() );
	GaussianBlur( inputImage, inputImage, Size(3,3), 0, 0, BORDER_DEFAULT );

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	Sobel( inputImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	/// Gradient Y
	Sobel( inputImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );

	convertScaleAbs( grad_x, abs_grad_x );
	convertScaleAbs( grad_y, abs_grad_y );
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, outputImage );
	
	//namedWindow( pzRotatedImage, CV_WINDOW_AUTOSIZE );
	//imshow( pzRotatedImage, outputImage );
}


