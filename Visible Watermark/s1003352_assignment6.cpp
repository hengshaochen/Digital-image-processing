/*------------------------------------------------------------------------------------------------------------------+
|   s1003352 陳恒劭 | Digital Image Processing Assignment5															|
|   (a) 可見浮水印							                                                          				|
+------------------------------------------------------------------------------------------------------------------*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <iostream>
#include <math.h>
#include <fstream>

using namespace cv;
using namespace std;

#define PI 3.14159265359

const char* pzOriginalImage = "Original Image";
const char* outImage = "Output Image";

void waterMark_DiffSize( Mat inputImage );
void waterMark_SameSize( Mat inputImage );


std::string getImageType(int number)
{
    // find type
    int imgTypeInt = number%8;
    std::string imgTypeString;

    switch (imgTypeInt)
    {
        case 0:
            imgTypeString = "8U";
            break;
        case 1:
            imgTypeString = "8S";
            break;
        case 2:
            imgTypeString = "16U";
            break;
        case 3:
            imgTypeString = "16S";
            break;
        case 4:
            imgTypeString = "32S";
            break;
        case 5:
            imgTypeString = "32F";
            break;
        case 6:
            imgTypeString = "64F";
            break;
        default:
            break;
    }

    // find channel
    int channel = (number/8) + 1;
    std::stringstream type;
    type<<"CV_"<<imgTypeString<<"C"<<channel;

    return type.str();
}



int main()
{
	Mat inputImage, outputImage;
	
	int selectFunction = 10;

	// Load the original image
	//cout << "Input image filename:";
	//cin >> fileName;
    inputImage = imread("a350xwb.jpg", CV_LOAD_IMAGE_COLOR);
    
    if( !inputImage.data )  // Check for invalid input
	{
        cout <<  "Could not open or find the image" << endl ;
        system("pause");
        return -1;
    }

	while( selectFunction < 1 || selectFunction > 3 )
	{
		cout << "Select function: \n 1.waterMark(Different image Size) \n 2.waterMark(Same image Size)\n ?" ; 
		cin >> selectFunction;
	}
	if( selectFunction == 1 )
		waterMark_DiffSize(inputImage);
	else if( selectFunction == 2 )
		waterMark_SameSize(inputImage);


    waitKey(0);
    // Wait for a keystroke in the window
    return 0;
}


void waterMark_DiffSize( Mat intputImage )
{
	Mat mainImage = imread("a350xwb.jpg");  
   // Mat logo = imread("airbus_icon.png");   
    Mat logo = imread("airbus_icon(nonsize).jpg");     
    Mat imageROI = mainImage(Rect(50,50,logo.cols,logo.rows));  

	addWeighted(imageROI,0.8,logo,0.2,0.0,imageROI);  
	
    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    imshow( pzOriginalImage, intputImage );
    namedWindow("OutputImage");  
    imshow("OutputImage",mainImage);  
}


void waterMark_SameSize( Mat inputImage )
{
	Mat mainImage = imread("b787.jpg");  
    Mat waterMark = imread("b787_watermark.jpg");  

    Mat outputImage;  
    outputImage = 0.8*mainImage + 0.2*waterMark;  
	
    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    imshow( pzOriginalImage, mainImage );
    namedWindow("OutputImage");  
    imshow("OutputImage",outputImage); 
}
