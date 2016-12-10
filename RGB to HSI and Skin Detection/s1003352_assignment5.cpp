/*------------------------------------------------------------------------------------------------------------------+
|   s1003352 陳恒劭 | Digital Image Processing Assignment5															|
|   (a) 將輸入圖轉HIS顏色空間,用灰階個別輸出h,s,i                             											|
|   (b) 偵測膚色(膚色,背景區域區分,以單色圖像輸出)                                                          				|
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
void HSI( Mat inputImage );
void skinDection( Mat inputImage );

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

float findMinimum(float num1, float num2, float num3 )
{
	float MinimumVal = 1;
	float allVal[3] = {num1, num2, num3};
	for (int i = 0; i < 3; i++) 
		if (allVal[i] < MinimumVal)
			MinimumVal = allVal[i];

	return MinimumVal;
}

int main()
{
	Mat inputImage, outputImage;
	
   	char fileName[128];
	int selectFunction = 10;

	// Load the original image
	cout << "Input image filename:";
	cin >> fileName;
    inputImage = imread(fileName, CV_LOAD_IMAGE_COLOR);
    
    if( !inputImage.data )  // Check for invalid input
	{
        cout <<  "Could not open or find the image" << endl ;
        system("pause");
        return -1;
    }

	while( selectFunction < 1 || selectFunction > 3 )
	{
		cout << "Select function: \n 1.HSI \n 2.Skin \n ?" ; 
		cin >> selectFunction;
	}

	if( selectFunction == 1 )
		HSI( inputImage );
	else if( selectFunction == 2 )
		skinDection( inputImage );

    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    imshow( pzOriginalImage, inputImage );

    waitKey(0);
    // Wait for a keystroke in the window
    return 0;
}

void HSI( Mat inputImage )
{
	Mat r,g,b;
	r.create(inputImage.rows, inputImage.cols, CV_32F);
	g.create(inputImage.rows, inputImage.cols, CV_32F);
	b.create(inputImage.rows, inputImage.cols, CV_32F);
	
	// 將r,g,b從0~255 --> 0~1 -- normalize
	for (int i = 0; i < inputImage.rows; i++) 
		for (int j = 0; j < inputImage.cols; j++) 
		{
			r.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[2] / 256.0;
			g.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[1] / 256.0;
			b.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[0] / 256.0;
		}

	Mat hue(inputImage.rows, inputImage.cols, CV_32F);
	for (int i = 0; i < inputImage.rows; i++) 
		for (int j = 0; j < inputImage.cols; j++) 
		{
			float firstEquivalent = ( (r.at<float>(i, j)-g.at<float>(i, j)) + (r.at<float>(i, j)-b.at<float>(i, j)) )*0.5;
			float sqEquivalent = sqrt( pow( r.at<float>(i, j) - g.at<float>(i, j), 2 ) + (r.at<float>(i, j)-b.at<float>(i, j))*(g.at<float>(i, j)-b.at<float>(i, j) ) );
			hue.at<float>(i, j) = acos( firstEquivalent / sqEquivalent );

			hue.at<float>(i, j) = hue.at<float>(i, j) * ( 360 / (2*PI) );

			// B > G --> 360 - theta / B <= G --> theta
			if ( b.at<float>(i, j) > g.at<float>(i, j) )
				hue.at<float>(i, j) = 360 - ( hue.at<float>(i, j) / 360 );
			else
				hue.at<float>(i, j) = hue.at<float>(i, j) / 360.0; 
		}
	
	Mat saturationMat(inputImage.rows, inputImage.cols, CV_32F);
	for (int i = 0; i < inputImage.rows; i++)
		for (int j = 0; j < inputImage.cols; j++)
			saturationMat.at<float>(i, j) = ( 1 - ( 3.0*findMinimum(r.at<float>(i, j), g.at<float>(i, j), b.at<float>(i, j)) ) / (r.at<float>(i, j)+g.at<float>(i, j)+b.at<float>(i, j)) );

	Mat intensityMat(inputImage.rows, inputImage.cols, CV_32F);

	for (int i = 0; i < inputImage.rows; i++) 
		for (int j = 0; j < inputImage.cols; j++) 
			intensityMat.at<float>(i, j) = ( r.at<float>(i, j)+g.at<float>(i, j)+b.at<float>(i, j) ) / 3.0;

	imshow("Hue", hue);
	imshow("Saturation", saturationMat);
	imshow("Intensity", intensityMat);
}

void skinDection( Mat inputImage )
{	
	Mat r,g,b;
	r.create(inputImage.rows, inputImage.cols, CV_32F);
	g.create(inputImage.rows, inputImage.cols, CV_32F);
	b.create(inputImage.rows, inputImage.cols, CV_32F);
	
	// 將r,g,b從0~255 --> 0~1 -- normalize
	for (int i = 0; i < inputImage.rows; i++) 
		for (int j = 0; j < inputImage.cols; j++) 
		{
			r.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[2] / 256.0;
			g.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[1] / 256.0;
			b.at<float>(i, j) = inputImage.at<Vec3b>(i, j)[0] / 256.0;
		}
	
	Mat hue(inputImage.rows, inputImage.cols, CV_32F);
	for (int i = 0; i < inputImage.rows; i++) 
		for (int j = 0; j < inputImage.cols; j++) 
		{
			float firstEquivalent = ( (r.at<float>(i, j)-g.at<float>(i, j)) + (r.at<float>(i, j)-b.at<float>(i, j)) )*0.5;
			float sqEquivalent = sqrt( pow( r.at<float>(i, j) - g.at<float>(i, j), 2 ) + (r.at<float>(i, j)-b.at<float>(i, j))*(g.at<float>(i, j)-b.at<float>(i, j) ) );
			hue.at<float>(i, j) = acos( firstEquivalent / sqEquivalent );

			hue.at<float>(i, j) = hue.at<float>(i, j) * ( 360 / (2*PI) );
			
			// B > G --> 360 - theta / B <= G --> theta
			if ( b.at<float>(i, j) > g.at<float>(i, j) )
				hue.at<float>(i, j) = 360 - ( hue.at<float>(i, j) / 360 );
			else
				hue.at<float>(i, j) = hue.at<float>(i, j) / 360.0; 
		}
	
		
	// 轉換Hue之後,根據色相各Range所對應之顏色,在0~360之間取14~40接近人之膚色(紅黃膚色)
	float skinColorRangeBegin = 14, skinColorRangeEnd = 40;
	float rangeBegin = skinColorRangeBegin / 360, rangeEnd = skinColorRangeEnd / 360;

	Mat skinMat(hue.rows, hue.cols, CV_32F); 
	for (int i = 0; i < hue.rows; i++) 
	{
		for (int j = 0; j < hue.cols; j++) 
		{
			if ( hue.at<float>(i, j) >= rangeBegin && hue.at<float>(i, j) < rangeEnd )
				skinMat.at<float>(i, j) = 1;  // 若符合接近皮膚之顏色-->設為白色
			else
				skinMat.at<float>(i, j) = 0;  // 若不符合接近皮膚之顏色-->設為黑色
		}
	}

	imshow("Skin Detection", skinMat);
}