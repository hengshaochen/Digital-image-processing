/*----------------------------------------------------------------------
|   s1003352 陳恒劭 | Image Process Program1 
|   (1) 大小縮放 - Nearest Neighbor Interpolation 
|   (2) 大小縮放 - Bilinear Interpolation
|   (3) 圖像旋轉 - Image rotation
+---------------------------------------------------------------------*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define PI 3.14159265

const char* pzOriginalImage = "Original Image";
const char* pzScaledImage = "Scaled Image";	
const char* pzRotatedImage = "Rotated Image";

float rowDown, columnDown, rotationDegree;

void nearest_neghbor_interpolation(Mat inputImage, Mat &outputImage);
void bilinear_interpolation(Mat inputImage, Mat &outputImage);
void rotation(Mat inputImage, Mat &outputImage);
string getImgType(Mat frame);
 
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
		cout << "Select function: \n 1.Nearest Neighbor Interpolation \n 2.Bilinear Interpolation \n 3.Image Rotation \n ?" ; 
		cin >> selectFunction;
	}

	// Mode = Nearest Neighbor Interpolation
	if( selectFunction == 1 )
		nearest_neghbor_interpolation(inputImage, outputImage);
	else if( selectFunction == 2 )
		bilinear_interpolation(inputImage, outputImage);
	else if ( selectFunction == 3 )
		rotation(inputImage, outputImage);

    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    namedWindow( pzScaledImage, CV_WINDOW_AUTOSIZE );
    // Create a window for display.

    imshow( pzOriginalImage, inputImage );
    imshow( pzScaledImage, outputImage );
    // Show our image inside it.
 
	//resizeWindow( pzScaledImage, int(inputImage.rows*rowDown+0.5), int(inputImage.cols*columnDown+0.5) );
	// Resize the Window after scaled.

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



void nearest_neghbor_interpolation(Mat inputImage, Mat &outputImage)
{
	cout << "Enter row of scaled down ?" ;
	cin  >> rowDown;
	cout << "Enter column of scaled down ?" ;
	cin  >> columnDown;
	outputImage.create( inputImage.rows*rowDown, inputImage.cols*columnDown, inputImage.type() );

	//cout << "out.cols" << outputImage.cols << "out.rows" << outputImage.rows << endl;
	int newX, newY;  // 經過縮放後 --> 要從舊圖取的x, y座標

	for( int i=0 ; i<outputImage.rows ; i++ )    // i, j 代表新圖的x, y座標
	{
		for( int j=0 ; j<outputImage.cols ; j++ )
		{
			newX = int(i/rowDown + 0.5);
			newY = int(j/columnDown + 0.5);

			//cout << "i,j" << i << "," << j << " " << "newX,newY" << newX << "," << newY << endl;
			if( newX < inputImage.rows && newY < inputImage.cols )
			{
				outputImage.at<Vec3b>( i ,j )[0] = inputImage.at<Vec3b>( newX, newY )[0];
				outputImage.at<Vec3b>( i ,j )[1] = inputImage.at<Vec3b>( newX, newY )[1];
				outputImage.at<Vec3b>( i ,j )[2] = inputImage.at<Vec3b>( newX, newY )[2]; 
			}
		}
	}
}


void bilinear_interpolation(Mat inputImage, Mat &outputImage)
{
	cout << "Enter row of scaled down ?" ;
	cin  >> rowDown;
	cout << "Enter column of scaled down ?" ;
	cin  >> columnDown;	
	outputImage.create( inputImage.rows*rowDown, inputImage.cols*columnDown, inputImage.type() );

	double newX, newY;  // 經過縮放後 --> 要從舊圖取的x, y座標
	float a, b;

	for( int i=0 ; i<outputImage.rows ; i++ )
	{
		for( int j=0 ; j<outputImage.cols ; j++ )
		{
			newX =  i/rowDown;
			newY =  j/columnDown;
			
			a = newY - int(newY);
			b = newX - int(newX);
			
			newX = int(newX);
			newY = int(newY);

			if( newX >= inputImage.rows-1 || newY >= inputImage.cols-1 )
			{
				newX = inputImage.rows -2 ;
				newY = inputImage.cols -2 ;
			}

			outputImage.at<Vec3b>( i ,j )[0] = (1-a)*(1-b)*inputImage.at<Vec3b>( newX, newY )[0] + (1-a)*(b)*inputImage.at<Vec3b>( newX+1, newY )[0] + (a)*(1-b)*inputImage.at<Vec3b>( newX, newY+1 )[0] + (a)*(b)*inputImage.at<Vec3b>( newX+1, newY+1 )[0];
			outputImage.at<Vec3b>( i ,j )[1] = (1-a)*(1-b)*inputImage.at<Vec3b>( newX, newY )[1] + (1-a)*(b)*inputImage.at<Vec3b>( newX+1, newY )[1] + (a)*(1-b)*inputImage.at<Vec3b>( newX, newY+1 )[1] + (a)*(b)*inputImage.at<Vec3b>( newX+1, newY+1 )[1];
			outputImage.at<Vec3b>( i ,j )[2] = (1-a)*(1-b)*inputImage.at<Vec3b>( newX, newY )[2] + (1-a)*(b)*inputImage.at<Vec3b>( newX+1, newY )[2] + (a)*(1-b)*inputImage.at<Vec3b>( newX, newY+1 )[2] + (a)*(b)*inputImage.at<Vec3b>( newX+1, newY+1 )[2];
		}	
			//system("PAUSE");
	}


}


void rotation(Mat inputImage, Mat &outputImage)
{
	outputImage.create( inputImage.rows, inputImage.cols, inputImage.type() );
    int newX, newY;
    // 假設對中心進行旋轉
    double COS, SIN;

    int ox = ( inputImage.rows-1 )/2;
    int oy = ( inputImage.cols-1 )/2;    
    int nx2, ny2; //平移後之點

	cout << "Enter degree of rotation ?" ;
	cin  >> rotationDegree;

    SIN = sin( rotationDegree * PI / 180.0 ), COS = cos( rotationDegree * PI / 180.0 );


	for( int i=0 ; i<inputImage.rows ; i++ )
	{
		for( int j=0 ; j<inputImage.cols ; j++ )
		{
            // 平移 ox,oy
            nx2 = i - ox, ny2 = j - oy;

            // 再旋轉, 平移(-ox,-oy)
            newX = (int)( nx2 * COS + ny2 * SIN + 0.5 + ox);
            newY = (int)(-nx2 * SIN + ny2 * COS + 0.5 + oy);

			if( newY>=0 && newY< inputImage.cols && newX>=0 && newX < inputImage.rows )
			{
				outputImage.at<Vec3b>( i, j )[0] = inputImage.at<Vec3b>( newX, newY )[0];
				outputImage.at<Vec3b>( i, j )[1] = inputImage.at<Vec3b>( newX, newY )[1];
				outputImage.at<Vec3b>( i, j )[2] = inputImage.at<Vec3b>( newX, newY )[2];
			}
		}
	}
}

/*
void rotation(Mat inputImage, Mat &outputImage)
{
	cout << "Enter degree of rotation ?" ;
	cin  >> rotationDegree;

	outputImage.at<Vec3b>( 20, 25 );

	for( int i=0 ; i<inputImage.rows ; i++ )
	{
		for( int j=0 ; j<inputImage.cols ; j++ )
		{
			//cout << "x:" <<  i*cos( rotationDegree * PI / 180.0 ) - j*sin( rotationDegree * PI / 180.0 ) << endl
			//	 << "y:" <<  i*sin( rotationDegree * PI / 180.0 ) + j*cos( rotationDegree * PI / 180.0 ) << endl;

			int newX = int( i*cos( rotationDegree * PI / 180.0 ) - j*sin( rotationDegree * PI / 180.0 ) +0.5 );
			int newY = int( i*cos( rotationDegree * PI / 180.0 ) + j*sin( rotationDegree * PI / 180.0 ) +0.5 );
			//int newY = int( i*sin( rotationDegree * PI / 180.0 ) + j*cos( rotationDegree * PI / 180.0 )  +0.5 );

			//cout << "newX:" << newX << "  " << "newY:" << newY << endl;
			//if( newY>=0 && newY<j && newX>=0 && newX<i )
			if( newY>=0 && newY < inputImage.cols && newX>=0 && newX < inputImage.rows )
			{
				//cout << "newX:" << newX << "  " << "newY:" << newY << endl;
				outputImage.at<Vec3b>( i, j )[0] = inputImage.at<Vec3b>( i, j )[0];
				outputImage.at<Vec3b>( i, j )[1] = inputImage.at<Vec3b>( i, j )[1];
				outputImage.at<Vec3b>( i, j )[2] = inputImage.at<Vec3b>( i, j )[2];
			}
			else
			{
				//cout << "-----newX:" << newX << "  " << "newY:" << newY << endl;
				//system("PAUSE");
			}
		}
	}
}

*/