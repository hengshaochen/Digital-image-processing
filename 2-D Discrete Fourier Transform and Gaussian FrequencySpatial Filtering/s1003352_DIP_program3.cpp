/*------------------------------------------------------------------------------------------------------------------+
|   s1003352 陳恒劭 | Digital Image Processing																		|
|   (a) 將輸入圖片做"離散傅立葉"轉換 , 並將"頻譜大小" , "相位角度"以Gray256色圖像呈現											|
|   (b) 在"空間域"和"頻域" 實做 Gaussian 平滑濾波器(必須可調整濾波器的標準差參數,濾波器大小)(filter size 3x5,5x5...)				|
+------------------------------------------------------------------------------------------------------------------*/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv.h>
#include <iostream>
#include <math.h>
#include <fstream>

#define PI 3.1415926535897932384

using namespace cv;
using namespace std;

const char* pzOriginalImage = "Original Image";
const char* outImage = "Output Image";
const char* Spectrum_by_MYSELF = "Spectrum_by_MYSELF";
const char* Spectrum_by_CV = "Spectrum_by_CV";

Mat backup_spectrum;

void gaussian_bycv( Mat inputImage, Mat &outputImage, int multiplicationRow, int multiplicationColumn );
void real_dft( Mat inputImage, Mat &outputImage, double dValue );
void fileTesting( Mat output, char fileName[] );
void SpatialDomain(Mat &inputImage, int sptial_p, int sptial_q, float sptial_X, float sptial_Y ) ;

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
	
   	char fileName[128];
	int selectFunction = 10;
	int multiplicationRow = 3;
	int multiplicationColumn = 3;

	int sptial_p, sptial_q; 
	float sptial_X, sptial_Y; 

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

	while( selectFunction < 1 || selectFunction > 3 )
	{
		cout << "Select function: \n 1.DFT( three output mySelf and OpenCV and Frequency Domain ) \n 2.Spatial Domain \n ?" ; 
		cin >> selectFunction;
	}

	if( selectFunction == 1 )
	{
		double dValue;
        // user specify sigma for creating the filter
        cout<<"請輸入frequency domain之D0值:"<<endl;
        cin>>dValue;
		real_dft(inputImage, outputImage, dValue );
	}
	else if( selectFunction == 2 )
	{
		cout<<"Mask Size PxQ(ex:3X3 ..)\n"; 
		cout<<"P: "; 
		cin>>sptial_p; 
		cout<<"Q: "; 
		cin>>sptial_q; 
		cout<<"sptial_X: "; 
		cin>>sptial_X; 
		cout<<"sptial_Y: "; 
		cin>>sptial_Y; 
		SpatialDomain( inputImage, sptial_p, sptial_q, sptial_X, sptial_Y );
	}



    namedWindow( pzOriginalImage, CV_WINDOW_AUTOSIZE );
    imshow( pzOriginalImage, inputImage );

    waitKey(0);
    // Wait for a keystroke in the window
    return 0;
}


float eulerBeautiful_for_real( float theta )
{
	return cos( theta );
}
float eulerBeautiful_for_fake( float theta )
{
	return sin( theta );
}


void quadrantShift( Mat& src, Mat& dst) {
    

    src = src(Rect(0, 0, src.cols & -2, src.rows & -2));
    int cx = src.cols/2;
    int cy = src.rows/2;
    
    Rect q0 = Rect(0, 0, cx, cy);   // Top-Left - Create a ROI per quadrant
    Rect q1 = Rect(cx, 0, cx, cy);  // Top-Right
    Rect q2 = Rect(0, cy, cx, cy);  // Bottom-Left
    Rect q3 = Rect(cx, cy, cx, cy); // Bottom-Right
    
    Mat temp;   // creating a temporary Mat object to protect the quadrant, in order to handle the situation where src = dst
    
    src(q0).copyTo(temp);   // preserve q0 section
    src(q3).copyTo(dst(q0));
    temp.copyTo(dst(q3));   // swap q0 and q3
    
    src(q1).copyTo(temp);
    src(q2).copyTo(dst(q1));
    temp.copyTo(dst(q2));
    
}

Mat createGaussianFilter( Size size_of_filter, double sigma, bool highpass_flag ) {
    
    Mat gaussian_filter = Mat(size_of_filter, CV_32F),
    filter_x = getGaussianKernel(size_of_filter.height, sigma, CV_32F),
    filter_y = getGaussianKernel(size_of_filter.width, sigma, CV_32F);
    
    // this will create filter as Mat object of which size is x*y
    gaussian_filter = filter_x * filter_y.t();
    normalize(gaussian_filter, gaussian_filter, 0, 1, CV_MINMAX);
    
    if (highpass_flag == true) gaussian_filter = 1 - gaussian_filter;
    
    Mat to_merge[] = {gaussian_filter, gaussian_filter};
    merge(to_merge, 2, gaussian_filter);
    // the filter is used to process spetrums before quadrant shift, so:
    quadrantShift(gaussian_filter, gaussian_filter);
    return gaussian_filter;
    
}

void DFT(Mat& src, Mat& dst) {
    
    // expand the source image to optimal size for dft
    copyMakeBorder(src, dst,
                   0, getOptimalDFTSize( src.rows ) - src.rows,
                   0, getOptimalDFTSize( src.cols ) - src.cols,
                   BORDER_ISOLATED);
    
    // create a plane containing 2 mat layer to form a 2-channel mat object
    Mat planes[] = {Mat_<float>(dst), Mat::zeros(dst.size(), CV_32F)};
    // this is the 2-channel object that I was talking about
    merge(planes, 2, dst);
    // dft result will be stored in dst, in which two channels holds separately real and imaginary components
    dft(dst, dst);
    
}

Mat visualDFT( Mat& dft_result ) {
    
    Mat dst;
    // create a plane containing 2 mat layer to form a 2-channel mat object
    Mat planes[2];
    // in order to calculate the magnitude, we'll have to split the image by channel in order to obtain each component
    split(dft_result, planes);
    magnitude(planes[0], planes[1], dst);
    
    // switch to logarithmic scale
    dst += Scalar::all(1);
    log(dst, dst);
    
    normalize(dst, dst,0,1,CV_MINMAX);
    //    cout<<dst<<"dst end";
    quadrantShift(dst, dst);
    return dst;
    
}


void inverseDFT(Mat& dft_result, Mat& dst) {
    
    dft(dft_result, dst, DFT_INVERSE|DFT_REAL_OUTPUT);
    normalize(dst, dst, 0, 1, CV_MINMAX);
    
}

void real_dft( Mat inputImage, Mat &outputImage, double dValue )
{
	// 將原本的input copy 一份到ouputImage
	outputImage = inputImage.clone();

	// 將影像的rows 和 cols 各放大兩倍, 並且補0(padding)
	Mat padded;    // 將outputImage + padding 後(包含補零)的結果儲存於這陣列 / 設定padded的型態轉成從unchar --> float
	padded.create( outputImage.rows*2, outputImage.cols*2, CV_32FC1 );

	padded.setTo(Scalar::all(0));

	// 將outputImage的每個Pixel放入padded中
	for( int i=0 ; i<outputImage.rows ; i++ )	
		for( int j=0 ; j<outputImage.cols ; j++ )
			padded.at<float>( i, j ) = outputImage.at<uchar>( i, j );

	// 將奇數Pixel乘(-1)^(padded.rows + padded.cols ) --> To center its transform
	for( int i=0 ; i<padded.rows ; i++ )
		for( int j=0 ; j<padded.cols ; j++ )
			if( (i + j )%2 != 0 )
				padded.at<float>( i, j ) = padded.at<float>( i, j ) * (-1);
	
	// 將padded做DFT轉換(轉換結果將存於dft_result)
	Mat dft_result_real, dft_result_fake;
	dft_result_real.create( padded.rows, padded.cols, CV_32FC1 );
	dft_result_fake.create( padded.rows, padded.cols, CV_32FC1 );
	// dft_byCv --> 這個變數使用openCV內建DFT
	Mat dft_byCv;
	dft_byCv.create( padded.rows, padded.cols, CV_32FC2 );

	// 先初始化 皆預設0
	for( int i=0 ; i<dft_result_real.rows ; i++ )
		for( int j=0 ; j <dft_result_real.cols ; j++ )
		{
			dft_result_real.at<float>( i, j ) = 0;
			dft_result_fake.at<float>( i, j ) = 0;
			dft_byCv.at<Vec2f>( i, j )[0] = padded.at<float>( i , j ) ;  // 實部
			dft_byCv.at<Vec2f>( i, j )[1] = 0;
		}

	dft( dft_byCv, dft_byCv );         // Mat dft_byCv使用openCV內建LIBRARY
	
	// ---------------------------Mat dft_result_real 使用 自己實做的DFT ---------------------------//
	for( float i=0 ; i<padded.rows ; i++ )
	{
		for( float j=0 ; j<padded.cols ; j++ )
		{
			for( float sigma_i=0 ; sigma_i <= padded.rows-1 ; sigma_i++ )
			{
				for( float sigma_j=0 ; sigma_j <= padded.cols-1 ; sigma_j++ )
				{
					dft_result_real.at<float>( i, j ) = dft_result_real.at<float>( i, j ) + ( padded.at<float>( sigma_i, sigma_j ) * eulerBeautiful_for_real(-2 * PI *( ( i*sigma_i / padded.rows) + ( j*sigma_j / padded.cols ) ) ) );	
					dft_result_fake.at<float>( i, j ) = dft_result_fake.at<float>( i, j ) + ( padded.at<float>( sigma_i, sigma_j ) * eulerBeautiful_for_fake(-2 * PI *( ( i*sigma_i / padded.rows) + ( j*sigma_j / padded.cols ) ) ) );
				
				}
			}
		}
	}
	// ---------------------------Mat dft_result_real 使用 自己實做的DFT ---------------------------//

	// 備份自己DFT轉換完的結果
	Mat backup_dft_result_real, backup_dft_fake;
	backup_dft_result_real = dft_result_real.clone();
	backup_dft_fake = dft_result_fake.clone();

	// 將DFT實做出來的實部,虛部 做 Phase angle
	Mat phaseAngle = padded.clone();
	for( int i=0 ; i<backup_dft_result_real.rows ; i++ )  // 實部, 虛部 各別平方
		for( int j=0 ; j<backup_dft_result_real.cols ; j++ )
			phaseAngle.at<float>( i, j ) = atan( backup_dft_fake.at<float>( i, j ) / backup_dft_result_real.at<float>( i, j ) );

	// 將DFT出來的實部,虛部做 Spectrum 做成頻譜圖
	for( int i=0 ; i<dft_result_real.rows ; i++ )  // 實部, 虛部 各別平方
	{
		for( int j=0 ; j<dft_result_real.cols ; j++ )
		{
			dft_result_real.at<float>( i ,j ) = pow( dft_result_real.at<float>( i ,j ), 2 );
			dft_result_fake.at<float>( i ,j ) = pow( dft_result_fake.at<float>( i ,j ), 2 );

			dft_byCv.at<Vec2f>( i, j )[0] = pow( dft_byCv.at<Vec2f>( i, j )[0], 2 );  
			dft_byCv.at<Vec2f>( i, j )[1] = pow( dft_byCv.at<Vec2f>( i, j )[1], 2 );
		}
	}

	// 存最後結果圖 Spectrum --> 自己實做的 / dft_byCv_Spectrum --> byOpenCV Library 的結果
	Mat Spectrum = padded.clone();
	Mat dft_byCv_Spectrum = padded.clone();

	for( int i=0 ; i<Spectrum.rows ; i++ )  // 實部, 虛部 各別平方
		for( int j=0 ; j<Spectrum.cols ; j++ )
		{
			Spectrum.at<float>( i ,j ) = sqrt( dft_result_real.at<float>( i ,j ) + dft_result_fake.at<float>( i ,j ) ) ;
			dft_byCv_Spectrum.at<float>( i, j ) = sqrt( dft_byCv.at<Vec2f>( i, j )[0] + dft_byCv.at<Vec2f>( i, j )[1] );
		}

	// 取值介於0-1之間
    normalize(Spectrum, Spectrum, 0, 1, CV_MINMAX);						// Transform the matrix with float values into a
    normalize(dft_byCv_Spectrum, dft_byCv_Spectrum, 0, 1, CV_MINMAX);	// viewable image form (float between values 0 and 1)
    normalize(phaseAngle, phaseAngle, 0, 1, CV_MINMAX);	

	// 備份一份Spectrum給高斯使用
	backup_spectrum = Spectrum.clone();


	// Frequency Domain
	
	// Step5 : 算H(u,v)
	const int D0 = 10;
	Mat H;
	H = backup_spectrum.clone();

	int P = backup_spectrum.rows ;
	int Q = backup_spectrum.cols ;

	for( int i=0 ; i<backup_spectrum.rows ; i++ )
		for( int j=0 ; j<backup_spectrum.cols ; j++ )
			H.at<float>( i, j ) =  exp( ( 0 - ( pow ( sqrt( pow( ( i - ( P/2 ) ), 2 ) + pow( ( j - ( Q/2 ) ), 2 ) ), 2 ) ) ) / ( 2* pow(D0, 2 ) ) );

	Mat dft_container,result;
	DFT(inputImage, dft_container);
	visualDFT(dft_container);
	Mat gaussian_filter;
	gaussian_filter = createGaussianFilter(dft_container.size(), dValue, false);
	mulSpectrums(dft_container, gaussian_filter, dft_container, DFT_ROWS);

	inverseDFT(dft_container, result);

	// Step5 : 算G(u,v) = F(u,v) * H(u,v)
	Mat G_real, G_fake;
	G_real.create( backup_spectrum.rows , backup_spectrum.cols , CV_32FC1 );
	G_fake.create( backup_spectrum.rows , backup_spectrum.cols , CV_32FC1 );

	for( int i=0 ; i < backup_dft_result_real.rows ; i++ )
		for( int j=0 ; j< backup_dft_result_real.cols ; j++ )
		{
			//G_real.at<float>( i ,j ) = H.at<float>( i, j ) * backup_dft_result_real.at<float>( i, j );
			G_real.at<float>( i ,j ) = H.at<float>( i, j ) * backup_dft_result_real.at<float>( i, j );
			G_fake.at<float>( i ,j ) = H.at<float>( i, j ) * backup_dft_fake.at<float>( i, j );
		}


	Mat after_IDFT_real, after_IDFT_fake;
	after_IDFT_real.create( G_real.rows , G_real.cols , CV_32FC1 );
	after_IDFT_fake.create( G_fake.rows , G_fake.cols , CV_32FC1 );
	// 先初始化 皆預設0
	for( int i=0 ; i<after_IDFT_real.rows ; i++ )
		for( int j=0 ; j <after_IDFT_real.cols ; j++ )
		{
			after_IDFT_real.at<float>( i, j ) = 0;
			after_IDFT_fake.at<float>( i, j ) = 0;
		}


	// Step6 : 將G(u,v)算Inverse Fourier Transform / IDFT
	for( float i=0 ; i<after_IDFT_real.rows ; i++ )
	{
		for( float j=0 ; j<after_IDFT_real.cols ; j++ )
		{
			for( float sigma_i=0 ; sigma_i <= after_IDFT_real.rows-1 ; sigma_i++ )
			{
				for( float sigma_j=0 ; sigma_j <= after_IDFT_real.cols-1 ; sigma_j++ )
				{
					after_IDFT_real.at<float>( i, j ) = after_IDFT_real.at<float>( i, j ) + ( G_real.at<float>( sigma_i, sigma_j ) * eulerBeautiful_for_real(2 * PI *( ( i*sigma_i / G_real.rows) + ( j*sigma_j / G_real.cols ) ) ) );	
					after_IDFT_fake.at<float>( i, j ) = after_IDFT_fake.at<float>( i, j ) + ( G_real.at<float>( sigma_i, sigma_j ) * eulerBeautiful_for_fake(2 * PI *( ( i*sigma_i / G_real.rows) + ( j*sigma_j / G_real.cols ) ) ) );
				}
			}
		}
	}
	
	for( float i=0 ; i<after_IDFT_real.rows ; i++ )
		for( float j=0 ; j<after_IDFT_real.cols ; j++ )
		{
			after_IDFT_real.at<float>( i, j ) =  after_IDFT_real.at<float>( i, j ) * ( ( (float)1 / (after_IDFT_real.rows*after_IDFT_real.cols) ) );
			after_IDFT_fake.at<float>( i, j ) =  after_IDFT_fake.at<float>( i, j ) * ( ( (float)1 / (after_IDFT_fake.rows*after_IDFT_fake.cols) ) );
		}
		
	fileTesting( after_IDFT_real, "(OK)after_IDFT_real(u,v).txt" );
	fileTesting( after_IDFT_fake, "(OK)after_IDFT_fake(u,v).txt" );
	
	// 將結果每個pixel * -1^(x+y)
	for( float i=0 ; i<after_IDFT_real.rows ; i++ )
		for( float j=0 ; j<after_IDFT_real.cols ; j++ )
		{
				after_IDFT_real.at<float>( i, j ) =  after_IDFT_real.at<float>( i, j ) * pow(-1, after_IDFT_real.rows + after_IDFT_real.cols ) ;
				after_IDFT_fake.at<float>( i, j ) =  after_IDFT_fake.at<float>( i, j ) * pow(-1, after_IDFT_fake.rows + after_IDFT_fake.cols ) ;
		}
	

	// 顯示 頻譜Spectrum(myself and bycv) 相位圖phaseAngle
    namedWindow( Spectrum_by_MYSELF, CV_WINDOW_AUTOSIZE );
	imshow( Spectrum_by_MYSELF, Spectrum );
	
    namedWindow( "phaseAngle", CV_WINDOW_AUTOSIZE );
	imshow( "phaseAngle", phaseAngle );
	
    namedWindow( Spectrum_by_CV, CV_WINDOW_AUTOSIZE );
	imshow( Spectrum_by_CV, dft_byCv_Spectrum );

	namedWindow( "H", CV_WINDOW_AUTOSIZE );
	imshow( "H", H );

	namedWindow( "Frequency Domain", CV_WINDOW_AUTOSIZE );
	imshow( "Frequency Domain", result );

}


void fileTesting( Mat output, char fileName[] )
{
    fstream fp5;
    fp5.open(fileName, ios::out);//開啟檔案
    if(!fp5){//如果開啟檔案失敗，fp為0；成功，fp為非0
        cout<<"Fail to open file: "<<fileName<<endl;
    }
 	for (int i=0; i< output.rows ; i++)
	{
		for (int j=0; j< output.cols ; ++j)
		{
			fp5 << "output.at<float>( " << i << "," << j << "):" << output.at<float>( i ,j );
			fp5 << endl;
		}
	}
    fp5.close();//關閉檔案
}


void SpatialDomain(Mat &inputImage, int sptial_p, int sptial_q, float sptial_X, float sptial_Y ) 
{ 
    Mat outputImage = inputImage.clone(); 
  
    float **mask = new float *[sptial_p]; 
    for(int i=0; i<sptial_p; i++) 
        mask[i] = new float[sptial_q]; 
  
    int radiusX = sptial_p / 2; 
    int radiusY = sptial_q / 2; 
    float r, s = 2.0 * sptial_X * sptial_Y; 
    float sum = 0.0; 
  
    for(int x=-radiusX; x<=radiusX; x++) 
        for(int y=-radiusY; y<=radiusY ;y++) 
        { 
            r = x*x + y*y; 
            mask[x+radiusX][y+radiusY] = (exp(-r/s))/(PI*s); 
            sum += mask[x+radiusX][y+radiusY]; 
        }  
  
    for(int i=0; i<sptial_p; i++) 
        for(int j=0; j<sptial_q; j++) 
            mask[i][j] /= sum; 

  
    // padding 
    const int top = sptial_p/2, bottom = sptial_q/2, left = sptial_p/2, right = sptial_q/2, borderType = 0; 
    copyMakeBorder(inputImage, outputImage, top, bottom, left, right, borderType); 
      
    int dstRow = outputImage.rows, dstCol = outputImage.cols, rowStart = sptial_p/2, colStart = sptial_q/2; 
  
    // convolution 
    for (int row = rowStart; row < dstRow-rowStart; row++)  
        for (int col = colStart; col < dstCol-colStart; col++)  
        { 
            double pixelValue = 0; 
            for (int filterRow = 0; filterRow < sptial_p; filterRow++)  
                for (int filterCol = 0; filterCol < sptial_q; filterCol++)             
                      pixelValue += outputImage.at<uchar>(row-rowStart+filterRow, col-colStart+filterCol) * mask[filterRow][filterCol];

            outputImage.at<uchar>(row, col) = pixelValue; 
        } 
  
    imshow( "InputImage", inputImage ); 
    imshow( "OutputImage", outputImage); 
  
    waitKey(0); 
} 



void gaussian_bycv( Mat inputImage, Mat &outputImage, int multiplicationRow, int multiplicationColumn )
{
	outputImage.create( inputImage.rows, inputImage.cols, inputImage.type() );
	GaussianBlur( inputImage, outputImage, Size( multiplicationRow, multiplicationColumn), 0, 0 );//applying Gaussian filter 
    
	namedWindow( outImage, CV_WINDOW_AUTOSIZE );
     //Create a window for display.

    imshow( pzOriginalImage, inputImage );
	imshow( outImage, outputImage );
}


