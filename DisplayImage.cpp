// ./DisplayImage -input1=./datasets/Motorcycle-perfect/im0.png -input2=./datasets/Motorcycle-perfect/im1.png
// ./DisplayImage -input1=./data/tsukuba_l.png -input2=./data/tsukuba_r.png
// ./DisplayImage -input1=./datasets/ours/left_1.jpg -input2=./datasets/ours/right_1.jpg
 // ./DisplayImage -input1=./camera-calibration/rectified1.jpg -input2=./camera-calibration/rectified2.jpg

#include <iostream>
#include <fstream>
#include <iomanip>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/calib3d.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/stereo.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <Eigen/Dense>

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

const char* keys =
    "{ help h |                          | Print help message. }"
    "{ input1 | ../data/box.png          | Path to input image 1. }"
    "{ input2 | ../data/box_in_scene.png | Path to input image 2. }";


inline int square(int x){ return x*x; }

inline cv::Mat getBlock(cv::Mat& matrix, const int row_start, const int row_end, const int col_start, const int col_end) 
{
	cv::Mat block = matrix.rowRange(row_start, row_end).colRange(col_start, col_end); 
	return block;
}

int main( int argc, char* argv[] )
{
    printf("%u.%u.%u\r\n", CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION);

    CommandLineParser parser( argc, argv, keys );

    Mat img1_color = imread( parser.get<String>("input1"), IMREAD_COLOR);
    Mat img2_color = imread( parser.get<String>("input2"), IMREAD_COLOR);

    if ( img1_color.empty() || img2_color.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }
    // if (img1_color.rows > 1500)
    // if (img1_color.rows > 700)
    // {
    // 	float ratio = 0.5;
	   //  resize(img1_color, img1_color, cv::Size(), ratio, ratio);
	   //  resize(img2_color, img2_color, cv::Size(), ratio, ratio);
    //     std::cout << "resized " << ratio << endl;
    // }

    Mat img1, img2, disp;

    // img1 = img1_color;
    // img2 = img2_color;
    cvtColor(img1_color, img1, COLOR_BGR2GRAY);
    cvtColor(img2_color, img2, COLOR_BGR2GRAY);

    cv::Mat left_image = img1;
    cv::Mat right_image = img2;
	int h = left_image.rows, w = left_image.cols, nchannels = left_image.channels();
	cout << "channels " << nchannels << endl;


    double min3, max3;

	cv::Mat disp_img_255 = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);//CV_32SC1
	
	cv::Mat disp_img_grouped = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);

	// set range to search from current pixel
	// int disparity_to_left = -90, disparity_to_right = 40;  // erlier -256, -48
    int disparity_to_left = -170, disparity_to_right = 85;  // erlier -256, -48
	int disparity_range = abs(disparity_to_right - disparity_to_left);
	int half_block_size = 7; // earlier 21, 5, 11

	int row_right, col_right;
	int row_start, row_end;
	int left_half_range, right_half_range;
	int col_start_right, col_end_right, col_start_left, col_end_left;

	cv::Mat left_block, right_block, block_diff;
	int total_sad_diff, lowest_sad_diff = 1e6, best_disparity = 0;
	bool early_found = false;
	for (int row_i = 0; row_i < h; ++row_i)
	{
		for (int col_i = 0; col_i < w; ++col_i)
		{
			if (!early_found)
			{
				// traverses each pixel in the left image
				lowest_sad_diff = 50000 * square(2 * half_block_size + 1);
				best_disparity = 0;
				// define which elements of the right matrix are going to be in the comparison block
				for (int col_diff = disparity_to_left; col_diff <= disparity_to_right; ++col_diff)
				{
					row_right = row_i;
					col_right = col_i + col_diff; 

					if (col_right < 0 || col_right >= w)
						continue;

					row_start = max(row_right - half_block_size, 0);
					row_end = min(row_right + half_block_size + 1, h - 1);

					col_start_left = max(col_i - half_block_size, 0);
					col_end_left = min(col_i + half_block_size + 1, w - 1);

					col_start_right = max(col_right - half_block_size, 0);
					col_end_right = min(col_right + half_block_size + 1, w - 1);

					left_half_range = std::min(col_i - col_start_left, col_right - col_start_right);
					right_half_range = std::min(col_end_left - col_i - 1, col_end_right - col_right - 1);

					col_start_left = max(col_i - left_half_range, 0);
					col_end_left = min(col_i + right_half_range + 1, w - 1);

					col_start_right = max(col_right - left_half_range, 0);
					col_end_right = min(col_right + right_half_range + 1, w - 1);
					
					left_block = getBlock(left_image, row_start, row_end, col_start_left, col_end_left); 
					right_block = getBlock(right_image, row_start, row_end, col_start_right, col_end_right); 

					cv::absdiff(left_block, right_block, block_diff);
					cv::Scalar sad_diff_array = cv::sum(block_diff);
					total_sad_diff = sad_diff_array[0] + sad_diff_array[1] + sad_diff_array[2]; 
					if (total_sad_diff < lowest_sad_diff)
					{
						lowest_sad_diff = total_sad_diff;
						best_disparity = col_diff;
					}
				}
			}
			disp_img_255.at<int>(row_i, col_i) = std::abs(255.0 - (int)255.0*(best_disparity-disparity_to_left)/disparity_range);
		}
	}

	int group_range = 2;  // modify this parameter to control how many groups at the end
	group_range = 255 / group_range + 1; 

	int original_value = 0;
	for (int row_i = 0; row_i < h; ++row_i)
	{
		for (int col_i = 0; col_i < w; ++col_i)
		{
			original_value = disp_img_255.at<int>(row_i, col_i);
			disp_img_grouped.at<int>(row_i, col_i) = ((original_value / group_range) * group_range + 1);
		}
	}

	cv::FileStorage disp_file_255("dispmap_255.xml", cv::FileStorage::WRITE);
	disp_file_255 << "disp" << disp_img_255; // Write entire cv::Mat
	cv::FileStorage disp_file_grouped("dispmap_grouped.xml", cv::FileStorage::WRITE);
	disp_file_grouped << "disp" << disp_img_grouped; // Write entire cv::Mat

	// disp = disp_img_grouped;
	disp = disp_img_255;

    minMaxLoc(disp, &min3, &max3);
    cout << min3 << " " << max3 << endl;

    bool our_camera = true; // using our camera or not? Calibration params need to be changed

	cv::FileStorage disp_conv_file("dismap_conv.xml", cv::FileStorage::WRITE);
	disp_conv_file << "disp_conv" << disp; // Write entire cv::Mat

    std::vector<Point3d> points;
    std::vector<Point3d> colors;
    std::vector<Point3d> faces;

    double focal_length;
    double baseline;

    Eigen::MatrixXd intrinsics(3, 3);

    if (our_camera)
    {
    	// baseline = 5.7; //web cam
    	// focal_length = 413.69216919;  // web cam
    	// // focal_length = 489.6895905;  // web cam
    	// // focal_length = 594.95166016;  // web cam
    	// // // web cam intrinsics
	    // intrinsics << 413.69216919, 0.0, 369.60504179,
	    //               0.0, 482.78549194, 242.59775176,
	    //               0.0, 0.0, 1.0;

        baseline = 100; //web cam
    	focal_length = 1740.01245;  // web cam


	    intrinsics <<   1740.01245, 0.0, 779.601468,
                        0.0, 1742.12231, 1138.81797,
                        0.0, 0.0, 1.0;


    }
    else   // if database images
    {

        focal_length = 3979.911;
        baseline = 193.001;

	    intrinsics << 3979.911, 0.0, 1244.772,
	                  0.0, 3979.911, 1019.507,
	                  0.0, 0.0, 1.0;
    }

    Eigen::MatrixXd intrinsics2(3, 3);

    // // web cam intrinsics
    if (our_camera)
   	{
	    // intrinsics2 << 565.68701172, 0.0, 319.80993603,
	    //               0.0, 594.95166016, 234.68265936,
	    //               0.0, 0.0, 1.0;



        intrinsics2 <<      1612.98267, 0.0, 778.885092,
                            0.0, 1642.61304, 1083.29674,
                            0.0, 0.0, 1.0;
   	} 
   	else  // if database images
   	{
	    intrinsics2 << 3979.911, 0.0, 1369.115,
	                  0.0, 3979.911, 1019.507,
	                  0.0, 0.0, 1.0;
   	}

    double edgeThreshold = 10;

    double Q03 = -intrinsics(0, 2);
    double Q13 = -intrinsics(1, 2);
    double Q23 = focal_length;
    double Q32 = -1.0 / baseline;
    double Q33 = (intrinsics(0, 2) - intrinsics2(0, 2)) / baseline;

    
    cout << img1_color.rows << " " << img1_color.cols << " " << img1_color.channels() << endl;
    cout << disp.rows << " " << disp.cols << endl;

    for(int i = 0; i < disp.rows; i++) {

        std::vector<double> v;
        disp.row(i).copyTo(v);

        for(int j = 0; j < disp.cols; j++) {
            if(v.at(j) > 0) {
                
                double pw = 1.0 / (v.at(j) * Q32 + Q33);
                double z = Q23 * pw;
                double x = ((float)j + Q03) * pw;
                double y = ((float)i + Q13) * pw;

                int color_b = img1_color.at<Vec3b>(i, j)[0];
                int color_g = img1_color.at<Vec3b>(i, j)[1];
                int color_r = img1_color.at<Vec3b>(i, j)[2];

                points.push_back(Point3d(x, y, z));
                colors.push_back(Point3d(color_r, color_g, color_b));
            }
        }
    }

    for(int idx = 0; idx < points.size()-2; idx++) {
        Point3d pt1 = points.at(idx);
        Point3d pt2 = points.at(idx+1);
        Point3d pt3 = points.at(idx+2);

        double len1 = norm(pt1-pt2);
        double len2 = norm(pt1-pt3);
        double len3 = norm(pt2-pt3);

        if(len1 + len2 + len3 < edgeThreshold) {
            faces.push_back(Point3d(idx, idx+1, idx+2));
        }
    }
    
    std::ofstream file; 
    file.open ("shape.off");
    
    file << std::fixed;

    file << "COFF\n";
    file << points.size() << " 0 0\n";

    for(int i = 0; i < points.size(); i++) {
        file << std::setprecision(2) << points.at(i).x << " " << points.at(i).y << " " << points.at(i).z << " " << (int)colors.at(i).x << " " << (int)colors.at(i).y << " " << (int)colors.at(i).z << " 255\n";
    } 
    
    
    file.close();

    imwrite("disp.jpg", disp);
    imwrite("disp_255_original.jpg", disp_img_255);
    imwrite("disp_grouped.jpg", disp_img_grouped);
    waitKey();
    return 0;


}

#else
int main()
{
    std::cout << "This tutorial code needs the xfeatures2d contrib module to be run." << std::endl;
    return 0;
}
#endif