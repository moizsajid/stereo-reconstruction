// ./DisplayImage -input1=./datasets/Motorcycle-perfect/im0.png -input2=./datasets/Motorcycle-perfect/im1.png
// ./DisplayImage -input1=./data/tsukuba_l.png -input2=./data/tsukuba_r.png

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
    // resize(img1_color, img1_color, cv::Size(), 0.2, 0.2);
    // resize(img2_color, img2_color, cv::Size(), 0.2, 0.2);

    Mat img1, img2, disp;

    cvtColor(img1_color, img1, COLOR_BGR2GRAY);
    cvtColor(img2_color, img2, COLOR_BGR2GRAY);

    cv::Mat left_image = img1;
    cv::Mat right_image = img2;
	int h = left_image.rows, w = left_image.cols, nchannels = left_image.channels();


    double min3, max3;

 //    // // Ptr<StereoBM> stereo = StereoBM::create(16, 15);
 //    // Ptr<StereoBM> stereo = StereoBM::create(256, 21);
 //    Ptr<StereoBM> stereo = StereoBM::create(48, 21);
 //    // Ptr<StereoBM> stereo = StereoBM::create(64, 21);
 //    // Ptr<StereoBM> stereo = StereoBM::create(128, 21);

 //    stereo->compute(img1, img2, disp);
 //    minMaxLoc(disp, &min3, &max3);
 //    cout << min3 << " " << max3 << endl;

	// std::string filename = "dispmap_original_opencv.xml";
	// // std::string sad_filename = "dispmapopencv_255.xml";
	// cv::FileStorage disp_file(filename, cv::FileStorage::WRITE);
	// // cv::FileStorage sad_file(sad_filename, cv::FileStorage::WRITE);

	// disp_file << "disp" << disp; // Write entire cv::Mat

 // //    //normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8UC1);
 //    disp.convertTo(disp, CV_32F, 1.0 / 16.0);


	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	//////////////////////////////////////////////////

	// cv::Mat disp_img = cv::Mat::zeros(left_image.rows, left_image.cols, CV_8UC1);
	// cv::Mat disp_img_255 = cv::Mat::zeros(left_image.rows, left_image.cols, CV_8UC1);

	cv::Mat disp_img = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);
	cv::Mat disp_img_255 = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);//CV_32SC1

	// cv::Mat disp_img = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32F);
	// cv::Mat disp_img_255 = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32F);
	// cv::Mat disp_img_255 = cv::Mat::zeros(left_image.rows, left_image.cols, CV_16UC1);
	
	cv::Mat disp_img_grouped = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);
	// set range to search from current pixel
	// int disparity_to_left = -64, disparity_to_right = 0;
	// int disparity_to_left = -256, disparity_to_right = 0;
	int disparity_to_left = -48, disparity_to_right = 0;
	int disparity_range = abs(disparity_to_right - disparity_to_left);
	// int half_block_size = 21;
	int half_block_size = 21;

	int row_right, col_right;
	int row_start, row_end;
	int left_half_range, right_half_range;
	int col_start_right, col_end_right, col_start_left, col_end_left;

	cv::Mat left_block, right_block, block_diff;
	int total_sad_diff, lowest_sad_diff, best_disparity;
	for (int row_i = 0; row_i < h; ++row_i)
	{
		for (int col_i = 0; col_i < w; ++col_i)
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
			// disp_img_255.at<uchar>(row_i, col_i) = std::abs(255 - (int)255.0*(best_disparity-abs(disparity_to_left))/disparity_range);
			// disp_img.at<uchar>(row_i, col_i) = std::abs(255 - (63+(int)192.0*(best_disparity-abs(disparity_to_left))/disparity_range));
		
			disp_img_255.at<int>(row_i, col_i) = std::abs(255 - (int)255.0*(best_disparity-disparity_to_left)/disparity_range);
			// // disp_img.at<uchar>(row_i, col_i) = std::abs((63+(int)192.0*(best_disparity-abs(disparity_to_left))/disparity_range));
			disp_img.at<int>(row_i, col_i) = best_disparity;

			// disp_img_255.at<float>(row_i, col_i) = std::abs((int)255.0*(best_disparity-disparity_to_left)/disparity_range);
			// disp_img.at<float>(row_i, col_i) = std::abs((63+(int)192.0*(best_disparity-abs(disparity_to_left))/disparity_range));
			// disp_img.at<float>(row_i, col_i) = best_disparity;

			// std::cout << best_disparity << " ";
		}
		// std::cout << std::endl;
	}
	int group_range = 85;
	int original_value = 0;
	for (int row_i = 0; row_i < h; ++row_i)
	{
		for (int col_i = 0; col_i < w; ++col_i)
		{
			original_value = disp_img_255.at<int>(row_i, col_i);
			disp_img_grouped.at<int>(row_i, col_i) = (original_value / group_range) * group_range;
		}
	}

	cv::Mat disp_img_median = cv::Mat::zeros(left_image.rows, left_image.cols, CV_32S);
	medianBlur(disp_img_255, disp_img_median, 3);
	// medianBlur(InputArray src, OutputArray dst, int ksize)

	// int filter_size = 5;
	// for (int row_i = 0; row_i < h; ++row_i)
	// {
	// 	for (int col_i = 0; col_i < w; ++col_i)
	// 	{
	// 		row_right = row_i;
	// 		col_right = col_i + col_diff; 

	// 		if (col_right < 0 || col_right >= w)
	// 			continue;

	// 		row_start = max(row_right - filter_size, 0);
	// 		row_end = min(row_right + filter_size + 1, h - 1);

	// 		col_start_left = max(col_i - filter_size, 0);
	// 		col_end_left = min(col_i + filter_size + 1, w - 1);

	// 		col_start_right = max(col_right - filter_size, 0);
	// 		col_end_right = min(col_right + filter_size + 1, w - 1);

	// 		left_half_range = std::min(col_i - col_start_left, col_right - col_start_right);
	// 		right_half_range = std::min(col_end_left - col_i - 1, col_end_right - col_right - 1);

	// 		col_start_left = max(col_i - left_half_range, 0);
	// 		col_end_left = min(col_i + right_half_range + 1, w - 1);

	// 		col_start_right = max(col_right - left_half_range, 0);
	// 		col_end_right = min(col_right + right_half_range + 1, w - 1);
			
	// 		left_block = getBlock(disp_img_255, row_start, row_end, col_start_left, col_end_left); 
	// 		// right_block = getBlock(right_image, row_start, row_end, col_start_right, col_end_right); 

	// 		original_value = disp_img_255.at<int>(row_i, col_i);
	// 		disp_img_grouped.at<int>(row_i, col_i) = (original_value / group_range) * group_range;
	// 	}
	// }

	cv::FileStorage disp_mine_file("dispmap_mine.xml", cv::FileStorage::WRITE);
	disp_mine_file << "disp" << disp_img; // Write entire cv::Mat
	cv::FileStorage disp_mine_file_255("dispmap_mine_255.xml", cv::FileStorage::WRITE);
	disp_mine_file_255 << "disp" << disp_img_255; // Write entire cv::Mat
	cv::FileStorage disp_mine_file_grouped("dispmap_mine_grouped.xml", cv::FileStorage::WRITE);
	disp_mine_file_grouped << "disp" << disp_img_grouped; // Write entire cv::Mat
	cv::FileStorage disp_mine_file_median("dispmap_mine_median.xml", cv::FileStorage::WRITE);
	disp_mine_file_median << "disp" << disp_img_median; // Write entire cv::Mat

	// disp = disp_img_255;
	disp = disp_img_grouped;
	// disp = disp_img;

    minMaxLoc(disp, &min3, &max3);
    cout << min3 << " " << max3 << endl;

	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	//////////////////////////////////////////////////
	//////////////////////////////////////////////////

    bool our_camera = true;

	cv::FileStorage disp_conv_file("dismap_conv.xml", cv::FileStorage::WRITE);
	disp_conv_file << "disp_conv" << disp; // Write entire cv::Mat

    std::vector<Point3d> points;
    std::vector<Point3d> colors;
    std::vector<Point3d> faces;

    double focal_length = 3979.911;
    double baseline = 193.001;

    Eigen::MatrixXd intrinsics(3, 3);

    if (our_camera){
    	baseline = 5.7; //web cam
    	focal_length = 413.69216919;  // web cam
    	// focal_length = 489.6895905;  // web cam
    	// focal_length = 594.95166016;  // web cam
    	// // web cam intrinsics
	    intrinsics << 413.69216919, 0.0, 369.60504179,
	                  0.0, 482.78549194, 242.59775176,
	                  0.0, 0.0, 1.0;
    // intrinsics << 594.95166016, 0.0, 319.80993603,
    //               0.0, 594.95166016, 234.68265936,
    //               0.0, 0.0, 1.0;
    }
    else {
	    intrinsics << 3979.911, 0.0, 1244.772,
	                  0.0, 3979.911, 1019.507,
	                  0.0, 0.0, 1.0;
    }

    Eigen::MatrixXd intrinsics2(3, 3);

    // // web cam intrinsics
    if (our_camera)
   	{
	    intrinsics2 << 565.68701172, 0.0, 319.80993603,
	                  0.0, 594.95166016, 234.68265936,
	                  0.0, 0.0, 1.0;
    // intrinsics2 << 594.95166016, 0.0, 319.80993603,
    //               0.0, 594.95166016, 234.68265936,
    //               0.0, 0.0, 1.0;
   	} 
   	else 
   	{
	    intrinsics2 << 3979.911, 0.0, 1369.115,
	                  0.0, 3979.911, 1019.507,
	                  0.0, 0.0, 1.0;
   	}

    double edgeThreshold = 10;
    // double edgeThreshold = 200;

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
            //cout << i << " " << j << " "  << endl;

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

    //-- Show detected matches 
    //imshow("Good Matches", img_matches );
    //imshow("Matches", img_matches);
    imwrite("stereo.jpg", disp);
    // imshow("Point", img1);
    // imshow("Same point", img2);
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