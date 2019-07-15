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

    //resize(img1, img1, cv::Size(), 0.2, 0.2);
    //resize(img2, img2, cv::Size(), 0.2, 0.2);
    
    //normalize(img1, img1, 0, 255, NORM_MINMAX, CV_8UC1);
    //normalize(img2, img2, 0, 255, NORM_MINMAX, CV_8UC1);

    Mat img1, img2, disp;

    cvtColor(img1_color, img1, COLOR_BGR2GRAY);
    cvtColor(img2_color, img2, COLOR_BGR2GRAY);

    //Ptr<StereoBM> stereo = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo = StereoBM::create(256, 21);

    stereo->compute(img1, img2, disp);

    //normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8UC1);
    disp.convertTo(disp, CV_32F, 1.0 / 16.0);

    // double min, max;
    // minMaxLoc(img1, &min, &max);

    // double min2, max2;
    // minMaxLoc(img2, &min2, &max2);

    double min3, max3;
    minMaxLoc(disp, &min3, &max3);

    // cout << min << " " << max << endl;
    // cout << min2 << " " << max2 << endl;
    cout << min3 << " " << max3 << endl;

    //cout << disp.rows << " " << disp.cols << endl;

    //cout << disp << endl;

    // //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    // int minHessian = 400;
    // Ptr<SURF> detector = SURF::create( minHessian );
    // std::vector<KeyPoint> keypoints1, keypoints2;
    // Mat descriptors1, descriptors2;
    // detector->detectAndCompute( img1, noArray(), keypoints1, descriptors1 );
    // detector->detectAndCompute( img2, noArray(), keypoints2, descriptors2 );
    
    // //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // // Since SURF is a floating-point descriptor NORM_L2 is used
    // Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    // std::vector< std::vector<DMatch> > knn_matches;
    // matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );
    
    // //-- Filter matches using the Lowe's ratio test
    // const float ratio_thresh = 0.4f;
    // std::vector<DMatch> good_matches;
    // for (size_t i = 0; i < knn_matches.size(); i++)
    // {
    //     if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
    //     {
    //         good_matches.push_back(knn_matches[i][0]);
    //     }
    // }
    
    // //-- Draw matches
    // Mat img_matches;
    // drawMatches( img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1),
    //              Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    // long num_matches = good_matches.size();
    // std::vector<Point2f> matched_points1;
    // std::vector<Point2f> matched_points2;

    // for (int i=0;i<num_matches;i++)
    // {
    //     int idx1=good_matches[i].trainIdx;
    //     int idx2=good_matches[i].queryIdx;
    //     matched_points1.push_back(keypoints1[idx1].pt);
    //     matched_points2.push_back(keypoints2[idx2].pt);
    // }

    // //cout << matched_points1.at(0) << endl;

    // //circle(img1, matched_points1.at(0), 10, (0, 255, 0), 2);
    // //circle(img2, matched_points2.at(0), 10, (0, 255, 0), 2);

    std::vector<Point3d> points;
    std::vector<Point3d> colors;
    std::vector<Point3d> faces;

    double focal_length = 3979.911;
    double baseline = 193.001;
    // double focal_length = 3979.911;
    // double baseline = 193.001;

    // for(int i = 0; i < matched_points1.size(); i++) {
    //     double diff = matched_points1.at(i).x - matched_points2.at(i).x;
    //     //cout << diff << endl;
    //     double z = focal_length * baseline / diff;
    //     double x = matched_points1.at(i).x * z / focal_length;
    //     double y = matched_points1.at(i).y * z / focal_length;
    //     points.push_back(Point3d(x, y, z));
    // }

    Eigen::MatrixXd intrinsics(3, 3);

    intrinsics << 3979.911, 0.0, 1244.772,
                  0.0, 3979.911, 1019.507,
                  0.0, 0.0, 1.0;

    Eigen::MatrixXd intrinsics2(3, 3);

    intrinsics2 << 3979.911, 0.0, 1369.115,
                  0.0, 3979.911, 1019.507,
                  0.0, 0.0, 1.0;

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
            
            //cout << i << " " << j << " "  << endl;

            if(v.at(j) > 0) {
                
                double pw = 1.0 / (v.at(j) * Q32 + Q33);
                double z = Q23 * pw;
                double x = ((float)j + Q03) * pw;
                double y = ((float)i + Q13) * pw;
                
                //Vec3b intensity = img1_color.at<Vec3b>(j, i);
                
                

                //for(int k = 0; k < image.channels(); k++) {
                //    uchar col = intensity.val[k]; 
                //}

                //int color_b = 0;
                //int color_g = 0;
                //int color_r = 0;

                //int color_b = intensity.val[0];
                //int color_g = intensity.val[1];
                //int color_r = intensity.val[2];

                int color_b = img1_color.at<Vec3b>(i, j)[0];
                int color_g = img1_color.at<Vec3b>(i, j)[1];
                int color_r = img1_color.at<Vec3b>(i, j)[2];

                //double z = focal_length * baseline / (v.at(j));
                //double x = (float) j * z / focal_length;
                //double y = (float) i * z / focal_length;
                points.push_back(Point3d(x, y, z));
                colors.push_back(Point3d(color_r, color_g, color_b));
                //cout << z << " ";
            }
        }

        //cout << endl;
    }

    //cout << "Here!" << endl;

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
        //pointsMatrix.ropointw(i) = Eigen::Vector3d(points.at(i).x, points.at(i).y, points.at(i).z);
        //Eigen::Vector3d point = intrinsics.inverse() * Eigen::Vector3d(points.at(i).x, points.at(i).y, points.at(i).z);
        //file << points.at(i).x << " " << points.at(i).y << " " << points.at(i).z << "\n";
        file << std::setprecision(2) << points.at(i).x << " " << points.at(i).y << " " << points.at(i).z << " " << (int)colors.at(i).x << " " << (int)colors.at(i).y << " " << (int)colors.at(i).z << " 255\n";
        //file << point << "\n";
    } 
    
    // for(int i = 0; i < faces.size(); i++) {
    //     file << (int)faces.at(i).x << " " << (int)faces.at(i).y << " " << (int)faces.at(i).z << "\n";
    // } 
    
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