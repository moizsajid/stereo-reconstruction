// ./DisplayImage -input1=./datasets/Motorcycle-perfect/im0.png -input2=./datasets/Motorcycle-perfect/im1.png
// ./DisplayImage -input1=./data/tsukuba_l.png -input2=./data/tsukuba_r.png

#include <iostream>
#include <fstream>
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
    CommandLineParser parser( argc, argv, keys );

    Mat img1 = imread( parser.get<String>("input1"), IMREAD_COLOR);
    Mat img2 = imread( parser.get<String>("input2"), IMREAD_COLOR);

    if ( img1.empty() || img2.empty() )
    {
        cout << "Could not open or find the image!\n" << endl;
        parser.printMessage();
        return -1;
    }

    //resize(img1, img1, cv::Size(), 0.2, 0.2);
    //resize(img2, img2, cv::Size(), 0.2, 0.2);
    
    //normalize(img1, img1, 0, 255, NORM_MINMAX, CV_8UC1);
    //normalize(img2, img2, 0, 255, NORM_MINMAX, CV_8UC1);

    cout << img1.channels() << endl;
    cout << img2.channels() << endl;

    cvtColor(img1, img1, COLOR_BGR2GRAY);
    cvtColor(img2, img2, COLOR_BGR2GRAY);

    imshow("Img 1", img1);
    imshow("Img 2", img2);

    Mat disp;

    //Ptr<StereoBM> stereo = StereoBM::create(16, 15);
    Ptr<StereoBM> stereo = StereoBM::create(64, 21);

    stereo->compute(img1, img2, disp);

    //normalize(disp, disp, 0, 255, NORM_MINMAX, CV_8UC1);

    // double min, max;
    // minMaxLoc(img1, &min, &max);

    // double min2, max2;
    // minMaxLoc(img2, &min2, &max2);

    double min3, max3;
    minMaxLoc(disp, &min3, &max3);

    // cout << min << " " << max << endl;
    // cout << min2 << " " << max2 << endl;
    // cout << min3 << " " << max3 << endl;

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

    double focal_length = 3979.911;
    double baseline = 193.001;

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

    // Eigen::MatrixXd intrinsics2(3, 3);

    // intrinsics2 << 3979.911, 0.0, 1369.115,
    //               0.0, 3979.911, 1019.507,
    //               0.0, 0.0, 1.0;

    //cout << disp.at<double>(0, 0) << endl;

    for(int i = 0; i < disp.rows; i++) {

        std::vector<double> v;
        disp.row(i).copyTo(v);

        for(int j = 0; j < disp.cols; j++) {
            if(v.at(j) != 0) {
                double z = focal_length * baseline / v.at(j);
                double x = j * z / focal_length;
                double y = i * z / focal_length;
                points.push_back(Point3d(x, y, z));
                cout << z << " ";
            }
        }
        cout << endl;
    }

    // double min4, max4;
    // minMaxLoc(disp, &min4, &max4);

    // cout << min << " " << max << endl;

    //cout << v.at(50) << endl;
    //cout << disp.at<double>(10, 50) << endl;
    
    //cout << img1.rows << " " << img1.cols << endl;
    //cout << disp.rows << " " << disp.cols << endl;
    

    int rows = points.size();

    std::ofstream file; 
    file.open ("shape.off");
    
    file << "OFF\n";
    file << points.size() << " 0 0\n";

    // //cout << intrinsics << endl;

    

    for(int i = 0; i < rows; i++) {
        //pointsMatrix.ropointw(i) = Eigen::Vector3d(points.at(i).x, points.at(i).y, points.at(i).z);
        Eigen::Vector3d point = intrinsics.inverse() * Eigen::Vector3d(points.at(i).x, points.at(i).y, points.at(i).z);
        file << point(0) << " " << point(1) << " " << point(2) << "\n";
        //file << point << "\n";
    }
    
    //file << pointsMatrix * intrinsics.inverse() << endl;

    // for(int i = 0; i < points.size(); i++) {
    //     file << points.at(i).x << " " << points.at(i).y << " " << points.at(i).z << "\n";
    // } 
    
    file.close();

    // cout << points.size() << endl;
    // cout << points.at(0) << endl;

    //-- Show detected matches
    //imshow("Good Matches", img_matches );
    //imshow("Matches", img_matches);
    imshow("Stereo", disp);
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