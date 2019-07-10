cmake_minimum_required(VERSION 2.8)
project( DisplayImage )

find_package( OpenCV REQUIRED )
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable( DisplayImage DisplayImage.cpp )

target_link_libraries( DisplayImage ${OpenCV_LIBS} )
target_link_libraries (DisplayImage Eigen3::Eigen)
