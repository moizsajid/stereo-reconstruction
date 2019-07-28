# Stereo Reconstruction

The images used are inside the images folder. The 3D reconstructions are inside the reconstructions folder. Camera Calibration images and code is stored inside the camera-calibration folder.

The recitification.py files performs rectification assuming that the camera calibration paramters are already known. The DisplayImage.cpp file uses SAD block matching for calculating disparity and back projection for projecting points to 3D. These algorithms were implemented by ourselves without using any library.