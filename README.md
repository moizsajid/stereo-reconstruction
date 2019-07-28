# Stereo Reconstruction

The images used are inside the **images** folder. The corresponding disparity maps are in **disparity** folder and the corresponding 3D reconstructions are inside the **reconstructions** folder. Camera Calibration images and code are stored inside the **camera-calibration** folder. The results for the camera calibration are documented in *calibration.ipynb* notebook inside the **camera-calibration** folder.

The **recitification.py** files performs rectification assuming that the camera calibration paramters are already known. The camera intrinsics were found using the OpenCV. Similarly, for performing stereo rectification, OpenCV was used. The **DisplayImage.cpp** file uses SAD block matching for calculating disparity maps and back projection for projecting points to 3D. These algorithms were implemented by ourselves without using any library.

For running the **rectification.py**, please run the following command:
    
    rectification.py

For running the **DisplayImage.cpp**, please run the following command:
   
    make
    DisplayImage -input1=./images/youssef_l.jpg -input2=./images/youssef_r.jpg
    
