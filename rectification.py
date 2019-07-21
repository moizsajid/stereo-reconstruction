import cv2
import numpy as np

def remove_black_region(img_color):
    
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape
    x1, x2, y1, y2 = (0,0,0,0)

    # Color threshold of the void part
    void_th = 5

    # List of colors in the vertical an horizontal lines that cross the image in the middle
    vertical = [img[i, int(w/2)] for i in range(h)]
    horizontal = [img[int(h/2), i] for i in range(w)]

    # Reverses both lists
    vertical_rev = vertical[::-1]
    horizontal_rev = horizontal[::-1]

    # Looks when the change of color is done
    for i in range(2,h):
        if vertical[i] > void_th and y1 == 0:
            y1 = i
        if vertical_rev[i] > void_th and y2 == 0:
            y2 = i
        if y1 != 0 and y2 != 0:
            break
    for i in range(2,w):
        if horizontal[i] > void_th and x1 == 0:
            x1 = i
        if horizontal_rev[i] > void_th and x2 == 0:
            x2 = i
        if x1 != 0 and x2 != 0:
            break

    desired_result = img_color[y1:h-y2, x1:w-x2]
    return desired_result

newcameramtx_left = np.array([[930.04052734, 0.0, 353.75734512], [0.0, 931.65527344, 633.84417384], [0.0, 0.0, 1.0]])
newcameramtx_right = np.array([[903.25213623, 0.0, 384.93938592], [0.0, 896.62097168, 639.9703307], [0.0, 0.0, 1.0]])

dist1 = np.array([[1.39169826e-01, -5.13185614e-01, -1.90417645e-03, 2.80508085e-04,  5.73013894e-01]])
dist2 = np.array([[0.08464559, -0.06763524,  0.00085903,  0.00293783, -0.14124984]])

cameraMatrix1 = newcameramtx_left
cameraMatrix2 = newcameramtx_right

distCoeffs1 = dist1
distCoeffs2 = dist2

img1_color = cv2.imread('./datasets/2_l.jpg') #queryimage # left image
img2_color = cv2.imread('./datasets/2_r.jpg') #trainimage # right image

img1_color = cv2.undistort(img1_color, cameraMatrix1, dist1, None, cameraMatrix1)
img2_color = cv2.undistort(img2_color, cameraMatrix2, dist2, None, cameraMatrix2)

img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

#img1 = cv2.imread('datasets/ours/left_1.jpg',0)
#img2 = cv2.imread('datasets/ours/right_1.jpg',0)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.4*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1_np = np.array(pts1)
pts2_np = np.array(pts2)

F, _ = cv2.findFundamentalMat(pts1_np, pts2_np, cv2.FM_RANSAC)

res, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_np, pts2_np, F, (720, 1280))

K1_inverse = np.linalg.inv(cameraMatrix1)
K2_inverse = np.linalg.inv(cameraMatrix2)

R1 = K1_inverse.dot(H1).dot(cameraMatrix1)
R2 = K2_inverse.dot(H2).dot(cameraMatrix2)

mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, cameraMatrix1, (720, 1280), cv2.CV_16SC2)
mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, cameraMatrix2, (720, 1280), cv2.CV_16SC2)

rectified1 = cv2.remap(img1_color, mapx1, mapy1, interpolation= cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
rectified2 = cv2.remap(img2_color, mapx2, mapy2, interpolation= cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

#rectified1 = remove_black_region(rectified1)
#rectified2 = remove_black_region(rectified2)

cv2.imwrite('rectified1.jpg', rectified1)
cv2.imwrite('rectified2.jpg', rectified2)