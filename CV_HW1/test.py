import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')


# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    # Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        img = cv2.resize(img, (960, 540))

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])

'''
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
'''

# Write your code here

# step1. Use the points in each images to find Hi 
# reference from http://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
# https://tigercosmos.xyz/post/2020/04/cv/camera-calibration/
# https://medium.com/%E6%95%B8%E5%AD%B8%E5%B7%A5%E5%BB%A0/camera-calibration%E7%9B%B8%E6%A9%9F%E6%A0%A1%E6%AD%A3-1d94ffa7cbb4


# Make a list of calibration images
images = glob.glob('data/*.jpg')
# total 7*7 point pairs(corner_x*corner_y)
point_pairs = corner_x*corner_y

def get_homo(images, point_pairs, objpoints, imgpoints):
    '''
    Argument:
    - images      : list of images, list
    - point_pairs : the correspondent point pairs numbers, int
    - objpoints   : 3d points in real world space, list
    - imgpoints   : 2d points in image plane, list
    -------------------------------------------------
    Return:
    - Homo_list: list of given homography matrix Hi
    '''
    # list of homography matrix of 10 imgs
    Homo_list = []
    print(f"total {len(images)} images!")
    for img_i in range(len(images)):        
        A = np.zeros((2*point_pairs, 9), dtype=np.float32)
        for point_pair in range(point_pairs):
            X1 = objpoints[img_i][point_pair][0]
            Y1 = objpoints[img_i][point_pair][1]
            u  = imgpoints[img_i][point_pair][0][0]
            v  = imgpoints[img_i][point_pair][0][1]
            A[2*point_pair  , :] = [X1, Y1, 1, 0, 0, 0, -u*X1, -u*Y1, -u]
            A[2*point_pair+1, :] = [0, 0, 0, X1, Y1, 1, -v*X1, -v*Y1, -v]

        U, D, V = np.linalg.svd(A)
        Homo = V[-1]
        Homo_list.append(Homo.reshape((3, 3)))

    return Homo_list

# create the homo_list
Homo_list = get_homo(images, point_pairs, objpoints, imgpoints)
print(Homo_list[0])
print(type(Homo_list[0]))



'''
# test cv2.findHomography()
list_3D = []
list_2D = []
for point_pair in range(49):
    X1 = objpoints[0][point_pair][0]
    Y1 = objpoints[0][point_pair][1]
    list_3D.append((X1, Y1))
    u  = imgpoints[0][point_pair][0][0]
    v  = imgpoints[0][point_pair][0][1]
    list_2D.append((u, v))
list_3D = np.asarray(list_3D, dtype=np.float32)
list_2D = np.asarray(list_2D, dtype=np.float32)
M, mask = cv2.findHomography(list_3D, list_2D, cv2.RANSAC,5.0)
print(M)
'''