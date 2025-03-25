# import cv2
# import open3d


# image = cv2.imread("./GrayCodes/chess/00.jpg")
# image = cv2.resize(image, (0, 0), fx=0.1, fy=0.1)
# corners = cv2.findChessboardCorners(image, (7, 9))
# print(corners)
# cv2.imshow("test", image)
# cv2.imshow("test", image)
# cv2.waitKey(10000)

# imageLeft = cv2.imread("./GrayCodes/view0/02.jpg")
# imageLeft = cv2.resize(imageLeft, (0, 0), fx=0.1, fy=0.1)
# imageRight = cv2.imread("./GrayCodes/view1/02.jpg")
# imageRight = cv2.resize(imageRight, (0, 0), fx=0.1, fy=0.1)
# cv2.imshow("test", image)
# cv2.waitKey(10000)

from genericpath import isfile
from ntpath import join
from os import listdir
import numpy as np
import cv2 as cv
import glob

def intrinsic_calibration(file_paths: list, grid_size: (int, int)):
	# termination criteria
	criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	grid_x, grid_y = grid_size
	objp = np.zeros((grid_x * grid_y,3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_x,0:grid_y].T.reshape(-1,2)
	
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	for fp in file_paths:
		img = cv.imread(fp)
		img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv.findChessboardCorners(gray, grid_size, None)

		# If found, add object points, image points (after refining them)
		if ret :
			objpoints.append(objp)

			corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
			imgpoints.append(corners2)

	# Calibration
	_, Kmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

	return Kmtx, dist, rvecs, tvecs

def undistort_image(file_path: str):
	img = cv.imread(file_path)
	img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)

	# Undistortion
	h,  w = img.shape[:2]
	newcameramtx, roi = cv.getOptimalNewCameraMatrix(Kmtx, dist, (w,h), 1, (w,h))

	# undistort
	dst = cv.undistort(img, Kmtx, dist, None, newcameramtx)
	
	# crop the image
	x, y, w, h = roi
	dst = dst[y:y+h, x:x+w]

	cv.imshow('distorted', img)
	cv.imshow("undistorted", dst)
	cv.waitKey()
	cv.imwrite('calibresult.png', dst)

directory = "./GrayCodes/chess"
file_paths = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]
Kmtx, dist, rvecs, tvecs = intrinsic_calibration(file_paths, (7,9))
# for fp in file_paths:
# 	undistort_image(fp)
