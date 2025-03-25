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


def loadChessCorners(
    file: str,
    objpoints: list,
    imgpoints: list,
    grid_size: tuple[int, int],
    criteria: tuple[int, int, float],
):
    grid_x, grid_y = grid_size
    objp = np.zeros((grid_x * grid_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)

    img = cv.imread(file)
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, grid_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)


def intrinsic_calibration(file_names: list, grid_size: tuple[int, int]):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fn in file_names:
        loadChessCorners(fn, objpoints, imgpoints, grid_size, criteria)

    img = cv.imread(file_names[0])
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)

    # Calibration
    return cv.calibrateCamera(objpoints, imgpoints, img.shape[::2], None, None)


def undistort(file: str, Kmtx: np.array, dist: np.array):
    img = cv.imread(file)
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(Kmtx, dist, (w, h), 1, (w, h))
    return cv.undistort(gray, Kmtx, dist, None, newcameramtx)


if __name__ == "__main__":
    files = glob.glob("./GrayCodes/chess/*.jpg")
    ret, Kmtx, dist, rvecs, tvecs = intrinsic_calibration(files, (7, 9))
    np.savez("intrinsic_calibration.npz", Kmtx=Kmtx, dist=dist)
    np.savez("extrinsic_vectors.npz", rvecs=rvecs, tvecs=tvecs)

    cv.imshow("Undistorted", undistort(files[0], Kmtx, dist))
    cv.waitKey()
