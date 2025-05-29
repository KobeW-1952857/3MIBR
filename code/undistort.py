import glob
import os
import numpy as np
import cv2 as cv
import cv2.typing as cvt
from utils.getFilenames import getView0, getView1


def undistortImages(
    files: list, resolution: tuple, save_path: str, intrinsic, distortion
):
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(intrinsic, distortion, resolution, 1)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for file in files:
        save_file = os.path.join(save_path, os.path.basename(file))
        print("\rUndistorting " + os.path.basename(save_file), end="")
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = cv.undistort(img, intrinsic, distortion, None, newcameramtx)
        cv.imwrite(save_file, img)
    print()


def undistortAllViews(base_path: str, save_path: str, intrinsic, distortion):
    for view in os.listdir(base_path):
        print("Undistorting " + view)
        view_path = os.path.join(base_path, view)
        files = glob.glob(view_path + "/*.jpg")
        resolution = cv.imread(files[0], cv.IMREAD_GRAYSCALE).shape
        save_path = os.path.join(save_path, view)
        undistortImages(files, resolution, save_path, intrinsic, distortion)
