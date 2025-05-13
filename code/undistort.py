import glob
import os
import numpy as np
import cv2 as cv

from calibration import undistortCustom
from getFilenames import getView0, getView1

def undistortImages(files: list, resolution: tuple, relativeDirPath: str):
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(Kmtx, dist, resolution, 1, resolution)

    gray_codes = [undistortCustom(file, Kmtx, dist, newcameramtx, resolution) for file in files]

    count = 0
    for im in gray_codes:
        np.save(relativeDirPath + str(count) +".npy", im)
        count += 1

if __name__ == "__main__":
    view0_files = getView0()
    undistortImages(view0_files, (4752, 3168), "../dataset/GrayCodes_HighRes/undistorted/view0/")
    view1_files = getView1()
    undistortImages(view1_files, (4752, 3168), "../dataset/GrayCodes_HighRes/undistorted/view1/")
