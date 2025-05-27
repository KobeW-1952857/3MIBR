import glob
import os
import numpy as np
import cv2 as cv
import cv2.typing as cvt
from getFilenames import getView0, getView1


def undistort(
    file: str,
    Kmtx: cvt.MatLike,
    dist: cvt.MatLike,
    newcameramtx: cvt.MatLike,
    shape: tuple,
) -> cvt.MatLike:
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, shape)

    return cv.undistort(gray, Kmtx, dist, None, newcameramtx)


def undistortImages(files: list, resolution: tuple, relativeDirPath: str):
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(
        Kmtx, dist, resolution, 1, resolution
    )

    gray_codes = [
        undistort(file, Kmtx, dist, newcameramtx, resolution) for file in files
    ]

    for i, im in enumerate(gray_codes):
        np.save(relativeDirPath + str(i) + ".npy", im)


if __name__ == "__main__":
    view0_files = getView0()
    undistortImages(
        view0_files, (4752, 3168), "../dataset/GrayCodes_HighRes/undistorted/view0/"
    )
    view1_files = getView1()
    undistortImages(
        view1_files, (4752, 3168), "../dataset/GrayCodes_HighRes/undistorted/view1/"
    )
