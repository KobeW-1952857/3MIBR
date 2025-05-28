import glob
import os
import cv2
import numpy as np

from matcher import correspond


def generateEssentialMatrix():
    view0_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view0_files = sorted(
        view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    view0_gray_codes = [np.load(file) for file in view0_files]

    view1_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view1/*.npy")
    view1_files = sorted(
        view1_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    view1_gray_codes = [np.load(file) for file in view1_files]

    threshold = 8  # compare with 9 or 10 when testing as lower values have more points in shadow

    kp0, kp1, _ = correspond(threshold, view0_gray_codes, view1_gray_codes)

    Kmtx = np.load("calibration.npz")["Kmtx"]

    E, mask = cv2.findEssentialMat(
        np.array([kp.pt for kp in kp0]),
        np.array([kp.pt for kp in kp1]),
        Kmtx,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    return E, mask


if __name__ == "__main__":
    print(generateEssentialMatrix())
