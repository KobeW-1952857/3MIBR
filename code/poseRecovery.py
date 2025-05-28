import glob
import os
import cv2
import numpy as np

from essentialMatrixGeneration import generateEssentialMatrix
from matcher import correspond

def recoverPose(kp0, kp1, matches):
    E, mask, K = generateEssentialMatrix(kp0, kp1, matches)

    kp0Temp = np.asarray([kp0[m.queryIdx].pt for m in matches])
    kp1Temp = np.asarray([kp1[m.queryIdx].pt for m in matches])

    _, R, t, mask_pose = cv2.recoverPose(E, kp0Temp, kp1Temp, K, mask=mask)
    return R, t, mask_pose

if __name__ == "__main__":
    view0_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view0_files = sorted(view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view0_gray_codes = [np.load(file) for file in view0_files]

    view1_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view1/*.npy")
    view1_files = sorted(view1_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view1_gray_codes = [np.load(file) for file in view1_files]

    threshold = 8 # compare with 9 or 10 when testing as lower values have more points in shadow

    kp0, kp1, matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    recoverPose(kp0, kp1, matches)