import glob
import os
import cv2
import numpy as np

from matcher import correspond
from poseRecovery import recoverPose


def triangulatePoints(K, R, t, kp0, kp1):
    proj0 = K @ np.hstack((np.eye(3), np.zeros((3,1))))

    proj1 = K @ np.hstack((R, t))

    kp0Temp = np.asarray([kp0[m.queryIdx].pt for m in matches])
    kp1Temp = np.asarray([kp1[m.queryIdx].pt for m in matches])

    points_4d_hom = cv2.triangulatePoints(proj0, proj1, kp0Temp, kp1Temp)
    points_3d = (points_4d_hom[:3] / points_4d_hom[3]).T
    return points_3d

if __name__ == "__main__":
    view0_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view0_files = sorted(view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view0_gray_codes = [np.load(file) for file in view0_files]

    view1_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view1/*.npy")
    view1_files = sorted(view1_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view1_gray_codes = [np.load(file) for file in view1_files]

    threshold = 8 # compare with 9 or 10 when testing as lower values have more points in shadow
    
    kp0, kp1, matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    R, t, mask_pose, K = recoverPose(kp0, kp1, matches)
    triangulatePoints(K, R, t, kp0, kp1) # TODO: manual triangulation