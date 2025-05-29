import cv2
import numpy as np

from utils.loadViewFiles import loadView
from matcher import correspond
from poseRecovery import recoverPose


def triangulatePoints(K, R, t, kp0, kp1):
    proj0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj1 = K @ np.hstack((R, t))

    points_4d_hom = cv2.triangulatePoints(proj0, proj1, kp0, kp1)
    return points_4d_hom


if __name__ == "__main__":
    view0_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view0/*.jpg")
    view1_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view1/*.jpg")

    threshold = 8  # compare with 9 or 10 when testing as lower values have more points in shadow

    kp0, kp1, matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    R, t, mask_pose, K = recoverPose(kp0, kp1, matches)
    # TODO: Manual triangulation
    print(triangulatePoints(K, R, t, kp0, kp1))
