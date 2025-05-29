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


def triangulatePointsCustom(projection0, projection1, kp0, kp1):
    p0_row0, p0_row1, p0_row2 = projection0
    p1_row0, p1_row1, p1_row2 = projection1
    print("split projection matrices")
    points_4d_hom = []

    for point0, point1 in zip(kp0, kp1):
        x0, y0 = point0
        x1, y1 = point1
        A = np.vstack(
            (
                p0_row0 - x0 * p0_row2,
                p0_row1 - y0 * p0_row2,
                p1_row0 - x1 * p1_row2,
                p1_row1 - y1 * p1_row2,
            )
        )
        _, _, Vt = cv2.SVDecomp(A)
        w = Vt[-1, :]
        points_4d_hom.append(w)

    return np.array(points_4d_hom).T


if __name__ == "__main__":
    view0_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view0/*.jpg")
    view1_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view1/*.jpg")

    threshold = 8  # compare with 9 or 10 when testing as lower values have more points in shadow

    kp0, kp1, matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    R, t, mask_pose, K = recoverPose(kp0, kp1, matches)
    # TODO: Manual triangulation
    print(triangulatePoints(K, R, t, kp0, kp1))
