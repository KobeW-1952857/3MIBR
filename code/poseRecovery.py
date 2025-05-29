import glob
import os
import cv2
import numpy as np

from calibration import combine_extrinsic_vec
from visualizeCameraPoints import drawCameraPoints
from utils.loadViewFiles import loadView
from essentialMatrixGeneration import generateEssentialMatrix
from matcher import correspond


def recoverPose(essential_matrix, kp0, kp1, intrinsic, mask):
    _, rotation_matrix, translation_vector, mask_pose = cv2.recoverPose(
        essential_matrix, kp0, kp1, intrinsic, mask=mask
    )
    return rotation_matrix, translation_vector, mask_pose


if __name__ == "__main__":
    view0_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view0/*.jpg")
    view1_gray_codes = loadView("../dataset/GrayCodes_HighRes/undistorted/view1/*.jpg")

    data = np.load("calibration.npz")
    intrinsic = data["Kmtx"]
    dist = data["dist"]
    # extrinsics = data["extrinsics"]
    img_size = data["img_size"]

    threshold = 8  # compare with 9 or 10 when testing as lower values have more points in shadow
    kp0, kp1, matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    print("Found matches")
    rotation, translation, mask, K = recoverPose(kp0, kp1, matches)

    cam_0_rotation = np.eye(3)
    cam_0_translation = np.zeros((3, 1))
    cam_0_extrinsic = combine_extrinsic_vec(cam_0_rotation, cam_0_translation)
    cam_1_extrinsic = combine_extrinsic_vec(rotation, translation)

    drawCameraPoints(
        img_size[0], img_size[1], intrinsic, [cam_0_extrinsic, cam_1_extrinsic]
    )
