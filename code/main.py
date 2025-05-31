# General imports
import argparse
import cv2
import numpy as np
from glob import glob
import os.path
from os import makedirs

from calibration import combine_extrinsic_vecs, intrinsic_calibration

BASE_DATA_PATH = "/Users/kobe/Documents/School/2024-2025/3D_Modeling_and_Image_Based_Rendering/project/dataset"
CURRENT_DATASET = BASE_DATA_PATH + "/GrayCodes_HighRes"
CHESS_PATH = CURRENT_DATASET + "/chess"
RAW_PATH = CURRENT_DATASET + "/raw"
UNDIST_PATH = CURRENT_DATASET + "/undistorted"
PATTERN_PATH = CURRENT_DATASET + "/patterns"


def useLoadedData():
    print("Using loaded data")

    print(f"Loading camera calibration data at {CURRENT_DATASET}/camCalibration.npz")
    calibration_data = np.load(f"{CURRENT_DATASET}/camCalibration.npz")
    intrinsic = calibration_data["intrinsic"]
    distortion = calibration_data["distortion"]
    extrinsics = calibration_data["extrinsics"]
    image_size = calibration_data["img_size"]

    print(f"Loading undistorted images at {UNDIST_PATH}")


def computeAllData():
    print("Computing all data")
    if not os.path.exists(UNDIST_PATH):
        makedirs(UNDIST_PATH)

    if not os.path.exists(PATTERN_PATH):
        makedirs(PATTERN_PATH)


parser = argparse.ArgumentParser(
    prog="3D Modeling and Image Based Rendering Project",
    description="Uses multiple images to create a 3D scene of the pictures",
)
parser.add_argument("-l", "--load", action="store_true", help="Load pre-processed data")
parser.add_argument(
    "-f",
    "--folder",
    type=str,
    help="Base path that will be used for all needed folders",
)
args = parser.parse_args()

if args.folder:
    CURRENT_DATASET = args.folder

if args.load:
    useLoadedData()
else:
    computeAllData()


def getFilesSortedNumeric(glob_pattern: str):
    files = glob(glob_pattern)
    return sorted(files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))


def calibrate_camera():
    chess_files = glob(f"{CHESS_PATH}/*.jpg")
    error, intrinsic, distortion, rotation_matrices, translation_vectors, image_size = (
        intrinsic_calibration(chess_files, (7, 9))
    )
    extrinsics = combine_extrinsic_vecs(rotation_matrices, translation_vectors)

    np.savez(
        f"{CURRENT_DATASET}/camCalibration.npz",
        intrinsic=intrinsic,
        distortion=distortion,
        extrinsics=extrinsics,
        img_size=image_size,
    )
