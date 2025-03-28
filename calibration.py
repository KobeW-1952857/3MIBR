import numpy as np
import numpy.typing as npt
import cv2 as cv
import cv2.typing as cvt
import glob


def loadChessCorners(
    file: str,
    objpoints: list,
    imgpoints: list,
    grid_size: tuple[int, int],
    criteria: tuple[int, int, float],
):
    grid_x, grid_y = grid_size
    objp = np.zeros((grid_x * grid_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_x, 0:grid_y].T.reshape(-1, 2)

    img = cv.imread(file)
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, grid_size, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)


def intrinsic_calibration(file_names: list, grid_size: tuple[int, int]):
    print("Staring calibration")
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for i, fn in enumerate(file_names):
        loadChessCorners(fn, objpoints, imgpoints, grid_size, criteria)
        print(f"\rProgress: {i / len(file_names) * 100:.2f}%", end="")
    print("\nCalibration complete")
    img = cv.imread(file_names[0])
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)

    # Calibration
    return cv.calibrateCamera(
        objpoints, imgpoints, img.shape[::2], None, None, criteria=criteria
    )


def undistort(file: str, Kmtx: cvt.MatLike, dist: cvt.MatLike):
    img = cv.imread(file)
    img = cv.resize(img, (0, 0), fx=0.1, fy=0.1)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(Kmtx, dist, (w, h), 1, (w, h))
    return cv.undistort(gray, Kmtx, dist, None, newcameramtx)


def combine_extrinsic_vecs(rvecs, tvecs):
    extrinsics = []
    for rvec, tvec in zip(rvecs, tvecs):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cv.Rodrigues(rvec)[0]
        extrinsic[:3, 3] = tvec.flatten()
        extrinsics.append(extrinsic)
    return extrinsics


def get_image_dimensions(file_path: str) -> tuple[int, int]:
    img = cv.imread(file_path)
    return img.shape[:2]


if __name__ == "__main__":
    files = glob.glob("./GrayCodes/chess/*.jpg")
    rms, Kmtx, dist, rvecs, tvecs = intrinsic_calibration(files, (7, 9))
    extrinsics = combine_extrinsic_vecs(rvecs, tvecs)
    img_size = get_image_dimensions(files[0])
    print(img_size)

    np.savez(
        "calibration.npz",
        Kmtx=Kmtx,
        dist=dist,
        extrinsics=extrinsics,
        img_size=img_size,
    )
