import open3d as o3d
import open3d.visualization
import numpy as np
from numpy.typing import NDArray


def createLineSet(height, width, intrinsic, extrinsic):
    line_set = o3d.geometry.LineSet.create_camera_visualization(
        width, height, intrinsic, extrinsic
    )
    return line_set


def drawCameraPoints(
    height: int,
    width: int,
    intrinsic: NDArray[np.float64],
    extrinsics: list[NDArray[np.float64]],
):
    line_sets = [
        o3d.geometry.LineSet.create_camera_visualization(
            width, height, intrinsic, extrinsic
        )
        for extrinsic in extrinsics
    ]
    open3d.visualization.draw_geometries(
        line_sets,
        front=np.array([0.0, 0.0, -1.0]),
        up=np.array([0.0, -1.0, 0.0]),
    )


if __name__ == "__main__":
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    extrinsics = data["extrinsics"]
    img_size = data["img_size"]

    width, height = img_size
    drawCameraPoints(height, width, Kmtx, extrinsics)
