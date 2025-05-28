import open3d as o3d
import numpy as np
from numpy.typing import NDArray


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
    o3d.visualization.draw_geometries(line_sets)


if __name__ == "__main__":
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    extrinsics = data["extrinsics"]
    img_size = data["img_size"]

    width, height = img_size
    drawCameraPoints(height, width, Kmtx, extrinsics)
