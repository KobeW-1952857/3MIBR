import open3d as o3d
import numpy as np

if __name__ == "__main__":
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    extrinsics = data["extrinsics"]
    img_size = data["img_size"]

    widht, height = img_size

    line_sets = []

    for extrinsic in extrinsics:
        line_sets.append(
            o3d.geometry.LineSet.create_camera_visualization(
                widht, height, Kmtx, extrinsic
            )
        )
    o3d.visualization.draw_geometries(line_sets)
