import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp


def getDepthMap(intrinsic, extrinsic, points, image_size):
    out_img = np.zeros(image_size, dtype=np.float32)

    transformed_points = extrinsic @ points
    for point in transformed_points.T:
        P_img = intrinsic @ point
        v = int(P_img[0] / P_img[2])
        u = int(P_img[1] / P_img[2])
        try:
            out_img[u, v] = -point[2]
        except:
            pass
    return ((out_img / np.max(np.abs(out_img))) * 255).astype(np.uint8)


def slerp(q0, q1, t):
    # Ensure input quaternions are unit quaternions
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot_product = np.dot(q0, q1)

    # 1. Handle shortest path: If dot product is negative, flip one quaternion
    if dot_product < 0.0:
        q1 = -q1
        dot_product = -dot_product  # Update dot product after flipping

    # Clamp dot_product to prevent numerical issues (should be in [-1, 1])
    dot_product = np.clip(dot_product, -1.0, 1.0)

    omega = np.arccos(dot_product)
    sin_omega = np.sin(omega)

    # 2. Handle near-parallel quaternions (avoid division by zero)
    if abs(sin_omega) < 1e-6:  # Quaternions are very close or identical
        # Fallback to linear interpolation (LERP) and normalize
        return (1.0 - t) * q0 + t * q1

    # Standard SLERP formula
    coeff0 = np.sin((1.0 - t) * omega) / sin_omega
    coeff1 = np.sin(t * omega) / sin_omega

    q_interpolated = coeff0 * q0 + coeff1 * q1

    # Normalize the result (good practice due to floating point inaccuracies)
    return q_interpolated / np.linalg.norm(q_interpolated)


def interpolateBetweenCams(extrinsic0, extrinsic1, alpha):
    rotation0 = Rotation.from_matrix(extrinsic0[:3, :3])
    rotation1 = Rotation.from_matrix(extrinsic1[:3, :3])
    translation0 = extrinsic0[:3, 3]
    translation1 = extrinsic1[:3, 3]

    interpolated_translation = (1 - alpha) * translation0 + alpha * translation1
    interpolated_rotation = Rotation.from_quat(
        slerp(rotation0.as_quat(), rotation1.as_quat(), alpha)
    ).as_matrix()

    return np.hstack((interpolated_rotation, interpolated_translation.reshape(-1, 1)))
