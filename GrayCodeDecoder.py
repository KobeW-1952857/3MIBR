import glob
import cv2
import numpy as np
import math

from GrayCodeEncoder import GrayCodeEncoder
from calibration import undistort


def gray_to_binary(num: int) -> int:
    """
    The purpose of this function is to convert a reflected binary Gray code to an unsigned binary number.
    """
    mask = num
    while mask:
        mask >>= 1
        num ^= mask
    return num


def get_depth_from_patterns(pattern_length: int) -> int:
    return (pattern_length - 2) // 4


# def set_bit(num: np.uint8, n: int, value: int) -> np.uint8:
#     mask: np.uint8 = np.uint8(1) << np.uint8(n - 1)
#     return num | mask if value else num & ~mask


def set_bit(nums: np.ndarray, n: int, value: int) -> np.ndarray:
    mask: np.uint8 = np.uint8(1) << np.uint8(n - 1)
    return nums | mask if value else nums & ~mask


class GrayCodeDecoder:
    def __init__(self, gray_codes: list[np.ndarray]):
        self.gray_codes = gray_codes
        self.shape = gray_codes[0].shape
        self.depth = get_depth_from_patterns(len(gray_codes))

    def decode(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        full, vertical, horizontal = self.__split_patterns()
        result_horizontal = np.zeros(self.shape, dtype=np.uint8)
        result_vertical = np.zeros(self.shape, dtype=np.uint8)
        proj_indices = self.__get_projection_indices(full, threshold)

        for i in range(0, 2 * self.depth, 2):
            padded_diff = np.zeros(self.shape, dtype=np.bool)
            diff = cv2.absdiff(horizontal[i][proj_indices[0], proj_indices[1]], horizontal[i + 1][proj_indices[0], proj_indices[1]])
            padded_diff[proj_indices[0], proj_indices[1]] = np.asarray(diff > threshold * 255)[:,0]
            indices = padded_diff.nonzero()
            result_horizontal[indices] = set_bit(
                result_horizontal[indices], i // 2 + 1, 1
            )
            padded_diff = np.zeros(self.shape, dtype=np.bool)
            diff = cv2.absdiff(vertical[i][proj_indices[0], proj_indices[1]], vertical[i + 1][proj_indices[0], proj_indices[1]])
            padded_diff[proj_indices[0], proj_indices[1]] = np.asarray(diff > threshold * 255)[:,0]
            indices = padded_diff.nonzero()
            result_vertical[indices] = set_bit(result_vertical[indices], i // 2 + 1, 1)

        return result_horizontal, result_vertical

    def __split_patterns(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            self.gray_codes[:2],
            self.gray_codes[2 : 2 + self.depth * 2],
            self.gray_codes[2 + self.depth * 2 :],
        )
    def __get_projection_indices(self, full: list[np.ndarray], threshold: float) -> tuple[np.ndarray]:
        proj_area = cv2.absdiff(full[0], full[1])
        proj_indices = np.asarray(proj_area > threshold * 255).nonzero()
        return proj_indices


if __name__ == "__main__":
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]

    files = glob.glob("./GrayCodes/view0/*.jpg")
    files.sort()
    gray_codes = [undistort(file, Kmtx, dist) for file in files]

    # encoder = GrayCodeEncoder(16, 16, 4)
    # gray_codes = [
    #     cv2.cvtColor(gray_code, cv2.COLOR_BGR2GRAY) for gray_code in encoder.patterns
    # ]

    decoder = GrayCodeDecoder(gray_codes)
    hor0, vert0 = decoder.decode(0.5)
    # print(hor, vert, sep="\n\n")

    files = glob.glob("./GrayCodes/view1/*.jpg")
    files.sort()
    gray_codes = [undistort(file, Kmtx, dist) for file in files]

    decoder = GrayCodeDecoder(gray_codes)
    hor1, vert1 = decoder.decode(0.5)
