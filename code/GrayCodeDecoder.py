import glob
import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from GrayCodeEncoder import GrayCodeEncoder
from calibration import undistort

def difference_indices(indices1, indices2):
    set1 = set(zip(*indices1))
    set2 = set(zip(*indices2))

    indices_set = set1.difference(set2)

    x = np.ndarray((0), dtype=np.uint32)
    y = np.ndarray((0), dtype=np.uint32)

    for idx in indices_set:
        x = np.append(x, [idx[0]])
        y = np.append(y, [idx[1]])

    return tuple([x, y])
     


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
    mask: np.uint32 = np.uint32(1) << np.uint32(n - 1)
    return nums | mask if value else nums & ~mask


class GrayCodeDecoder:
    def __init__(self, gray_codes: list[np.ndarray]):
        self.gray_codes = gray_codes
        self.shape = gray_codes[0].shape
        self.depth = get_depth_from_patterns(len(gray_codes))

    def decode(self, threshold: float | int) -> np.ndarray:
        full, patterns = self.__split_patterns()
        result = np.zeros(self.shape, dtype=np.uint32)
        proj_indices = self.__get_projection_indices(full, threshold)
        projection_area = np.zeros(self.shape, dtype=np.bool)
        projection_area[proj_indices] = True
        for i in range(0, 4 * self.depth, 2):
            current_projection_area = np.zeros(self.shape, dtype=np.bool) # bool matrix representing the pixels that are in the projection area for the current greycode

            img1: np.ndarray = patterns[i][proj_indices]
            img2: np.ndarray = patterns[i + 1][proj_indices]

            diff = cv2.absdiff(img1, img2) # absolute difference between the projection area's of pattern and its inverse

            if (img1.shape[0] < diff.shape[0]): # counter absdiff behavior when original shape is (2,) or smaller where it pads unneeded 0's
                diff = diff[0:img1.shape[0]]
            
            mask: np.ndarray

            if type(threshold) == float:
                mask = (diff > int(threshold * 255))
            else:
                mask = (diff > int(threshold))

            current_projection_area[proj_indices] = mask[:,0] # insert into bool matrix where absdiff is above threshold
            indices = current_projection_area.nonzero() # get coordinates of pixels in current projection area
            result[indices] = set_bit(result[indices], i // 2 + 1, 1) # update greycode for pixels in current projection area
            
            unknown_pixels = np.zeros(self.shape, dtype=np.bool) # bool matrix representing pixels whose greycode is unknown, and are thus invalid

            if type(threshold) == float:
                mask = (diff == int(threshold * 255))
            else:
                mask = (diff == threshold)
            
            unknown_pixels[proj_indices] = mask[:,0] # insert into bool matrix where absdiff is equal to threshold
            invalid_indices = unknown_pixels.nonzero() # get coordinates of pixels with absdiff smaller or equal to threshold or larger than 0
            result[invalid_indices] = 0 # set codes to zero because bit in current pattern was unknown, making whole code for that pixel unknown and thus invalid
            
            projection_area[invalid_indices] = False # set all invalid pixels in projection area to false 
            proj_indices = projection_area.nonzero() # update projection area indices
        return result

    def __split_patterns(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            self.gray_codes[:2],
            self.gray_codes[2 : ],
        )
    
    def __get_projection_indices(self, full: list[np.ndarray], threshold: float | int) -> tuple[np.ndarray]:
        proj_area = cv2.absdiff(full[0], full[1])
        if type(threshold) == float:
            proj_indices = np.asarray(proj_area > int(threshold * 255)).nonzero()
        else:
            proj_indices = np.asarray(proj_area > threshold).nonzero()
        return proj_indices


if __name__ == "__main__":
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]

    view0_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view0_files = sorted(view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    gray_codes = [undistort(file, Kmtx, dist) for file in view0_files]

    # encoder = GrayCodeEncoder(16, 16, 4)
    # gray_codes = [
    #     cv2.cvtColor(gray_code, cv2.COLOR_BGR2GRAY) for gray_code in encoder.patterns
    # ]

    decoder = GrayCodeDecoder(gray_codes)
    res0 = decoder.decode(0.2)
    # print(hor, vert, sep="\n\n")

    view1_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view1_files = sorted(view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    gray_codes = [undistort(file, Kmtx, dist) for file in view1_files]

    decoder = GrayCodeDecoder(gray_codes)
    res1 = decoder.decode(0.2)
