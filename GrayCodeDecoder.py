import cv2
import numpy as np
import math

from GrayCodeEncoder import GrayCodeEncoder


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


def set_bit(num: np.uint8, n: int, value: int) -> np.uint8:
    mask: np.uint8 = np.uint8(1) << np.uint8(n - 1)
    return num | mask if value else num & ~mask


class GrayCodeDecoder:
    def __init__(self, gray_codes: list[np.ndarray], shape: tuple[int, int]):
        self.gray_codes = gray_codes
        for i in range(len(gray_codes)):
            self.gray_codes[i] = cv2.cvtColor(self.gray_codes[i], cv2.COLOR_BGR2GRAY)
        self.shape = shape
        self.depth = get_depth_from_patterns(len(gray_codes))

    def decode(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        full, vertical, horizontal = self.__split_patterns()
        result_horizontal = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)
        result_vertical = np.zeros((self.shape[0], self.shape[1]), dtype=np.uint8)

        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                # TODO: Check if pixel is inside capture region with full pattern
                for i in range(0, 2 * self.depth, 2):
                    # Vertical part
                    if vertical[i][x, y] - vertical[i + 1][x, y] > threshold * 255:
                        result_horizontal[x, y] = set_bit(
                            result_horizontal[x, y], i // 2 + 1, 1
                        )
                    else:
                        result_horizontal[x, y] = set_bit(
                            result_horizontal[x, y], i // 2 + 1, 0
                        )
                    # Horizontal part
                    if horizontal[i][x, y] - horizontal[i + 1][x, y] > threshold * 255:
                        result_vertical[x, y] = set_bit(
                            result_vertical[x, y], i // 2 + 1, 1
                        )
                    else:
                        result_vertical[x, y] = set_bit(
                            result_vertical[x, y], i // 2 + 1, 0
                        )
        return result_horizontal, result_vertical

    def __split_patterns(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        return (
            self.gray_codes[:2],
            self.gray_codes[2 : 2 + self.depth * 2],
            self.gray_codes[2 + self.depth * 2 :],
        )


if __name__ == "__main__":
    gray_codes = []
    encoder = GrayCodeEncoder(16, 16, 3)
    decoder = GrayCodeDecoder(encoder.patterns, (16, 16))
    print(decoder.decode(0.5))
