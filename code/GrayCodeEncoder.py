import math
import cv2
import numpy as np
import numpy.typing as npt
import optparse as opt


def binary_to_gray(num: int) -> int:
    """
    The purpose of this function is to convert an unsigned
    binary number to reflected binary Gray code.

    The operator >> is shift right. The operator ^ is exclusive or.
    Source: http://en.wikipedia.org/wiki/Gray_code
    """
    return (num >> 1) ^ num


def get_bit(decimal: int, n: int) -> int:
    """
    Function takes the decimal number
    Function takes the Nth bit (1 to 31)
    Return the value of Nth bit from decimal
    Source: http://icfun.blogspot.com/2009/04/get-n-th-bit-value-of-any-integer.html
    """
    constant = 1 << (n - 1)
    return 1 if decimal & constant else 0


class GrayCodeEncoder:
    def __init__(self, rows: int, cols: int, depth: int):
        self.cols = cols
        self.rows = rows
        self.n = depth * 4  # number of frames to be computed
        self.patterns: list[np.ndarray] = []

        n_horiz_bits = math.ceil(math.log2(float(self.cols)))
        n_vert_bits = math.ceil(math.log2(float(self.rows)))

        # compute horizontal encoding patterns
        full_white = np.ones((rows, cols, 3), np.uint8) * 255
        self.patterns.append(full_white)
        self.patterns.append(cv2.bitwise_not(full_white))

        for i in range(depth):
            pattern = np.zeros((1, cols, 3), np.uint8)
            # loop through columns in first row
            for j in range(cols):
                gray = binary_to_gray(j)
                # amplitude of channels
                amp = get_bit(gray, n_horiz_bits - i)
                pattern[0, j, :] = [255.0 * amp, 255.0 * amp, 255.0 * amp]
            pattern = pattern.repeat(rows, axis=0)
            self.patterns.append(pattern)
            self.patterns.append(cv2.bitwise_not(pattern))

        # compute vertical encoding patterns
        for i in range(depth):
            pattern = np.zeros((rows, 1, 3), np.uint8)
            # Loop through rows in first column
            for j in range(rows):
                gray = binary_to_gray(j)
                # Amplitude of channels
                amp = get_bit(gray, n_vert_bits - i)
                pattern[j, 0, :] = [255.0 * amp, 255.0 * amp, 255.0 * amp]
            pattern = pattern.repeat(cols, axis=1)
            self.patterns.append(pattern)
            self.patterns.append(cv2.bitwise_not(pattern))

    def get_encoding_pattern(self, depth) -> np.ndarray:
        return self.patterns[depth]


if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = opt.OptionParser(usage=usage)
    parser.add_option(
        "-r",
        "--rows",
        dest="rows",
        type="int",
        default=1080,
        help="Number of rows for the patterns [default: %default]",
    )
    parser.add_option(
        "-c",
        "--cols",
        dest="cols",
        type="int",
        default=1920,
        help="Number of columns for the patterns [default: %default]",
    )
    parser.add_option(
        "-d",
        "--depth",
        dest="depth",
        type="int",
        default=10,
        help="Depth for Gray code encoding [default: %default]",
    )
    parser.add_option(
        "-o",
        "--output",
        dest="output_prefix",
        type="string",
        default="pattern_",
        help="Prefix for output image files [default: %default]",
    )
    parser.add_option(
        "-s",
        "--show",
        dest="show_images",
        action="store_true",
        default=False,
        help="Show images as they are generated",
    )

    (options, args) = parser.parse_args()  # Parses arguments from sys.argv

    encoder = GrayCodeEncoder(options.rows, options.cols, options.depth)
    print(
        f"Generating {len(encoder.patterns)} patterns with {options.cols}x{options.rows} and depth {options.depth}."
    )

    for i, pattern in enumerate(encoder.patterns):
        if options.show_images:
            cv2.imshow(str(i), pattern)
            cv2.waitKey(0)  # Wait for a key press to show the next image

        output_filename = f"{options.output_prefix}{i}.png"
        cv2.imwrite(output_filename, pattern)
        print(f"\rSaved {output_filename}", end="")
    print()

    if options.show_images:
        cv2.destroyAllWindows()
    print("Done.")
