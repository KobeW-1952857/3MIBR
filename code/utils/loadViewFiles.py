from glob import glob
import os.path
from cv2 import imread, IMREAD_GRAYSCALE


def loadView(filepath: str):
    view_files = glob(filepath)
    view_files = sorted(
        view_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    return [imread(file, IMREAD_GRAYSCALE) for file in view_files]
