import glob

import numpy as np
from pyparsing import col
from GrayCodeDecoder import GrayCodeDecoder
from calibration import undistort, undistortCustom
import cv2 as cv

def correspond():
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]
    newcameramtx, _ = cv.getOptimalNewCameraMatrix(Kmtx, dist, (1584, 1056), 1, (1584, 1056))

    files = glob.glob("./GrayCodes/view0/*.jpg")
    files.sort()
    gray_codes = [undistortCustom(file, Kmtx, dist, newcameramtx, (1584, 1056)) for file in files]

    print("progress")
    
    # decoder need fix since it returns a lot of double codes which shouldn't be happening
    decoder = GrayCodeDecoder(gray_codes) # too slow, need to find fix. decoder needs to be able to decode 2073600 (1920X1080) pixels in reasonable time
    view0 = decoder.decode(0.1)

    print("progress")

    files = glob.glob("./GrayCodes/view1/*.jpg")
    files.sort()
    gray_codes = [undistortCustom(file, Kmtx, dist, newcameramtx, (1584, 1056)) for file in files]

    print("progress")

    decoder = GrayCodeDecoder(gray_codes)
    view1 = decoder.decode(0.1)

    print("progress")

    view0posVal = dict()
    view1posVal = dict()

    valid_indices = view0.nonzero()
    
    coordCount = len(valid_indices[0])
    i = 0
    values = []
    while i < coordCount:
        r = valid_indices[0][i]
        c = valid_indices[1][i]
        values.append(view0[r, c])
        view0posVal[view0[r, c]] = (r, c)
        view1posVal[view1[r, c]] = (r, c)
        i += 1

    print(values)

    matches = list()
    
    for code in view0posVal.keys():
        try:
            matches.append((view0posVal[code], view1posVal[code]))
        except:
            pass
    print(matches)
            

if __name__ == "__main__":
    correspond()