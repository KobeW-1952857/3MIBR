import glob

import numpy as np
from pyparsing import col
from GrayCodeDecoder import GrayCodeDecoder
from calibration import undistort

def correspond():
    data = np.load("calibration.npz")
    Kmtx = data["Kmtx"]
    dist = data["dist"]

    files = glob.glob("./GrayCodes/view0/*.jpg")
    files.sort()
    gray_codes = [undistort(file, Kmtx, dist) for file in files]

    decoder = GrayCodeDecoder(gray_codes)
    view0 = decoder.decode(0.2)

    files = glob.glob("./GrayCodes/view1/*.jpg")
    files.sort()
    gray_codes = [undistort(file, Kmtx, dist) for file in files]

    decoder = GrayCodeDecoder(gray_codes)
    view1 = decoder.decode(0.2)

    view0posVal = dict()
    view1posVal = dict()

    valid_indices = view0.nonzero()
    print(valid_indices)
    
    coordCount = len(valid_indices[0])
    i = 0
    while i < coordCount:
        r = valid_indices[0][i]
        c = valid_indices[1][i]
        view0posVal[view0[r, c]] = (r, c)
        view1posVal[view1[r, c]] = (r, c)
    
    matches = list()

    for pos in view0posVal.items():
        code = view0[pos[0], pos[1]]
        matches.append((view0posVal[code], view1posVal[code]))

    print(matches)
            

if __name__ == "__main__":
    correspond()