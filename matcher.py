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
    view0 = decoder.decode(0.1)

    files = glob.glob("./GrayCodes/view1/*.jpg")
    files.sort()
    gray_codes = [undistort(file, Kmtx, dist) for file in files]

    decoder = GrayCodeDecoder(gray_codes)
    view1 = decoder.decode(0.1)

    view0posVal = dict()
    view1posVal = dict()

    valid_indices = view0.nonzero()
    
    coordCount = len(valid_indices[0])
    i = 0
    while i < coordCount:
        r = valid_indices[0][i]
        c = valid_indices[1][i]
        view0posVal[view0[r, c]] = (r, c)
        view1posVal[view1[r, c]] = (r, c)
        i += 1
    
    matches = list()
    
    for code in view0posVal.keys():
        try:
            matches.append((view0posVal[code], view1posVal[code]))
        except:
            pass
    print(matches)
            

if __name__ == "__main__":
    correspond()