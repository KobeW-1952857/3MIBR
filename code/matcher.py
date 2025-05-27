import glob

from math import ceil
import os
from threading import Thread
from time import sleep
from matplotlib import pyplot as plt
import numpy as np
from GrayCodeDecoder import GrayCodeDecoder
import cv2 as cv
import random

def averageCoords(coords: list):
    n = len(coords)
    x: float = 0
    y: float = 0
    for coord in coords:
        x += coord[0]
        y += coord[1]
    x = x / n
    y = y / n

    return (y, x)

def correspond(threshold: float | int, view0_gray_codes: list, view1_gray_codes: list):
    decoder = GrayCodeDecoder(view0_gray_codes)
    view0, areaMask0 = decoder.decode(threshold)

    decoder = GrayCodeDecoder(view1_gray_codes)
    view1, areaMask1= decoder.decode(threshold)

    view0_value_positions = dict()
    view1_value_positions = dict()

    mask = areaMask0 | areaMask1

    valid_indices = mask.nonzero() # only search relevant indices
    
    coord_count = len(valid_indices[0])
    i = 0
    while i < coord_count:
        c = valid_indices[0][i]
        r = valid_indices[1][i]
        if areaMask0[c, r]:
            if view0[c, r] not in view0_value_positions:
                view0_value_positions[view0[c, r]] = [(c, r)]
            else:
                view0_value_positions[view0[c, r]].append((c, r))
        if areaMask1[c, r]:
            if view1[c, r] not in view1_value_positions:
                view1_value_positions[view1[c, r]] = [(c, r)]
            else:
                view1_value_positions[view1[c, r]].append((c, r))
        i += 1

    codePoints0 = list()
    kp1 = list()
    codePoints1 = list()
    kp0 = list()

    dMatches = list()
    count = 0
    for code in view0_value_positions.keys():
        if code in view1_value_positions:
            avgCoord1 = averageCoords(view1_value_positions[code])
            kp1.append(cv.KeyPoint(avgCoord1[0], avgCoord1[1], 1, -1)) # find common code positions
            avgCoord0 = averageCoords(view0_value_positions[code])
            kp0.append(cv.KeyPoint(avgCoord0[0], avgCoord0[1], 1, -1)) # find common code positions
            codePoints0.append(cv.KeyPoint(avgCoord0[0], avgCoord0[1], 1, -1)) # get all detected points
            codePoints1.append(cv.KeyPoint(avgCoord1[0], avgCoord1[1], 1, -1))
            dMatches.append(cv.DMatch(_queryIdx=count, _trainIdx=count, _distance=0))
            count += 1

    return kp0, kp1, dMatches

def drawMatches(kp0: list, kp1: list, dMatches: list, view0_gray_codes: list, view1_gray_codes: list, matchAmount: int = 1000):
    sampleIndices = list(np.random.randint(0, len(dMatches), size=(matchAmount)))
    kp0Sample = [kp0[i] for i in sampleIndices]
    kp1Sample = [kp1[i] for i in sampleIndices]
    dMatchesSample = dMatches[:matchAmount]

    matchImg = cv.drawMatches(view0_gray_codes[0], kp0Sample, view1_gray_codes[0], kp1Sample, dMatchesSample, None)
    plt.imshow(matchImg)
    plt.show()

results = list()

def find_ideal_threshold(minThreshold: int, maxThreshold: int, donenessIndex: int, view0_gray_codes: list, view1_gray_codes: list):
    maxMatches = -1
    bestThreshold = -1
    for i in range(minThreshold, maxThreshold):
        threshold = i
        matches = correspond(threshold, view0_gray_codes, view1_gray_codes)
        if maxMatches == -1 or maxMatches < len(matches):
            maxMatches = len(matches)
            bestThreshold = threshold
    results[donenessIndex] = (maxMatches, bestThreshold)

def find_ideal_threshold_threaded(threadCount: int = 12, threshRangeStart = 0, threshRangeEnd = 256):
    view0_files = glob.glob("./GrayCodes/undistorted/view0/*.npy")
    view0_files.sort()
    view0_gray_codes = [np.load(file) for file in view0_files]

    view1_files = glob.glob("./GrayCodes/undistorted/view1/*.npy")
    view1_files.sort()
    view1_gray_codes = [np.load(file) for file in view1_files]

    threads: list = list()
    step = int(ceil(threshRangeEnd - threshRangeStart / threadCount))
    previousStep = threshRangeStart
    currentStep = step
    for i in range(threadCount):
        results.append(())
        t = Thread(target=find_ideal_threshold, args=(previousStep, currentStep, i, view0_gray_codes, view1_gray_codes))
        threads.append(t)
        t.start()
        previousStep = currentStep
        currentStep += step
        currentStep = min(threshRangeEnd - 1, currentStep)
    done = False
    while done == False:
        done = True
        for r in results:
            done = done and r != ()
            if not done:
                break
        if not done:
            sleep(10)
    
    for t in threads: # may be unnescesary but present just in case
        t.join()

    bestThreshold: int = -1
    maxMatches: int = -1
    for r in results:
        if maxMatches == -1 or maxMatches < r[0]:
            maxMatches = r[0]
            bestThreshold = r[1]
        
    return (maxMatches, bestThreshold)
     
            

if __name__ == "__main__":
    view0_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view0/*.npy")
    view0_files = sorted(view0_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view0_gray_codes = [np.load(file) for file in view0_files]

    view1_files = glob.glob("../dataset/GrayCodes_HighRes/undistorted/view1/*.npy")
    view1_files = sorted(view1_files, key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    view1_gray_codes = [np.load(file) for file in view1_files]

    threshold = 8 # compare with 9 or 10 when testing as lower values have more points in shadow
    kp0, kp1, dMatches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    drawMatches(kp0, kp1, dMatches, view0_gray_codes, view1_gray_codes)

    # print(find_ideal_threshold_threaded(threshRangeStart=24, threshRangeEnd=100))