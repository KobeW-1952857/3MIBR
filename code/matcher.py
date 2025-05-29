from collections import defaultdict
import glob

from math import ceil
import os
from threading import Thread
from time import sleep
from idna import decode
from matplotlib import pyplot as plt
import numpy as np
from GrayCodeDecoder import GrayCodeDecoder
import cv2 as cv
import random

from utils.loadViewFiles import loadView


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


def correspond(decoded_views: dict[str, dict[str, np.ndarray]]):
    # decoder = GrayCodeDecoder(view0_gray_codes)
    # view0, areaMask0 = decoder.decode(threshold)

    # decoder = GrayCodeDecoder(view1_gray_codes)
    # view1, areaMask1 = decoder.decode(threshold)
    areaMask0 = decoded_views["view0"]["mask"]
    areaMask1 = decoded_views["view1"]["mask"]
    view0 = decoded_views["view0"]["codes"]
    view1 = decoded_views["view1"]["codes"]

    view0_value_positions: defaultdict[np.int64, list[tuple[float, float]]] = (
        defaultdict(list)
    )
    view1_value_positions: defaultdict[np.int64, list[tuple[float, float]]] = (
        defaultdict(list)
    )

    mask = areaMask0 | areaMask1

    valid_indices = mask.nonzero()  # only search relevant indices

    for i in range(len(valid_indices[0])):
        r = valid_indices[0][i]
        c = valid_indices[1][i]
        if areaMask0[r, c]:
            view0_value_positions[view0[r, c]].append((r, c))
        if areaMask1[r, c]:
            view1_value_positions[view1[r, c]].append((r, c))

    codePoints0 = list()
    kp1 = list()
    codePoints1 = list()
    kp0 = list()

    matches: dict[np.int64, list[tuple[float, float]]] = dict()

    dMatches = list()
    count = 0
    for code in view0_value_positions.keys():
        if code in view1_value_positions:
            matches[code] = [
                averageCoords(view0_value_positions[code]),
                averageCoords(view1_value_positions[code]),
            ]
            avgCoord1 = averageCoords(view1_value_positions[code])
            # find common code positions
            kp1.append(cv.KeyPoint(avgCoord1[0], avgCoord1[1], 1, -1))
            avgCoord0 = averageCoords(view0_value_positions[code])
            # find common code positions
            kp0.append(cv.KeyPoint(avgCoord0[0], avgCoord0[1], 1, -1))
            # get all detected points
            codePoints0.append(cv.KeyPoint(avgCoord0[0], avgCoord0[1], 1, -1))
            codePoints1.append(cv.KeyPoint(avgCoord1[0], avgCoord1[1], 1, -1))
            dMatches.append(cv.DMatch(_queryIdx=count, _trainIdx=count, _distance=0))
            count += 1
    # print(matches)
    return kp0, kp1, dMatches, matches


def createKeypointsAndDMatches(matches):
    kp0 = list()
    kp1 = list()
    dMatches = list()
    for i, (code, coords) in enumerate(matches.items()):
        kp0.append(cv.KeyPoint(coords[0][0], coords[0][1], 1, -1))
        kp1.append(cv.KeyPoint(coords[1][0], coords[1][1], 1, -1))
        dMatches.append(cv.DMatch(_queryIdx=i, _trainIdx=i, _distance=0))
    return kp0, kp1, dMatches


def drawMatches(matches, img0, img1, sample_size=100):
    kp0, kp1, dMatches = createKeypointsAndDMatches(matches)
    random_sample = np.zeros(len(dMatches))
    random_sample[np.random.choice(len(dMatches), size=sample_size, replace=False)] = (
        True
    )

    matchImg = cv.drawMatches(
        img0,
        kp0,
        img1,
        kp1,
        dMatches,
        None,
        matchesMask=random_sample.astype(np.uint8),
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv.imshow("Matches", matchImg)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    view0_gray_codes = loadView("dataset/GrayCodes_HighRes/undistorted/view0/*.jpg")
    view1_gray_codes = loadView("dataset/GrayCodes_HighRes/undistorted/view1/*.jpg")

    # compare with 9 or 10 when testing as lower values have more points in shadow
    threshold = 8
    kp0, kp1, dMatches = correspond(threshold, view0_gray_codes, view1_gray_codes)
    drawMatches(kp0, kp1, dMatches, view0_gray_codes, view1_gray_codes)
