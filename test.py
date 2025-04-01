import cv2 as cv
import numpy as np

if __name__ == "__main__":
    img1 = cv.imread("./GrayCodes/view0/10.jpg")
    img2 = cv.imread("./GrayCodes/view0/11.jpg")

    cv.imshow("Difference normal - inverse", img1 - img2)
    cv.imshow("Difference inverse - normal", img2 - img1)
    cv.waitKey(0)
    cv.destroyAllWindows()
