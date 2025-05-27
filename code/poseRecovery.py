import cv2

from essentialMatrixGeneration import generateEssentialMatrix


if __name__ == "__main__":
    E, K = generateEssentialMatrix()
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)