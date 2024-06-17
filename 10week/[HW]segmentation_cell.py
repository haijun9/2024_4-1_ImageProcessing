import cv2
import numpy as np
from collections import deque


def get_hist(src):
    hist = np.zeros(256)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            intensity = src[y, x]
            hist[intensity] += 1

    hist /= np.sum(hist)

    return hist

def otsu_method(src):
    p = get_hist(src)
    q1 = np.zeros(256)
    m1 = np.zeros(256)
    m2 = np.zeros(256)

    q1[0] = p[0] + 1e-10
    m1[0] = 0
    m2[0] = np.sum([i * p[i] for i in range(256)]) / (1 - q1[0])
    k = 0
    max_var = q1[0] * (1 - q1[0]) * (m1[0] - m2[0]) ** 2
    for i in range(0, 255):
        q1[i + 1] = q1[i] + p[i + 1]
        m1[i + 1] = (q1[i] * m1[i] + (i + 1) * p[i + 1]) / q1[i + 1]
        m2[i + 1] = ((1 - q1[i]) * m2[i] - (i + 1) * p[i + 1]) / (1 - q1[i + 1])
        between_class_variance = q1[i + 1] * (1 - q1[i + 1]) * (m1[i + 1] - m2[i + 1]) ** 2

        if between_class_variance > max_var:
            max_var = between_class_variance
            k = i

    # k를 이용한 thresholding
    threshold = np.ones_like(src)
    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            if src[y, x] <= k:
                threshold[y, x] = 255

    return threshold

def dilation(B, S):
    height, width = B.shape
    height_S, width_S = S.shape
    dst = np.zeros_like(B)

    whites_y, whites_x = np.where(B == 255)
    for idx in range(len(whites_y)):
        y, x = whites_y[idx], whites_x[idx]

        for i in range(-height_S // 2, height_S // 2 + 1):
            if not (0 <= y + i < height):
                continue
            for j in range(-width_S // 2, width_S // 2 + 1):
                if not (0 <= x + j < width):
                    continue
                dst[y + i, x + j] = 255

    return dst

def erosion(B, S):
    height, width = B.shape
    height_S, width_S = S.shape
    dst = np.zeros_like(B)

    whites_y, whites_x = np.where(B == 255)
    for idx in range(len(whites_y)):
        y, x = whites_y[idx], whites_x[idx]

        flag = True
        for i in range(-height_S // 2, height_S // 2 + 1):
            if not (0 <= y + i < height):
                continue
            for j in range(-width_S // 2, width_S // 2 + 1):
                if not (0 <= x + j < width):
                    continue
                if B[y + i, x + j] != 255:
                    flag = False
                    break

        dst[y, x] = 255 if flag else 0

    return dst

def opening(B, S):
    return dilation(erosion(B, S), S)

def closing(B, S):
    return erosion(dilation(B, S), S)

def main():
    original = cv2.imread('cell.png')
    gray_scale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # B_test = np.array(
    #     [[0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 1, 1, 1, 1, 1, 1, 0],
    #      [0, 0, 0, 1, 1, 1, 1, 0],
    #      [0, 0, 0, 1, 1, 1, 1, 0],
    #      [0, 0, 1, 1, 1, 1, 1, 0],
    #      [0, 0, 0, 1, 1, 1, 1, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0]])
    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    mask = closing(opening(otsu_method(gray_scale), S), S)
    original[mask == 0] -= 127
    cv2.imwrite('result.png', original)

if __name__ == '__main__':
    main()
