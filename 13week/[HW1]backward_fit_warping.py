import cv2
import numpy as np
import matplotlib.pyplot as plt

def backward_fit(src, matrix):

    h, w  = src.shape
    src = src.astype(np.float32)
    # matrix 역행렬 구하기
    M_inv = np.linalg.inv(matrix)

    # matrix * [x, y, 1]
    src_dot1 = np.dot(matrix, [0, 0, 1])
    src_dot2 = np.dot(matrix, [w, 0, 1])
    src_dot3 = np.dot(matrix, [0, h, 1])
    src_dot4 = np.dot(matrix, [w, h, 1])

    # src_dotN은 [x, y, 1] 꼴
    y_max = max(src_dot1[1], src_dot2[1], src_dot3[1], src_dot4[1])
    y_min = min(src_dot1[1], src_dot2[1], src_dot3[1], src_dot4[1])
    x_max = max(src_dot1[0], src_dot2[0], src_dot3[0], src_dot4[0])
    x_min = min(src_dot1[0], src_dot2[0], src_dot3[0], src_dot4[0])

    H_prime, W_prime = round(y_max - y_min), round(x_max - x_min)
    dst = np.zeros((H_prime, W_prime))
    for y in range(H_prime):
        for x in range(W_prime):
            src_dot = np.dot(M_inv, [x + x_min, y + y_min, 1])
            src_x, src_y = src_dot[0], src_dot[1]
            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                src_y_floor, src_x_floor = int(src_y), int(src_x)
                dy, dx = src_y - src_y_floor, src_x - src_x_floor
                dst[y, x] = ((1 - dx) * (1 - dy) * src[src_y_floor, src_x_floor]
                             + dx * (1 - dy) * src[src_y_floor, src_x_floor + 1]
                             + (1 - dx) * dy * src[src_y_floor + 1, src_x_floor]
                             + dx * dy * src[src_y_floor + 1, src_x_floor + 1])

    dst = np.clip(np.round(dst), 0, 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # Rotation 20 -> shearing -> scaling
    M_ro = np.array([[np.cos(np.pi / 9), -np.sin(np.pi / 9), 0],
                     [np.sin(np.pi / 9), np.cos(np.pi / 9), 0],
                     [0, 0, 1]])
    M_sh = np.array([[1, 0.2, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    M_sc = np.array([[1.2, 0, 0],
                     [0, 1.2, 0],
                     [0, 0, 1]])
    matrix = np.dot(np.dot(M_ro, M_sh), M_sc)

    final = backward_fit(src, matrix)

    cv2.imshow('lena_gray', src)
    cv2.imshow('final', final)
    # cv2.imwrite('[HW1]final.png', final)

    cv2.waitKey()
    cv2.destroyAllWindows()