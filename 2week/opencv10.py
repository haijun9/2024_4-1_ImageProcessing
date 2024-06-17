import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('./Lena.png')
    dst1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # dst2 = (B+G+R)/3
    dst2 = (src[:, :, 0] * 1/3 +
            src[:, :, 1] * 1/3 +
            src[:, :, 2] * 1/3)
    # dst3 = 0.0721*B + 0.7154*G + 0.2125*R
    dst3 = (src[:, :, 0] * 0.0721 +
            src[:, :, 1] * 0.7154 +
            src[:, :, 2] * 0.2125)

    dst2 = np.round(dst2).astype(np.uint8)
    dst3 = np.round(dst3).astype(np.uint8)

    cv2.imshow('Original', src)
    cv2.imshow('Gray (cvtColor)', dst1)
    cv2.imshow('Gray (1/3)', dst2)
    cv2.imshow('Gray (formula)', dst3)

    cv2.waitKey()
    cv2.destroyAllWindows()