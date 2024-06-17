import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    kernel1 = np.zeros((3, 3))
    kernel1[1, 1] = 2
    dst1 = cv2.filter2D(src, cv2.CV_32F, kernel1)

    kernel2 = np.ones((3, 3))
    kernel2 = kernel2 / np.sum(kernel2)
    dst2 = cv2.filter2D(src, cv2.CV_32F, kernel2)

    # case 1
    dst3 = dst1 - dst2
    dst3 = (np.clip(src + 0.5, 0, 255)).astype(np.uint8) # 반올림

    # case 2
    dst4 = cv2.filter2D(src, -1, kernel1 - kernel2)
    dst4 = (np.clip(src + 0.5, 0, 255)).astype(np.uint8) # 반올림

    print(dst3 == dst4)

    cv2.imshow('original', src)
    cv2.imshow('dst3', dst3)
    cv2.imshow('dst4', dst4)

    cv2.waitKey()
    cv2.destroyAllWindows()