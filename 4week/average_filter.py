import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # average filter 생성
    kernel = np.ones((3, 3))
    kernel = kernel / np.sum(kernel)
    print(np.sum(kernel))

    dst1 = cv2.filter2D(src, -1, kernel)
    dst2 = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    dst3 = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    cv2.imshow('original', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst3', dst3)

    cv2.waitKey()
    cv2.destroyAllWindows()