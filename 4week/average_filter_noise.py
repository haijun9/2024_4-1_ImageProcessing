import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('Lena_noise.png', cv2.IMREAD_GRAYSCALE)

    # average filter 생성
    kernel1 = np.ones((3, 3))
    kernel1 = kernel1 / np.sum(kernel1)
    dst1 = cv2.filter2D(src, -1, kernel1,
                        borderType=cv2.BORDER_CONSTANT)

    kernel2 = np.ones((9, 9))
    kernel2 = kernel2 / np.sum(kernel2)
    dst2 = cv2.filter2D(src, -1, kernel2,
                        borderType=cv2.BORDER_CONSTANT)

    kernel3 = np.ones((15, 15))
    kernel3 = kernel3 / np.sum(kernel3)
    dst3 = cv2.filter2D(src, -1, kernel3,
                        borderType=cv2.BORDER_CONSTANT)

    cv2.imshow('original', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst3', dst3)

    cv2.waitKey()
    cv2.destroyAllWindows()