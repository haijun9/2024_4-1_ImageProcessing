import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # average filter 생성
    kernel1 = np.ones((5, 5))
    kernel1 = kernel1 / 25  # 총합 1
    print(np.sum(kernel1))
    dst1 = cv2.filter2D(src, -1, kernel1,
                        borderType=cv2.BORDER_REPLICATE)

    kernel2 = np.ones((5, 5))
    kernel2 = kernel2 / 40  # 총합 1 미만
    print(np.sum(kernel2))
    dst2 = cv2.filter2D(src, -1, kernel2,
                        borderType=cv2.BORDER_REPLICATE)

    kernel3 = np.ones((5, 5))
    kernel3 = kernel3 / 10  # 총합 1 초과
    print(np.sum(kernel3))
    dst3 = cv2.filter2D(src, -1, kernel3,
                        borderType=cv2.BORDER_REPLICATE)

    cv2.imshow('original', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst3', dst3)

    cv2.waitKey()
    cv2.destroyAllWindows()