import cv2
import numpy as np

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # average filter 생성
    kernel = np.ones((7, 7))
    kernel = kernel / np.sum(kernel)
    print(np.sum(kernel))

    dst1 = cv2.filter2D(src, -1, kernel)
    dst2 = cv2.filter2D(src, cv2.CV_32F, kernel) # Float으로 변환

    dst3 = cv2.filter2D(dst1, -1, kernel)
    dst4 = cv2.filter2D(dst2, -1, kernel)

    dst3 = np.clip(dst3 + 0.5, 0, 255).astype(np.uint8)
    dst4 = np.clip(dst4 + 0.5, 0, 255).astype(np.uint8)

    print('dst3:', dst3[-2, :10])
    print('dst4:', dst4[-2, :10])

    cv2.imshow('original', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)
    cv2.imshow('dst3', dst3)
    cv2.imshow('dst4', dst4)

    cv2.waitKey()
    cv2.destroyAllWindows()