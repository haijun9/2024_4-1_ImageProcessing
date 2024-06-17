import cv2
import numpy as np

def my_single_thresholding(src, threshold):
    # src > threshold를 만족하는 pixel을 255로 수정
    dst = np.zeros(src.shape, dtype=np.uint8)
    dst[src > threshold] = 255

    return dst

def my_double_thresholding(src, low_threshold, high_threshold):
    # low_threshold < src < high_threshold를 만족하는 pixel을 255로 수정
    dst = np.zeros(src.shape, dtype=np.uint8)
    dst[(low_threshold < src) & (src < high_threshold)] = 255

    return dst

def main():
    # OpenCV를 이용하여 threshold_test.png를 grayscale로 불러오기
    src = cv2.imread('threshold_test.png', cv2.IMREAD_GRAYSCALE)

    # my_single_thresholding 함수를 이용하여 single thresholding
    dst1 = my_single_thresholding(src, 150)

    # my_double_thresholding 함수를 이용하여 double thresholding
    dst2 = my_double_thresholding(src,100, 200)

    # OpenCV를 이용하여 thresholding img 저장
    cv2.imshow('original', src)
    cv2.imshow('dst1', dst1)
    cv2.imshow('dst2', dst2)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
