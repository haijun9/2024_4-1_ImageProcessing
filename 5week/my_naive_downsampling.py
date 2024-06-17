import cv2
import numpy as np

def my_naive_downsampling(old_img, ratio_y, ratio_x):
    # old_img의 height와 width
    h_old, w_old = old_img.shape

    # new_img의 height와 width
    h_new = h_old // ratio_y
    w_new = w_old // ratio_x

    # 빈 new_img 선언
    new_img = np.zeros((h_new, w_new), np.uint8)

    # new_img 픽셀 값 채워넣기
    for row in range(h_new):
        for col in range(w_new):
            new_img[row, col] = old_img[row*ratio_y, col*ratio_x]

    return new_img

def main():
    # OpenCV를 이용하여 Lena.png를 grayscale로 불러오기
    old_img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # my_naive_downsampling 함수를 이용하여 old_img를 downsampling
    new_img = my_naive_downsampling(old_img, 2, 2)

    # OpenCV를 이용하여 new_img 저장
    cv2.imshow('old_img', old_img)

    cv2.imshow('new_img', new_img)
    cv2.imwrite('new_img.png', new_img)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()