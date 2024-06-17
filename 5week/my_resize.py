import cv2
import numpy as np


def match_up_coordinates(old_shape, new_shape):
    a_y = (old_shape[0] - 1) / (new_shape[0] - 1)  # y축 변화 비율
    a_x = (old_shape[1] - 1) / (new_shape[1] - 1)  # x축 변화 비율
    b_y = 0
    b_x = 0
    return (a_y, a_x, b_y, b_x)


def my_resize(old_img, new_shape):
    # 빈 배열 만들기
    new_img = np.zeros(new_shape, dtype=np.uint8)
    old_shape = old_img.shape  # old_shape로 간소화

    # match_up_coordinates
    a_y, a_x, b_y, b_x = match_up_coordinates(old_shape, new_img.shape)

    # new_img의 모든 픽셀 값 채워넣기
    for row in range(new_shape[0]):
        for col in range(new_shape[1]):
            # y, x 좌표 매칭
            y = row * a_y + b_y  # a_y는 비율
            x = col * a_x + b_x  # a_x는 비율

            # ⌊𝑦⌋, ⌊𝑥⌋, ⌊𝑦+1⌋, ⌊𝑥+1⌋ 구하기
            y_floor = int(y)
            x_floor = int(x)
            y_ceil = min(y_floor + 1, old_shape[0] - 1)  # old_shape[0]이 높이
            x_ceil = min(x_floor + 1, old_shape[1] - 1)  # old_shape[1]이 너비

            # binary interpolation을 통해 픽셀 값 구하기
            t = y - y_floor  # y값 가중치
            s = x - x_floor  # x값 가중치

            intensity = ((1 - s) * (1 - t) * old_img[y_floor, x_floor]
                         + s * (1 - t) * old_img[y_floor, x_ceil]
                         + (1 - s) * t * old_img[y_ceil, x_floor]
                         + s * t * old_img[y_ceil, x_ceil])

            # 반올림하여 정수 값 만들기
            new_img[row, col] = round(intensity)

    return new_img


def main():
    old_img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    new_img_1000by1024 = my_resize(old_img, new_shape=(1000, 1024))
    new_img_256by200 = my_resize(old_img, new_shape=(256, 200))

    cv2.imwrite('new_img_1000by1024.png', new_img_1000by1024)
    cv2.imwrite('new_img_256by200.png', new_img_256by200)

    return


if __name__ == '__main__':
    main()
