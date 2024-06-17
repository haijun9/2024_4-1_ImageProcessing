from collections import deque

import cv2
import numpy as np


def min_max_scaling(src):
    return 255 * ((src - src.min()) / (src.max() - src.min()))


def get_DoG_filter(fsize, sigma):
    DoG_x = np.zeros((fsize, fsize), np.float64)
    DoG_y = np.zeros((fsize, fsize), np.float64)
    half = fsize // 2
    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            DoG_x[y + half, x + half] = (-x / (2 * np.pi * sigma ** 4)) * np.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))
            DoG_y[y + half, x + half] = (-y / (2 * np.pi * sigma ** 4)) * np.exp(
                -((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    return DoG_x, DoG_y


def calculate_magnitude(gradient_x, gradient_y):
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    return magnitude


def non_maximum_suppression(gradient_x, gradient_y, magnitude, n):
    height, width = magnitude.shape
    large_magnitude = np.zeros((height, width))
    half_n = n // 2

    for y in range(half_n, height - half_n):
        for x in range(half_n, width - half_n):
            if gradient_x[y, x] == 0:  # case 1 : gradient_x = 0
                # 최대 값이 아닌 경우 0으로 설정
                large_magnitude[y, x] = magnitude[y, x] if magnitude[y, x] == np.max(magnitude[y - half_n:y + half_n, x]) else 0

            else:
                a = np.abs(gradient_y[y, x] / gradient_x[y, x]) # 해당 픽셀 기울기
                large_magnitude[y, x] = magnitude[y, x]         # 초기값 설정

                if a < 1:  # case 2 : |a| < 1
                    for i in range(-half_n, half_n + 1):
                        y_floor = int(np.floor(y + a * i))
                        t = np.abs(y_floor + 1 - y) # 가중치 계산
                        point = magnitude[y_floor, x + i] * t + (1 - t) * magnitude[y_floor + 1, x + i] # linear interpolation
                        if magnitude[y, x] < point: # 최대 값이 아닌 경우 의미 없는 값이자 연산임
                            large_magnitude[y, x] = 0
                            break

                else:  # case 3 : |a| >= 1
                    for i in range(-half_n, half_n + 1):
                        x_floor = int(np.floor(x + i / a))
                        t = np.abs(x_floor + 1 - x) # 가중치 계산
                        point = magnitude[y + i, x_floor] * t + (1 - t) * magnitude[y + i, x_floor + 1] # linear interpolation
                        if magnitude[y, x] < point: # 최대 값이 아닌 경우 의미 없는 값이자 연산임
                            large_magnitude[y, x] = 0
                            break

    return large_magnitude


def double_thresholding(nms_result, high_threshold, low_threshold):
    height, width = nms_result.shape
    thresholding_result = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            if nms_result[y, x] > high_threshold:  # strong edge 판별
                thresholding_result[y, x] = 255
            elif nms_result[y, x] < low_threshold: # not edge 판별
                thresholding_result[y, x] = 0
            else:                                  # week edge 판별
                thresholding_result[y, x] = 128

    return thresholding_result


def determine_edge(thresholding_result):
    rows, cols = thresholding_result.shape
    canny_edge_result = np.zeros((rows, cols))          # 반환할 결과
    visited = np.zeros((rows, cols), dtype=bool)  # 방문한 픽셀 기록

    def bfs(start_y, start_x):
        connect = deque([(start_y, start_x)]) # 연결된 week edge 저장
        visited[start_y, start_x] = True  # 시작 픽셀을 방문 표시
        while connect:
            pop_y, pop_x = connect.popleft()
            for offset_y, offset_x in [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]:
                check_y, check_x = pop_y + offset_y, pop_x + offset_x # offset 따른 탐색 위치 지정
                if (0 <= check_y < rows) and (0 <= check_x < cols) and not visited[check_y, check_x]:
                    if thresholding_result[check_y, check_x] == 255: # strong edge 경우
                        return True # 바로 True 리턴, 종료
                    elif thresholding_result[check_y, check_x] == 128: # week edge 경우
                        visited[check_y, check_x] = True   # 방문 배열에 추가
                        connect.append((check_y, check_x)) # connect 큐에 추가
        return False

    for y in range(rows):
        for x in range(cols):
            if thresholding_result[y, x] == 128:
                canny_edge_result[visited] = 255 if bfs(y, x) else 0 # bfs 결과에 따라 edge 판별
                visited = np.zeros((rows, cols), dtype=bool)  # 방문 배열 초기화
            elif thresholding_result[y, x] == 255:
                canny_edge_result[y, x] = 255   # strong edge 살리기

    return canny_edge_result


def main():
    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    DoG_x, DoG_y = get_DoG_filter(fsize=5, sigma=1)
    gradient_y = cv2.filter2D(image, -1, DoG_y)
    gradient_x = cv2.filter2D(image, -1, DoG_x)
    magnitude = calculate_magnitude(gradient_x=gradient_x, gradient_y=gradient_y)
    nms_result = non_maximum_suppression(gradient_x=gradient_x, gradient_y=gradient_y, magnitude=magnitude, n=5)
    thresholding_result = double_thresholding(nms_result=nms_result, high_threshold=10, low_threshold=4)
    canny_edge_result = determine_edge(thresholding_result=thresholding_result)

    cv2.imwrite('magnitude.png', min_max_scaling(magnitude))
    cv2.imwrite('nms.png', min_max_scaling(nms_result))
    cv2.imwrite('thresholding.png', thresholding_result)
    cv2.imwrite('canny_edge.png', canny_edge_result)


if __name__ == '__main__':
    main()