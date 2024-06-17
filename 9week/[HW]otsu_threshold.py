import cv2
import numpy as np

def otsu_method_by_inter_class_variance(src, mask):
    height, width = src.shape

    # histogram 생성
    p = np.zeros(256)
    mask_white = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0:
                continue
            intensity = src[y, x]
            p[intensity] += 1
            mask_white += 1
    p /= mask_white

    # moving average 이용 inter-class variance 구하기
    q1 = np.zeros(256)
    m1 = np.zeros(256)
    m2 = np.zeros(256)

    q1[0] = p[0] + 1e-10
    m1[0] = 0
    m2[0] = np.sum([i * p[i] for i in range(256)]) / (1 - q1[0])
    k = 0
    max_var = q1[0] * (1 - q1[0]) * (m1[0] - m2[0])**2
    for i in range(0, 255):
        q1[i + 1] = q1[i] + p[i + 1]
        m1[i + 1] = (q1[i] * m1[i] + (i + 1) * p[i + 1]) / q1[i + 1]
        m2[i + 1] = ((1 - q1[i]) * m2[i] - (i + 1) * p[i + 1]) / (1 - q1[i + 1])
        between_class_variance = q1[i + 1] * (1 - q1[i + 1]) * (m1[i + 1] - m2[i + 1])**2

        if between_class_variance > max_var:
            max_var = between_class_variance
            k = i

    # k를 이용한 thresholding
    fat = np.zeros((height, width))
    fat_white = 0
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0:
                continue
            if src[y, x] > k:
                fat[y, x] = 255
                fat_white += 1

    # Fat의 하얀색 영역 / Mask의 하얀색 영역
    fat_ratio = fat_white / mask_white
    print(f'등심 영역중 지방의 비율: {fat_ratio}')

    return k, fat


def main():
    meat = cv2.imread('meat.png', cv2.IMREAD_COLOR)
    src = cv2.cvtColor(meat, cv2.COLOR_BGR2GRAY)
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

    # otsu's method 적용
    k, fat = otsu_method_by_inter_class_variance(src, mask)
    print(f'threshold: {k}')

    fat_3ch = np.zeros((fat.shape[0], fat.shape[1], 3), dtype=np.uint8)
    fat_3ch[:, :, 1] = fat

    # 원본 이미지에 dst 적용하기
    final = cv2.addWeighted(meat, 1, fat_3ch, 0.5, 0)

    cv2.imshow('meat', meat)
    cv2.imshow('fat_area', fat)
    cv2.imshow('fat_area_3ch', fat_3ch)
    cv2.imshow('final', final)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 보고서 첨부용
    cv2.imwrite('fat_area.png', fat)
    cv2.imwrite('final.png', final)

    return


if __name__ == '__main__':
    main()