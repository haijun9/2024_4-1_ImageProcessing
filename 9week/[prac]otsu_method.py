import cv2
import numpy as np

def otsu_method_by_inter_class_variance(src):
    height, width = src.shape
    hist = np.zeros(256)

    for y in range(height):
        for x in range(width):
            intensity = src[y, x]
            hist[intensity] += 1
    hist /= height * width

    k, max_var = -1, -1
    for i in range(256):
        q1 = np.sum(hist[:i + 1])
        q2 = 1 - q1
        if q1 == 0 or q2 == 0:
            continue
        m1 = np.sum([j * hist[j] for j in range(i + 1)]) / q1
        m2 = (np.sum([j * hist[j] for j in range(256)]) - m1 * q1) / q2
        inter_class_var = q1 * q2 * (m1 - m2)**2

        if inter_class_var > max_var:
            max_var = inter_class_var
            k = i

    fat = np.zeros((height, width))
    fat[src > k] = 255

    return k, fat


def main():

    src = cv2.imread('meat.png', cv2.IMREAD_GRAYSCALE)
    src_color = cv2.imread('meat.png', cv2.IMREAD_COLOR)

    k, fat = otsu_method_by_inter_class_variance(src)

    h, w = fat.shape
    fat_3ch = np.zeros((h, w, 3), dtype=np.uint8)
    fat_3ch[:, :, 1] = fat

    final_color = cv2.addWeighted(src_color, 1, fat_3ch, 0.5, 0)

    cv2.imshow('src', src)
    cv2.imshow('src_color', src_color)
    cv2.imshow('fat', fat)
    cv2.imshow('fat_3ch ', fat_3ch)
    cv2.imshow('final_color', final_color)

    cv2.waitKey()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    main()