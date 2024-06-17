import numpy as np
import cv2
import time

# def my_median_filtering(src, fsize):
#     # np.median() 사용 가능
#     height, width = src.shape
#     pad_size = fsize // 2
#     padded_src = np.pad(src, pad_size, mode='constant', constant_values=0)
#     dst = np.zeros_like(src)
#
#     for y in range(height):
#         for x in range(width):
#             neighborhood = padded_src[y:y + fsize, x:x + fsize]
#             dst[y, x] = np.median(neighborhood)
#
#     return dst

def my_median_filtering(src, fsize):
    # np.median() 사용 가능
    h, w = src.shape
    dst = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            r_start = np.clip(row - (fsize // 2), 0, h)
            r_end = np.clip(row + (fsize // 2), 0, h)

            c_start = np.clip(col - (fsize // 2), 0, h)
            c_end = np.clip(col + (fsize //2), 0, h)
            filter = src[r_start:r_end+1, c_start:c_end+1]

            dst[row, col] = np.median(filter)

    return np.clip(np.round(dst), 0, 255).astype(np.uint8)

def add_snp_noise(src, prob):

    h, w = src.shape

    # np.random.rand = 0 ~ 1 사이의 값이 나옴
    noise_prob = np.random.rand(h, w)
    dst = np.zeros((h, w), dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            if noise_prob[row, col] < prob:
                # pepper noise
                dst[row, col] = 0
            elif noise_prob[row, col] > 1 - prob:
                # salt noise
                dst[row, col] = 255
            else:
                dst[row, col] = src[row, col]

    return dst


def main():

    np.random.seed(seed=100)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 원본 이미지에 노이즈를 추가
    src_snp_noise = add_snp_noise(src, prob=0.05)

    # nxn average filter
    filter_size = 5
    avg_filter = np.ones((filter_size, filter_size)) / (filter_size * filter_size)

    # 평균 필터 적용
    average_start_time = time.time()
    dst_avg = cv2.filter2D(src_snp_noise, -1, avg_filter)
    dst_avg = dst_avg.astype(np.uint8)
    print('average filtering time : ', time.time() - average_start_time)

    # median filter 적용
    median_start_time = time.time()
    dst_median = my_median_filtering(src_snp_noise, 5)
    print('median filtering time : ', time.time() - median_start_time)

    cv2.imshow('original', src)
    cv2.imshow('Salt and pepper noise', src_snp_noise)
    cv2.imshow('noise removal(average fileter)', dst_avg)
    cv2.imshow('noise removal(median filter)', dst_median)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
