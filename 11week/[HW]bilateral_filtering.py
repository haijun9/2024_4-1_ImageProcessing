import numpy as np
import cv2
import time


def convert_uint8(src):
    return np.round((((src - src.min()) / (src.max() - src.min())) * 255)).astype(np.uint8)

def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=np.float32)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]
    return pad_img

def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):
    dst = src / 255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return dst

def my_bilateral_with_patch(src, fsize, bsize, sigma_xy, sigma_r, pad_type='zero'):
    (h, w) = src.shape
    half_fsize, half_bsize = fsize // 2, bsize // 2
    padded_src = my_padding(src, (half_bsize, half_bsize), pad_type)
    dst = np.zeros_like(src, dtype=np.float32)

    # Gaussian 영역은 미리 계산 (시그마 x = 시그마 y)
    y, x = np.mgrid[-half_fsize:half_fsize + 1, -half_fsize:half_fsize + 1]
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma_xy**2))

    for i in range(h):
        print('\r%d / %d ...' % (i, h), end="")
        for j in range(w):
            # 본인 패치
            my_patch = padded_src[i + (half_bsize - half_fsize): i + (half_bsize + half_fsize) + 1,
                       j + (half_bsize - half_fsize):j + (half_bsize + half_fsize) + 1]

            min_ssd = float('inf')  # 가장 작은 ssd
            similar_patch = None    # 유사 패치
            for k in range(bsize - fsize):
                for l in range(bsize - fsize):
                    near_patch = padded_src[i+k:i+fsize+k, j+l:j+fsize+l]  # box 내 인접 패치
                    if k == half_fsize and l == half_fsize:                # 동일 패치 건너 뛰기
                        continue
                    ssd = np.sum((my_patch - near_patch) ** 2)             # ssd 계산
                    if ssd < min_ssd:
                        min_ssd = ssd
                        similar_patch = near_patch

            # bilateral filter F 이용 필터링
            pixel_difference = np.exp(-(my_patch - my_patch[half_fsize, half_fsize])**2 / (2 * sigma_r**2))
            F = gaussian * pixel_difference
            F_result = np.sum(my_patch * F) / np.sum(F)

            # bilateral filter F' 이용 필터링
            similar_pixel_difference = np.exp(-(similar_patch - similar_patch[half_fsize, half_fsize])**2 / (2 * sigma_r**2))
            F_prime = gaussian * similar_pixel_difference
            F_prime_result = np.sum(similar_patch * F_prime) / np.sum(F_prime)

            # 가중치 계산
            w1 = np.exp(-min_ssd / (2 * sigma_r**2))
            w2 = 1 - w1

            # 값 저장
            dst[i, j] = w2 * F_result + w1 * F_prime_result

    dst = my_normalize(dst)
    return dst

if __name__ == '__main__':

    src = cv2.imread('baby.png', cv2.IMREAD_GRAYSCALE)
    np.random.seed(seed=100)

    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)
    start = time.time()
    dst = my_bilateral_with_patch(src_noise, 11, 17, sigma_xy=5, sigma_r=0.2)
    print('\n', time.time() - start)
    cv2.imshow('src', src)
    cv2.imshow('src_noise', convert_uint8(src_noise))
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
