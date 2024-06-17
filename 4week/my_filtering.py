import cv2
import numpy as np
import time
def my_padding(src, pad_size, pad_type='zeros'):
    (h, w) = src.shape
    p_h, p_w = pad_size
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2), dtype=np.uint8)
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        #down
        pad_img[p_h + h:, p_w:p_w + w] = src[h-1, :]
        #left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        #right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1:p_w + w]

    else:
        # else is zero padding
        print('zero padding')

    return pad_img

def my_filtering(src, kernel, pad_type='zeros'):
    (h, w) = src.shape
    (k_h, k_w) = kernel.shape

    # 직접 구현한 my_padding 함수를 이용 (filter의 (높이/너비 - 1) / 2 만큼)
    img_pad = my_padding(src, (k_h // 2, k_w // 2))
    print(f'<img_pad.shape>: {img_pad.shape}')

    dst = np.zeros((h, w))
    time_start = time.time()
    
    # filtering 진행하는 반복문 구현
    for i in range(h):      # row
        for j in range(w):  # column
            dst[i, j] = np.sum(kernel * img_pad[i:i+k_h, j:j+k_w])


    print(f'filtering time: {time.time()-time_start}')

    # dst = ??? # float -> uint8 변환
    dst = (np.clip(dst + 0.5, 0, 255)).astype(np.uint8)  # 반올림

    return dst

if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    
    # average filter 생성
    # kernel = ??? (홀수 제한)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size))
    kernel = kernel / np.sum(kernel)
    print('<kernel>')
    print(kernel)

    # dst = my_filtering(???)
    dst = my_filtering(src, kernel)

    print(f'src.shape: {src.shape}')
    print(f'dst.shape: {dst.shape}')

    cv2.imshow('original', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()