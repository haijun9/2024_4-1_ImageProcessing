import numpy as np
import cv2
import pandas as pd


def Quantization_matrix(scale=1, n=8):
    if n == 8:
        quantization_matrix = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]])
    return quantization_matrix * scale


def img2blocks(input_img, n=8):
    blocks = []
    height, width = input_img.shape
    for y in range(0, height, n):
        for x in range(0, width, n):
            blocks.append(input_img[y:y + n, x:x + n])
    return np.array(blocks)


def blocks2img(blocks, img_shape, n=8):
    height, width = img_shape
    decompressed_img = np.zeros(img_shape, dtype=np.uint8)
    index = 0
    for y in range(0, height, n):
        for x in range(0, width, n):
            decompressed_img[y:y + n, x:x + n] = blocks[index]
            index += 1
    return decompressed_img


def C(w, n=8):
    return np.sqrt(1 / n) if w == 0 else np.sqrt(2 / n)


def DCT(block, n=8):
    DCT_result = np.zeros((n, n), dtype=np.float32)
    for v in range(n):
        for u in range(n):
            F = 0
            for y in range(n):
                for x in range(n):
                    F += (block[y, x]
                          * np.cos((2 * y + 1) * v * np.pi / (2 * n))
                          * np.cos((2 * x + 1) * u * np.pi / (2 * n)))
            DCT_result[v, u] = C(v) * C(u) * F
    return DCT_result


def IDCT(inverse_quantization_result, n=8):
    IDCT_result = np.zeros((n, n), dtype=np.float32)
    for y in range(n):
        for x in range(n):
            f = 0
            for v in range(n):
                for u in range(n):
                    f += (C(v) * C(u) * inverse_quantization_result[v, u]
                          * np.cos((2 * y + 1) * v * np.pi / (2 * n))
                          * np.cos((2 * x + 1) * u * np.pi / (2 * n)))
            IDCT_result[y, x] = f
    return np.round(IDCT_result).astype(np.int64)


def zigzag_scanning(quantization_result):
    n = quantization_result.shape[0]
    scanning_result = []
    i, j = 0, 0
    goUp = True # 위로 가는 경우
    EOB_index, prev_value = None, None  # EOB 기록 위치
    while i < n and j < n:
        value = quantization_result[i, j]
        if prev_value != 0 and value == 0:
            EOB_index = len(scanning_result)
        scanning_result.append(value)
        prev_value = value

        if goUp:
            if j == n - 1:  # 꼭짓점 도달
                i += 1
                goUp = False
            elif i == 0:    # 모서리 이동
                j += 1
                goUp = False
            else:           # 이 외 이동
                i -= 1
                j += 1
        else:
            if i == n - 1:  # 꼭짓점 도달
                j += 1
                goUp = True
            elif j == 0:    # 모서리 이동
                i += 1
                goUp = True
            else:           # 이 외 이동
                i += 1
                j -= 1

    scanning_result[EOB_index:] = ["EOB"]
    return scanning_result


def inverse_zigzag_scanning(compressed_block, n=8):
    inverse_scanning_result = np.zeros((n, n), dtype=np.int64)
    index = 0
    i, j = 0, 0
    goUp = True # 위로 가는 경우
    while i < n and j < n:
        value = compressed_block[index]
        if value == "EOB":
            break

        inverse_scanning_result[i, j] = value
        index += 1
        if goUp:
            if j == n - 1:  # 꼭짓점 도달
                i += 1
                goUp = False
            elif i == 0:    # 모서리 이동
                j += 1
                goUp = False
            else:           # 이 외 이동
                i -= 1
                j += 1
        else:
            if i == n - 1:  # 꼭짓점 도달
                j += 1
                goUp = True
            elif j == 0:    # 모서리 이동
                i += 1
                goUp = True
            else:           # 이 외 이동
                i += 1
                j -= 1
    return inverse_scanning_result


def residual(compressed_blocks, img_shape, n=8):
    def sub_block(current_block, other_block):  # 두 블록 간 원소를 뺀 값으로 구성된 블록을 반환
        result_block = []
        current_len, other_len = len(current_block), len(other_block)
        for index in range(current_len):
            if index >= other_len or current_block[index] == "EOB" or other_block[index] == "EOB":
                result_block.append(current_block[index])
            else:
                result_block.append(current_block[index] - other_block[index])
        return result_block

    height, width = img_shape
    compressed_img = np.zeros(img_shape, dtype=object)
    block_idx = 0
    for y in range(0, height, n):
        for x in range(0, width, n):
            if y == 0 and x == 0:   # 시작 (0, 0) 일 때
                compressed_img[y, x] = compressed_blocks[block_idx]
            elif x == 0:            # row의 시작, 왼쪽 블록이 없을 때 위 쪽 블록 이용
                compressed_img[y, x] = sub_block(compressed_blocks[block_idx], compressed_blocks[block_idx - n])
            else:                   # 이 외는 본인의 왼쪽 블록 이용
                compressed_img[y, x] = sub_block(compressed_blocks[block_idx], compressed_blocks[block_idx - 1])
            block_idx += 1

    return compressed_img


def inverse_residual(compressed_img, img_shape, n=8):
    def add_img(current_img, other_img):    # 두 블록 간 원소를 더한 값으로 구성된 블록을 복원
        result_img = []
        current_len, other_len = len(current_img), len(other_img)
        for index in range(current_len):
            if index >= other_len or current_img[index] == "EOB" or other_img[index] == "EOB":
                result_img.append(current_img[index])
            else:
                result_img.append(current_img[index] + other_img[index])
        return result_img

    height, width = img_shape
    inverse_residual_result = []
    for y in range(0, height, n):
        for x in range(0, width, n):
            if y == 0 and x == 0:   # 시작 (0, 0) 일 때
                inverse_residual_result.append(compressed_img[y, x])
            elif x == 0:            # row의 시작, 왼쪽 블록이 없을 때 위 쪽 블록 이용
                inverse_residual_result.append(add_img(compressed_img[y, x], inverse_residual_result[-n]))
            else:                   # 이 외는 본인의 왼쪽 블록 이용
                inverse_residual_result.append(add_img(compressed_img[y, x], inverse_residual_result[-1]))

    return inverse_residual_result


def Encoding(input_img, matrix):
    blocks = img2blocks(input_img)
    compressed_blocks = []
    for block in blocks:
        block = block.astype(np.int64)
        block -= 128
        DCT_block = DCT(block)
        quantized_block = np.round(DCT_block / matrix).astype(np.int64)
        scanned_block = zigzag_scanning(quantized_block)
        compressed_blocks.append(scanned_block)
    compressed_blocks = residual(compressed_blocks, input_img.shape)
    return np.array(compressed_blocks, dtype=object), input_img.shape


def Decoding(compressed_img, img_shape, matrix):
    compressed_img = inverse_residual(compressed_img, img_shape)
    decompressed_blocks = []
    for block in compressed_img:
        inverse_scanned_result = inverse_zigzag_scanning(block)
        inverse_quantized_result = inverse_scanned_result * matrix
        IDCT_result = IDCT(inverse_quantized_result)
        IDCT_result += 128
        IDCT_result = np.clip(IDCT_result, 0, 255)
        decompressed_blocks.append(IDCT_result)
    decompressed_img = blocks2img(np.array(decompressed_blocks), img_shape)
    return decompressed_img

def main():
    # example_block = np.array([
    #     [52, 55, 61, 66, 70, 61, 64, 73],
    #     [63, 59, 66, 90, 109, 85, 69, 72],
    #     [62, 59, 68, 113, 144, 104, 66, 73],
    #     [63, 58, 71, 122, 154, 106, 70, 69],
    #     [67, 61, 68, 104, 126, 88, 68, 70],
    #     [79, 65, 60, 70, 77, 68, 58, 75],
    #     [85, 71, 64, 59, 55, 61, 65, 83],
    #     [87, 79, 69, 68, 65, 76, 78, 94]
    # ], np.int64)

    original_img = cv2.imread('caribou.png', cv2.IMREAD_GRAYSCALE)

    scale = 1

    Q = Quantization_matrix(scale=scale)
    compressed_img, input_img_shape = Encoding(input_img=original_img, matrix=Q)

    np.save('compressed_img', compressed_img)
    np.save('img_shape', input_img_shape)

    compressed = np.load('compressed_img.npy', allow_pickle=True)
    img_shape = np.load('img_shape.npy')

    decompressed_img = Decoding(compressed, img_shape, matrix=Q)

    cv2.imwrite('decompressed_img_' + str(scale) + '.png', decompressed_img)
    cv2.imwrite('difference_' + str(scale) + '.png', (original_img - decompressed_img + 128))

if __name__ == '__main__':
    main()