import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_out_list(output_list=[], titles=[], figsize=(10, 10)):
  plt.rcParams['figure.figsize'] = figsize
  row = 1
  col = len(output_list)

  for i in range(len(output_list)):
    image_index = i + 1
    plt.subplot(row, col, image_index)
    plt.imshow(output_list[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
  plt.show()

def backward_fit(src, matrix):

    h, w  = src.shape
    src = src.astype(np.float32)
    M_inv = np.linalg.inv(matrix)

    # matrix * [x, y, 1]
    src_dot1 = np.dot(matrix, [0, 0, 1])
    src_dot2 = np.dot(matrix, [w, 0, 1])
    src_dot3 = np.dot(matrix, [0, h, 1])
    src_dot4 = np.dot(matrix, [w, h, 1])

    # src_dotN은 [x, y, 1] 꼴
    y_max = max(0, h, src_dot1[1], src_dot3[1])
    y_min = min(0, h, src_dot1[1], src_dot3[1])
    if M_inv[0][2] >= 0:  # x축 trans 값에 따라
        x_max = max(0, w, src_dot1[0], src_dot4[0])
        x_min = min(0, w, src_dot1[0], src_dot4[0])
    else:
        x_max = max(0, w, src_dot1[0], src_dot2[0])
        x_min = min(0, w, src_dot1[0], src_dot2[0])

    H_prime, W_prime = round(y_max - y_min), round(x_max - x_min)
    dst = np.zeros((H_prime, W_prime))
    for y in range(H_prime):
        for x in range(W_prime):
            src_dot = np.dot(M_inv, [x + x_min, y + y_min, 1])
            src_x, src_y = src_dot[0], src_dot[1]
            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                src_y_floor, src_x_floor = int(src_y), int(src_x)
                dy, dx = src_y - src_y_floor, src_x - src_x_floor
                dst[y, x] = ((1 - dx) * (1 - dy) * src[src_y_floor, src_x_floor]
                             + dx * (1 - dy) * src[src_y_floor, src_x_floor + 1]
                             + (1 - dx) * dy * src[src_y_floor + 1, src_x_floor]
                             + dx * dy * src[src_y_floor + 1, src_x_floor + 1])

    dst = np.clip(np.round(dst), 0, 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # translation
    M1 = np.array([[1, 0, 500],
                  [0, 1, 600],
                  [0, 0, 1]])
    M2 = np.array([[1, 0, -500],
                  [0, 1, 600],
                  [0, 0, 1]])
    M3 = np.array([[1, 0, 500],
                  [0, 1, -600],
                  [0, 0, 1]])
    M4 = np.array([[1, 0, -500],
                  [0, 1, -600],
                  [0, 0, 1]])

    final1 = backward_fit(src, M1)
    final2 = backward_fit(src, M2)
    final3 = backward_fit(src, M3)
    final4 = backward_fit(src, M4)

    plot_out_list([src, final1, final2, final3, final4], ['Original', 'final1', 'final2', 'final3', 'final4'], figsize=(15, 15))

    cv2.waitKey()
    cv2.destroyAllWindows()