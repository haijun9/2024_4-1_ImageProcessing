import cv2
import numpy as np
import matplotlib.pyplot as plt


def min_max_scaling(src):
    return 255 * ((src - src.min()) / (src.max() - src.min()))


def generate_sobel_filter_mine():
    blurring = np.array([[1], [2], [1]])
    derivative_filter = np.array([[-1], [0], [1]])
    sobel_x = np.dot(blurring, derivative_filter.T)
    sobel_y = np.dot(derivative_filter, blurring.T)
    return sobel_x, sobel_y


def get_Gaussian2D_kernel(fsize, sigma=1):
    gaus2D = np.zeros((fsize, fsize), np.float64)
    half = fsize // 2
    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            gaus2D[y+half, x+half] = 1 / (2 * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2)/(2 * sigma ** 2)))
    return gaus2D


def get_DoG_filter_by_filtering(fsize, sigma):
    gaus2D = get_Gaussian2D_kernel(fsize + 2, sigma)
    derivative_filter = np.array([[-1], [0], [1]])

    DoG_y_filter = np.zeros((fsize, fsize))
    for row in range(fsize):
        for col in range(fsize - 2):
            DoG_y_filter[row, col] = np.sum(derivative_filter * gaus2D[row:row+3, col:col+3])

    DoG_x_filter = np.zeros((fsize, fsize))
    for row in range(fsize - 2):
        for col in range(fsize):
            DoG_x_filter[row, col] = np.sum(gaus2D[row:row+3, col:col+3] * derivative_filter.T)

    return DoG_y_filter, DoG_x_filter


def get_DoG_filter_by_expression(fsize, sigma):
    half = fsize // 2
    DoG_x = np.zeros((fsize, fsize), np.float64)
    DoG_y = np.zeros((fsize, fsize), np.float64)

    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            DoG_x[y + half, x + half] = (-x / (2 * np.pi * sigma ** 4)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            DoG_y[y + half, x + half] = (-y / (2 * np.pi * sigma ** 4)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return DoG_y, DoG_x

    return DoG_y, DoG_x


def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude


def merge_images(images):
    if len(images) == 3:
        titles = ['Original', 'Sobel', 'DoG (Equation)']
        fig, axs = plt.subplots(1, 3)
    elif len(images) == 4:
        titles = ['Original', 'Sobel', 'DoG (Equation)', 'DoG (Filtering)']
        fig, axs = plt.subplots(2, 2)

    for image, title, ax in zip(images, titles, axs.flatten()):
        image = image.clip(0, 255)
        ax.imshow(image, cmap='gray')
        ax.set_title(title, y=-0.17, fontsize=18)
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sobel_filter_x, sobel_filter_y = generate_sobel_filter_mine()
    sobel_gradient_x = cv2.filter2D(image, -1, sobel_filter_x)
    sobel_gradient_y = cv2.filter2D(image, -1, sobel_filter_y)
    sobel_magnitude = calculate_magnitude(sobel_gradient_x, sobel_gradient_y)

    f_size = 13
    sigma = 3

    DoG_y, DoG_x = get_DoG_filter_by_expression(f_size, sigma)
    DoG_gradient_x = cv2.filter2D(image, -1, DoG_y)
    DoG_gradient_y = cv2.filter2D(image, -1, DoG_x)
    DoG_expression_magnitude = calculate_magnitude(DoG_gradient_x, DoG_gradient_y)
    DoG_expression_magnitude = min_max_scaling(DoG_expression_magnitude)

    filtered_DoG_y, filtered_DoG_x = get_DoG_filter_by_filtering(f_size, sigma)
    DoG_gradient_filtered_x = cv2.filter2D(image, -1, filtered_DoG_x)
    DoG_gradient_filtered_y = cv2.filter2D(image, -1, filtered_DoG_y)
    DoG_filtering_magnitude = calculate_magnitude(DoG_gradient_filtered_x, DoG_gradient_filtered_y)
    DoG_filtering_magnitude = min_max_scaling(DoG_filtering_magnitude)

    cv2.imwrite('sobel_magnitude.png', sobel_magnitude)
    cv2.imwrite('DoG_expression_magnitude.png', DoG_expression_magnitude)
    cv2.imwrite('DoG_filtering_magnitude.png', DoG_filtering_magnitude)

    merge_image = merge_images([image, sobel_magnitude, DoG_expression_magnitude, DoG_filtering_magnitude])
