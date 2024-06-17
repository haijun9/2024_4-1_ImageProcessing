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


def get_DoG_filter_by_expression(fsize, sigma):
    half = fsize // 2
    DoG_x = np.zeros((fsize, fsize), np.float64)
    DoG_y = np.zeros((fsize, fsize), np.float64)

    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            DoG_x[y + half, x + half] = (-x / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            DoG_y[y + half, x + half] = (-y / (2 * np.pi * sigma**4)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
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
    gradient_x = cv2.filter2D(image, -1, sobel_filter_x)
    gradient_y = cv2.filter2D(image, -1, sobel_filter_y)
    sobel_magnitude = calculate_magnitude(gradient_x, gradient_y)

    fsize = 13
    sigma = 3
    DoG_y, DoG_x = get_DoG_filter_by_expression(fsize, sigma)
    DoG_gradient_x = cv2.filter2D(image, -1, DoG_y)
    DoG_gradient_y = cv2.filter2D(image, -1, DoG_x)
    DoG_expression_magnitude = calculate_magnitude(DoG_gradient_x, DoG_gradient_y)
    DoG_expression_magnitude = min_max_scaling(DoG_expression_magnitude)

    merge_image = merge_images([image, sobel_magnitude, DoG_expression_magnitude])
