import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_sobel_filter():
    # sobel_x = np.array([[-1, 0, 1],
    #                     [-2, 0, 2],
    #                     [-1, 0, 1]])
    # sobel_y = np.array([[-1, -2, -1],
    #                     [0, 0, 0],
    #                     [1, 2, 1]])

    blurring = np.array([[1], [2], [1]])
    derivative_filter = np.array([[-1], [0], [-1]])
    sobel_x = np.dot(blurring, derivative_filter.T)
    sobel_y = np.dot(derivative_filter, blurring.T)

    return sobel_x, sobel_y


def calculate_magnitude(gradient_x, gradient_y):
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return magnitude


def show_images(images, titles):
    fig, axs = plt.subplots(2, 2)

    for image, title, ax in zip(images, titles, axs.flatten()):
        image = image.clip(0, 255)
        ax.imshow(image, cmap='gray')
        ax.set_title(title, y=-0.17, fontsize=18)
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    # image = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # sobel_filter_x, sobel_filter_y = generate_sobel_filter()
    # gradient_x = cv2.filter2D(image, -1, sobel_filter_x)
    # gradient_y = cv2.filter2D(image, -1, sobel_filter_y)
    # magnitude = calculate_magnitude(gradient_x, gradient_y)

    image = cv2.imread('noise_Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    sobel_filter_x, sobel_filter_y = generate_sobel_filter()
    gradient_x = cv2.filter2D(image, -1, sobel_filter_x)
    gradient_y = cv2.filter2D(image, -1, sobel_filter_y)
    magnitude = calculate_magnitude(gradient_x, gradient_y)

    show_images([image, np.abs(gradient_x), magnitude, np.abs(gradient_y)],
                ['Original', 'x-direction derivative', 'Gradient magnitude', 'y-direction derivative'])
