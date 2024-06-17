import numpy as np
import cv2
from collections import deque
def labeling(B, neighbor):
    height, width = B.shape
    label_image = np.zeros_like(B)

    if neighbor == 8:
        offsets = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]
    else:
        offsets = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if (dx, dy) != (0, 0)]

    label = 1
    for y in range(height):
        for x in range(width):
            if B[y, x] == 255 and label_image[y, x] == 0:
                queue = deque([(y, x)])
                while queue:
                    pop_y, pop_x = queue.popleft()
                    label_image[pop_y, pop_x] = label

                    for offset_y, offset_x in offsets:
                        next_y, next_x = pop_y + offset_y, pop_x + offset_x
                        if 0 <= next_y < height and 0 <= next_x < width and B[next_y, next_x] == 255:
                            if label_image[next_y, next_x] == 0:
                                queue.append((next_y, next_x))
                label += 1

    num_features = label - 1
    return label_image, num_features

def main():
    example_2D = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 0, 1, 0, 0, 1, 1, 1]
    ], np.uint8)

    label_image, num_features = labeling(example_2D * 255, neighbor=8)
    print(example_2D)
    print(label_image)
    print(num_features)

if __name__ == '__main__':
    main()