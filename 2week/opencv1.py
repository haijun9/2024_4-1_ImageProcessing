import cv2
import numpy as np

src = np.full((2, 2), 100, dtype=np.uint8)
print('<original>')
print(src)
print('<add>')
print(cv2.add(src, 3))
print(cv2.add(src, 200))
print('<subtract>')
print(cv2.subtract(src, 10))
print(cv2.subtract(src, 150))
print('<multiply>')
print(cv2.multiply(src, 2))
print(cv2.multiply(src, 5))
print('<divide>')
print(cv2.divide(src, 10))
print(cv2.divide(src, 200))