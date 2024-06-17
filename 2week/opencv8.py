import cv2
import numpy as np

src1 = np.zeros((300, 200))
src2 = np.zeros((300, 200), dtype=np.uint8)

src1[:100] = 0
src1[100:200] = 0.5
src1[200:] = 1

src2[:100] = 0
src2[100:200] = 128
src2[200:] = 255

cv2.imshow('src1', src1)
cv2.imshow('src2', src2)

cv2.waitKey()
cv2.destroyAllWindows()
