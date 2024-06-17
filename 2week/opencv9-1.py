import cv2
import numpy as np

### tensorflow
src1 = cv2.imread('./Lena.png')
src2 = cv2.imread('./cat.png')
src3 = cv2.imread('./fruits.png')
print(f'tf 1::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# batch dimension 생성 (batch, height, width, channel)
src1 = np.expand_dims(src1, axis=0)
src2 = np.expand_dims(src2, axis=0)
src3 = np.expand_dims(src3, axis=0)
print(f'tf 2::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# batch dim으로 연결 (batch, height, width, channel)
tf_batch = np.concatenate([src1, src2, src3], axis=0)
print(f'tf 3::batch.shape: {tf_batch.shape}\n')

cv2.imshow('Lena', tf_batch[0])
cv2.imshow('Cat', tf_batch[1])
cv2.imshow('Fruits', tf_batch[2])

cv2.waitKey()
cv2.destroyAllWindows()
