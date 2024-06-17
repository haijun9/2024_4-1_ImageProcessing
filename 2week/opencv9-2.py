import cv2
import numpy as np

### pytorch
src1 = cv2.imread('./Lena.png')
src2 = cv2.imread('./cat.png')
src3 = cv2.imread('./fruits.png')
print(f'torch 1::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# batch dimension 생성 (batch, height, width, channel)
src1 = np.expand_dims(src1, axis=0)
src2 = np.expand_dims(src2, axis=0)
src3 = np.expand_dims(src3, axis=0)
print(f'torch 2::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# height와 channel 전환 (batch, channel, width, height)
src1 = np.swapaxes(src1, 1, 3)
src2 = np.swapaxes(src2, 1, 3)
src3 = np.swapaxes(src3, 1, 3)
print(f'torch 3::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# height와 width 전환 (batch, channel, height, width)
src1 = np.swapaxes(src1, 2, 3)
src2 = np.swapaxes(src2, 2, 3)
src3 = np.swapaxes(src3, 2, 3)
print(f'torch 4::src1.shape: {src1.shape}, src2.shape: {src2.shape}, src3.shape:{src3.shape}')

# batch dim으로 연결 (batch, channel, height, width)
batch = np.concatenate([src1, src2, src3], axis=0)
print(f'batch.shape: {batch.shape}')

cv2.waitKey()
cv2.destroyAllWindows()
