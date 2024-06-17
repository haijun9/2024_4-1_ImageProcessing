import cv2

src = cv2.imread('./Lena.png')

print(f'type(src): {type(src)}')
print(f'src.dtype: {src.dtype}')
print(f'src.shape: {src.shape}')

cv2.imshow('lena', src)
cv2.waitKey()
cv2.destroyAllWindows()
