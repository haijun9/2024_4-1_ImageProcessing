import cv2

src = cv2.imread('logo.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

print(f'[color shape]: {src.shape}')
print(f'[gray shape]: {gray.shape}')

cv2.imshow('color', src)
cv2.imshow('gray', gray)
cv2.imshow('slice', src[50:230, 50:230, :])

cv2.waitKey()
cv2.destroyAllWindows()