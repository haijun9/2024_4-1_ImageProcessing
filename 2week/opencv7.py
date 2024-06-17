import cv2

src = cv2.imread('./Lena.png')
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

cv2.imshow('original', src)
cv2.imshow('RGB', rgb)
cv2.imshow('GRAY', gray)

print(f'[BGR]: {src[0, 0]}')
print(f'[RGB]: {rgb[0, 0]}')
print(f'[GRAY]: {gray[0, 0]}')

cv2.waitKey()
cv2.destroyAllWindows()
