import cv2
import numpy as np

img = cv2.imread('/Users/yaodongyu/Coding/chess/My/data/r_can/0.jpg')
img = cv2.resize(img, (96, 96))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

enhanced = np.zeros(img.shape, dtype=np.uint8)
cnt = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (hsv[i, j, 0] > 170 or hsv[i, j, 0] < 10) and hsv[i, j, 1] > 100 and hsv[i, j, 2] > 40:
            enhanced[i, j] = [0, 0, 255]
            cnt += 1
        else:
            enhanced[i, j] = [255, 255, 255]
print(cnt/(img.shape[0]*img.shape[1]))
cv2.imwrite('original.jpg', img)
cv2.imwrite('enhanced.jpg', enhanced)