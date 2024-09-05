import cv2
import numpy as np
import os
import params
from PIL import Image
import matplotlib.pyplot as plt

count = {
    'r_kin': 0,
    'r_man': 0,
    'r_ele': 0,
    'r_hor': 0,
    'r_roo': 0,
    'r_paw': 0,
    'r_can': 0,
    'b_kin': 0,
    'b_man': 0,
    'b_ele': 0,
    'b_hor': 0,
    'b_roo': 0,
    'b_paw': 0,
    'b_can': 0
}

def generate_rotation(roi, kind, num):
    start = count[kind]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) 
    dif = 360/num
    roi = cv2.resize(roi, params.size)
    img = Image.fromarray(roi)

    for i in range(start,start+num):
        temp = img.rotate(i*dif)
        temp.save(f'data/noModify/{kind}/{i}.jpg', quality=100, subsampling=0)
    count[kind] += num

# 读取图像
image = cv2.imread('chess.jpg')
# 检测象棋的圆形
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=31)
circles = np.uint16(np.around(circles))

os.makedirs('data', exist_ok=True)
for kind in params.types:
    os.makedirs(f'data/noModify/{kind}', exist_ok=True)
# 在原图像上标出识别出的圆
cnt = 0
for (x, y, r) in circles[0, :]:
    # print(x, y, r)
    r -= 4
    roi = image[y-r:y+r+1, x-r:x+r+1]
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 > r**2:
                roi[i, j] = 0
    kind = params.mark[cnt]
    generate_rotation(roi, kind, 360//params.duplicates[kind])

    cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
    cnt += 1
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
