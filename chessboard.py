import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import params
from PIL import Image
import CnnModel
import torch
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ]
)
def classify(roi, model, device, transform):
    roi = transform(roi)
    roi = roi.unsqueeze(0)
    roi = roi.to(device)

    with torch.no_grad():
        output = model(roi)
        _, predicted = torch.max(output.data, 1)
    return params.types[predicted.item()]

def process(roi, r):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 > r**2:
                roi[i, j] = 0
    return roi

def circle_detection(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (804, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=31)
    return img, np.uint16(np.around(circles))

def recognize(img, circles, model, device, transform):
    for i, circle in enumerate(circles[0, :]):
        temp = img
        x, y, r = circle
        roi = img[y-r:y+r+1, x-r:x+r+1]
        roi = process(roi, r)
        data = Image.fromarray(roi)
        class_name = classify(data, model, device, transform)
        plt.imshow(roi)
        plt.title(class_name)
        plt.show()
        # cv2.circle(temp, (x, y), r, (0, 255, 0), 2)
        # cv2.imshow(class_name, temp)
        # cv2.waitKey(1000)

# model loading
model = CnnModel.ConvNet()
model.load_state_dict(torch.load('model/chess_9821.pth'))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# recognition
img_path = 'train/1.jpg'
img, circles = circle_detection(img_path)
recognize(img, circles, model, device, transform)