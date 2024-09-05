import cv2
import numpy as np
import matplotlib.pyplot as plt
import params
import CnnModel
import train
import torch
from torchvision import transforms

def classify(roi, model, device):
    roi = roi.unsqueeze(0)
    roi = roi.to(device)

    with torch.no_grad():
        output = model(roi)
        _, predicted = torch.max(output.data, 1)
    return params.chinese[predicted.item()]

def process(roi, r):
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 > r**2:
                roi[i, j] = 0
    roi = cv2.resize(roi, params.size)
    return roi

def circle_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        # print(f'File {image_path} not found')
        return (None, None)
    img = cv2.resize(img, (804, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=31)
    return img, np.uint16(np.around(circles))

def recognize(img, circles, model, device):
    global cnt, correct
    for i, circle in enumerate(circles[0, :]):
        temp = img
        x, y, r = circle
        r -= 4
        roi = img[y-r:y+r+1, x-r:x+r+1]
        roi = process(roi, r)
        # cv2.imwrite(f'validation/{cnt}.jpg', roi)
        data = train.enhance_color(roi)
        data = train.transf(data)
        class_name = classify(data, model, device)

        true = params.vali[cnt]
        cnt += 1
        # correct.append(class_name == true)
        if class_name == true:
            correct += 1

        roi = transforms.ToPILImage()(data)
        plt.imshow(roi)
        plt.title(class_name)
        plt.show()

def test(model, device, img_num, debug=False):
    batch = []
    for i in range(img_num):
        img_path = f'validation/{i}.jpg'
        img = cv2.imread(img_path)
        data = train.enhance_color(img)
        data = train.transf(data)
        batch.append(data)

    batch = torch.stack(batch)
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch)
        _, predicted = torch.max(output.data, 1)
    correct = (predicted == torch.tensor([map[i] for i in params.vali])).sum().item()
    if debug:
        print(batch.shape, predicted.shape)
        for i, (img, tag) in enumerate(zip(batch, predicted)):
            img = transforms.ToPILImage()(img)
            plt.imshow(img)
            plt.title(params.types[tag.item()])
            plt.show()
            if i+1 > 5: break

    # co = predicted == torch.tensor([map[i] for i in params.vali])
    # print(co)
    # co = list(co.cpu().numpy())
    # print([i!=j for i, j in zip(co, correct)])
    return correct / img_num

plt.rcParams['font.sans-serif'] = ['Heiti TC']
# model loading
model = CnnModel.ConvNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model/chess_9479+.pth', map_location=device))
model.to(device)
model.eval()

# recognition
cnt = 0
correct = 0
# for i in range(1, 11):
#     img_path = f'train/{i}.jpg'
#     img, circles = circle_detection(img_path)
#     if img is not None:
#         recognize(img, circles, model, device)
# print(correct)

map = {piece: i for i, piece in enumerate(params.types)}
acc = test(model, device, 96, debug=False)
print(f'vali accuracy: {acc * 100}%')

# uncomment the following code to test the model visually
# img, circles = circle_detection('test/1.jpg')
# recognize(img, circles, model, device)