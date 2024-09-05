import torch.nn as nn
import torch
from torch.utils import data
from PIL import Image
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import os
import params
import CnnModel

def enhance_color(roi, thresh=0.1):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    enhanced = np.zeros(roi.shape, dtype=np.uint8)
    red_cnt = 0

    r = int(roi.shape[0] / 2)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if (i-r)**2 + (j-r)**2 > (r-1)**2:
                enhanced[i, j] = [255, 255, 255]
            else:
                if (hsv[i, j, 0] > 172 or hsv[i, j, 0] < 8) and hsv[i, j, 1] > 100 and hsv[i, j, 2] > 50:
                    enhanced[i, j] = [255, 0, 0]
                    red_cnt += 1
                elif hsv[i, j, 2] < 50:
                    enhanced[i, j] = [0, 0, 0]
                else:
                    enhanced[i, j] = [255, 255, 255]
    if red_cnt / (roi.shape[0] * roi.shape[1]) > thresh:
        mask = (enhanced == [0, 0, 0]).all(axis=2)
        enhanced[mask] = [255, 0, 0]
    else:
        mask = (enhanced == [255, 0, 0]).all(axis=2)
        enhanced[mask] = [255, 255, 255]

    return enhanced

transf = transforms.ToTensor()

# 定义数据集类
class MyDataset(data.Dataset):
    def __init__(self, all_img_paths, all_labels, type, transform=None):
        try:
            self.data = torch.load(f'./dataset/{type}_dataset.pth')
            with open(f'./dataset/{type}_labels.txt', 'r') as f:
                self.labels = [int(line.strip()) for line in f]
        except:
            print(f'No {type} dataset available')
            if not os.path.exists('dataset'):
                os.mkdir('dataset')

            self.img_paths = all_img_paths
            self.data = []
            self.labels = all_labels
            self.transform = transform
            length = len(all_img_paths)

            for i, path in enumerate(all_img_paths):
                print(f'{i} / {length}', end='\r')
                img = cv2.imread(path)
                img = enhance_color(img)
                self.data.append(transf(img))

            print(f'\nSaving as {type}_dataset.pth and {type}_labels.txt')
            torch.save(self.data, f'./dataset/{type}_dataset.pth')
            with open(f'./dataset/{type}_labels.txt', 'w') as f:
                for label in self.labels:
                    f.write(f'{label}\n')

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        # if self.transform:
        #     img = self.transform(img)
        img = self.data[index]
        return img, label

## 初始化模型参数
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0, std=convstd)
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=linstd)

def load_data(root_dir, batch_size, ratio):
    # 处理图像数据
    all_labels = []
    all_img_paths = []
    for label, piece in enumerate(params.types):
        class_dir = os.path.join(root_dir, piece)
        for img_name in os.listdir(class_dir):
            all_img_paths.append(os.path.join(class_dir, img_name))
            all_labels.append(label)
    all_labels = np.array(all_labels)
    # all_labels = to_categorical(all_labels, num_classes=params.num_classes)

    # 随机打乱数据
    index = np.random.permutation(len(all_img_paths))
    all_img_paths = np.array(all_img_paths)[index]
    all_labels = np.array(all_labels)[index]

    # 划分训练集和测试集
    train_size = int(ratio * len(all_img_paths))
    train_imgs = all_img_paths[:train_size]
    train_labels = all_labels[:train_size]
    test_imgs = all_img_paths[train_size:]
    test_labels = all_labels[train_size:]

    # 创建数据集和数据加载器
    train_dataset = MyDataset(train_imgs, train_labels, 'train')
    test_dataset = MyDataset(test_imgs, test_labels, 'test')
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def test(model, device, img_num, batch, debug=False):
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
    return correct / img_num

if __name__ == '__main__':
    convstds = [0.4, 0.5]
    linstds = [0.01, 0.03]
    batchs = [36, 60, 120]
    lrs = [0.001]
    total_epoch = 20

    vali_num = 96
    seq = []
    for i in range(vali_num):
        img_path = f'validation/{i}.jpg'
        img = cv2.imread(img_path)
        img_data = enhance_color(img)
        img_data = transf(img_data)
        seq.append(img_data)

    root_dir = './data/noModify'
    ratio = 0.8
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for convstd, linstd, BATCH_SIZE, lr in itertools.product(convstds, linstds, batchs, lrs):
        best = 0
        bests = []
        print(f'convstd: {convstd}, linstd: {linstd}, BATCH_SIZE: {BATCH_SIZE}, lr: {lr}')

        # 加载数据集
        train_loader, test_loader = load_data(root_dir, BATCH_SIZE, ratio)

        for itr in range(3):
            # 实例化模型
            net = CnnModel.ConvNet()
            net.to(device)

            # 初始化
            net.apply(init_weights)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.01)

            net, acc = CnnModel.train_model(
                net, train_loader, test_loader, criterion, optimizer, device, seq, num_epochs=total_epoch, debug=False, verbose=1
            )
            if not os.path.exists('model'):
                os.mkdir('model')

            highest = max(acc)
            epoch = acc.index(highest)
            # print(f'highest vali accuracy: {highest * 100}% at epoch{epoch+1}')
            bests.append(highest)
            print(f'\t{highest}')
            if highest > best:
                best = highest
                os.rename('model/tmp.pth', f'model/chess_{str(highest)[2:]}.pth')
        print("MEAN:", sum(bests) / len(bests))
        print("BEST RESULT:", best, '\n')