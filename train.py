import torch.nn as nn
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import params
import CnnModel

# 定义数据集类
class MyDataset(data.Dataset):
    def __init__(self, all_img_paths, all_labels, transform=None):
        self.img_paths = all_img_paths
        self.labels = all_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label

# 定义数据预处理
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]
)

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
    train_dataset = MyDataset(train_imgs, train_labels, transform=transform)
    test_dataset = MyDataset(test_imgs, test_labels, transform=transform)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 加载数据集
root_dir = './data'
BATCH_SIZE = 120
ratio = 0.8
train_loader, test_loader = load_data(root_dir, BATCH_SIZE, ratio)

# 实例化模型
net = CnnModel.ConvNet()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)

# 初始化
net.apply(CnnModel.init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.1)

net, acc = CnnModel.train_model(net, train_loader, test_loader, criterion, optimizer, device, num_epochs=4)
if not os.path.exists('model'):
    os.mkdir('model')
acc = str(acc)[2:6]
torch.save(net.state_dict(), f'model/chess_{acc}.pth')