import torch.nn as nn
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import params
from IPython import embed

# 定义模型
class ConvNet(nn.Module):
    def __init__(self, ch = 3, h = params.size[0], w = params.size[1]):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = ch,
                out_channels = 16,
                kernel_size = 3,
                padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        self.classifier = nn.Sequential(
            # nn.Linear(128*(h//16)*(w//16), 2048),
            nn.Linear(64*(h//8)*(w//8), 2048),
            nn.ReLU(),
            # nn.Linear(128,84),
            # nn.ReLU(),
            nn.Linear(2048,params.num_classes)) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = self.classifier(x)
        return x

map = {piece: i for i, piece in enumerate(params.types)}
def test(model, device, batch, debug=False):
    img_num = len(batch)
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

# 定义训练函数
def train_model(model, train_loader, test_loader, loss_func, optimizer, device, vali_seq, num_epochs = 5, debug=False, verbose=False):    
    """
    model: 网络模型；     train_loader: 训练数据集； test_loader: 测试数据集
    loss_func: 损失函数； optimizer: 优化方法；      num_epochs: 训练的轮数
    device: 控制是否使用GPU
    """
    train_loss_all = []
    train_acc_all = []
    val_acc_all = []
    val_loss_all = []
    test_acc_all = []
    max_acc = 0
    
    length = len(train_loader)
    gap = length//30
    total = length//gap - 1

    for epoch in range(num_epochs):
        if verbose:
            if epoch:
                print('-'*10)
            print("Epoch {}/{}".format(epoch + 1,num_epochs))
        
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        
        for step,data in enumerate(train_loader):
            # if epoch == 0 and step == 0:
            #     embed()
            model.train() 
            x,y = data[0].to(device), data[1].to(device)
            output = model(x) ## 模型在 X 上的输出: N * num_class
            pre_lab = torch.argmax(output, 1) ## 获得预测结果

            if debug and epoch == 0:
                # print(epoch,step)
                debug = False
                for i, img in enumerate(x):
                    img = transforms.ToPILImage()(img)
                    plt.imshow(img)
                    plt.title(params.chinese[int(pre_lab[i])])
                    plt.show()
                    if i > 3: break

            loss = loss_func(output, y) ## 损失
            optimizer.zero_grad() ## 每次迭代将梯度初始化为0
            loss.backward() ## 损失的后向传播， 计算梯度
            optimizer.step() ## 使用梯度进行优化
            train_loss += loss.item() * x.size(0) ## 统计模型预测损失
            train_corrects += torch.sum(pre_lab == y.data)
            train_num += x.size(0)
            
            if verbose and step % gap  == gap - 1:
                cont = step//gap
                if cont > total:
                    cont = total
                print('%2d'%cont+'/','total','['+'='*cont+'>'+'-'*(total-cont)+']',
                      'loss: {:.4f} - accuracy: {:.4f}'.format(train_loss/train_num,train_corrects.double().item()/train_num)
                      ,'\r',end="")
 
        # 计算验证集上的准确率
        for data in test_loader:
            model.eval()
            X_test, y_test = data[0].to(device), data[1].to(device)

            if debug and epoch == 4:
                debug = 0
                for i, img in enumerate(X_test):
                    img = transforms.ToPILImage()(img)
                    plt.imshow(img)
                    plt.title(params.types[int(y_test[i])])
                    plt.show()
                    if i: break

            with torch.no_grad():
                output = model(X_test)
            test_loss = loss_func(output, y_test)
            _, pred = torch.max(output.data, 1)
            val_corrects += torch.sum(pred == y_test.data)
            val_loss += test_loss.item()*X_test.size(0)
            val_num += X_test.size(0)
   
        train_loss_all.append(train_loss/train_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_loss_all.append(val_loss/val_num)
        val_acc_all.append(val_corrects.double().item()/val_num)
        
        test_acc = test(model, device, vali_seq)
        test_acc_all.append(test_acc)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), f'model/tmp.pth')
        
        if verbose:
            print('')
            print("No.{} Train Loss is:{:.4f}, Train_accuracy is {:.4f}%"
                .format(epoch+1, train_loss_all[-1],train_acc_all[-1] * 100))
            print("No.{} Val Loss is:{:.4f},  Val_accuracy is {:.4f}%"
                .format(epoch+1, val_loss_all[-1], val_acc_all[-1] * 100))    
            print(f"No.{epoch+1} Test_accuracy is {test_acc_all[-1]*100}%")
        
    return model, test_acc_all