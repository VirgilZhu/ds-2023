import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# 创建模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        # 定义模型
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self,x):
        # 定义前向算法
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x,1)
        # print(x.shape)
        result = self.classifier(x)
        return result

# 下载数据集或者加载数据集
train_dataset = MNIST(root='../data/',train=True,transform=transforms.ToTensor(),download=True)
test_dataset = MNIST(root='../data/',train=False,transform=transforms.ToTensor())
# 加载数据: 分批次，每批256个数据
batch_size = 32
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)
# start time
start_time = time.time()
# 创建模型
model = LeNet()
# 模型放入GPU中
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 定义损失函数
loss_func = nn.CrossEntropyLoss()
loss_list = [] # 用来存储损失值
# 定义优化器
SGD = optim.SGD(params=model.parameters(),lr=0.001,momentum=0.9)
# 训练指定次数
for i in range(10):
    loss_temp = 0 # 定义一个损失值，用来打印查看
    # 其中j是迭代次数，data和label都是批量的，每批32个
    for j,(batch_data,batch_label) in enumerate(train_loader):
        # 启用GPU
        batch_data,batch_label = batch_data.cuda(),batch_label.cuda()
        # 清空梯度
        SGD.zero_grad()
        # 模型训练
        prediction = model(batch_data)
        # 计算损失
        loss = loss_func(prediction,batch_label)
        loss_temp += loss
        # BP算法
        loss.backward()
        # 更新梯度
        SGD.step()
        if (j + 1) % 200 == 0:
            print('第%d次训练，第%d批次，损失值: %.3f' % (i + 1, j + 1, loss_temp / 200))
            loss_temp = 0

end_time = time.time()
print('训练花了: %d s' % int((end_time-start_time)))

# 测试
correct = 0
for batch_data,batch_label in test_loader:
    batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
    prediction = model(batch_data)
    predicted = torch.max(prediction.data, 1)[1]
    correct += (predicted == batch_label).sum()
print('准确率: %.2f %%' % (100 * correct / 10000)) # 总共10000个测试数据
