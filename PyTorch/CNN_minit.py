import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MINIST = False

# 下载MNIST数据集，并保存在当前目录
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    # 将下载的格式改为tensor格式
    transform=torchvision.transforms.ToTensor(),  # 数据的内容为（0,1)--->  （0，255）
    download=DOWNLOAD_MINIST  # 下载数据
)
# plot one example
# print(train_data.data.size())
# print(train_data.targets.size())
# plt.imshow(train_data.data[0].numpy(),cmap='gray')
# plt.title('%i'%train_data.targets[0])
# plt.show()

# 批处理训练数据，每次50个
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 测试前两千个数据,并把0-255压缩化为0-1
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
with torch.no_grad():
    test_x = Variable(
        torch.unsqueeze(
            test_data.data,
            dim=1).type(
            torch.FloatTensor)[
                :2000] /
        255.)
test_y = test_data.targets[:2000]

# 建立CNN网络


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1
        # 维度为 1,28,28
        self.conv1 = nn.Sequential(
            # 卷积层---过滤器（三维，长宽高[特征属性 如rgb三原色]）
            nn.Conv2d(
                # 告诉前一个层的高度
                in_channels=1,
                # 输出16个feature的高度 也就是说提取了16个特征，进入下一层
                out_channels=16,
                # 表示feature的区域扫描为5*5
                kernel_size=5,
                # 表示调度范围为1，即每隔一步扫描一次
                stride=1,
                # 表示在长度或宽度不够的时候加0
                # if stride=1,padding=(kernel_size-1)/2=(5-1)/2
                padding=2
            ),
            # 维度为16,28,28
            nn.ReLU(),   # 激活函数
            # 维度为16,28,28
            # 池化层 即筛选更重要的信息
            nn.MaxPool2d(
                # 选择2*2区域内的最大值
                kernel_size=2
            )
            # 维度为16,14,14
        )
        # 维度为16,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # 维度为32,14,14
            nn.ReLU(),                  # 维度为32,14,14
            nn.MaxPool2d(2)              # 维度为32,7,7
        )
        # 10个分类
        # 维度为32,7,7
        self.out = nn.Linear(32 * 7 * 7, 10)

        # 展开为二维数据
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)         # (batch,32,7,7)
        x = x.view(x.size(0), -1)   # (batch,32*7*7)
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

# 优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 查看训练效果
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # 计算测试正确的数据的个数
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print(
                'Epoch:',
                epoch,
                '|train loss: %.4f' %
                loss.item(),
                '| test accuracy: %.2f' % accuracy)

test_output=cnn(test_x[:10])
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(test_y[:10].numpy(),'real number')
