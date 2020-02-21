import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 建立数据集
n_data = torch.ones(100, 2)
# 数据1
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
# 数据2
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
# x表示坐标值，y表示标签2，torch.cat的作用是合并数据
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)   # 32 位的浮点数
y = torch.cat((y0, y1),).type(torch.LongTensor)    # 64位的int
x = Variable(x)
y = Variable(y)
# 画图
# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


# 建立神经网络

# 继承torch的module的功能
class Net(torch.nn.Module):
    # 输入值数量，每一层神经元的数量
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        # 定义每一层的样式
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    # 前向传递,x为输入信息
    def forward(self, x):
        # 激励函数嵌套隐藏层的输出信息
        x1 = F.relu(self.hidden(x))
        x = self.predict(x1)
        return x


# 输入两个特征，10个隐藏层，2个输出特征
net = Net(2, 10, 2)
# 输出[0,1]表示为第一种类型
# 输出[1.0]表示为第二种类型
print(net)
# 可视化
plt.ion()  # 将图像变为实时变化的
plt.show()

# 优化器优化神经网络的参数
optimizer = torch.optim.SGD(
    net.parameters(),
    lr=0.002)  # ls是学习效率=梯度下降的幅度=每一步前进的大小
# 计算误差,CrossEntropyLoss计算概率，用于分类
# 例如三分类：[0.1,0.2,0.7]=1,表示该输入值为第三类
# 对应标签的值为[0, 0,1]
# loss_func（误差函数）计算得到的是误差，
# 在代码中，用交叉熵来描述这种误差，【交叉熵为分布概率的信息熵的差】
# 损失函数的计算过程就是实际结果和预测结果的误差大小
# 损失函数，有多种，得到的是误差，也就是说损失函数可以得到交叉熵，kl散度，残差等等
# 交叉熵、KL散度等都是描述误差的
# 综上所述损失函数（爷爷）->误差（爸爸）->用交叉熵来描述（儿子）
loss_func = torch.nn.CrossEntropyLoss()

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)  # [-2,-12,20]然后用F.softmax(out)转化为概率

    # 优化过程
    optimizer.zero_grad()  # 梯度清零，避免爆栈
    loss.backward()  # 反向传递参数给每一个神经元
    optimizer.step()  # 优化梯度

    # 画图
    if t % 5 == 0:
        plt.cla()
        # 过了一道 softmax 的激励函数后的最大概率才是预测值
        prediction = torch.max(F.softmax(out), 1)[1]

        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(
            x.data.numpy()[
                :, 0], x.data.numpy()[
                :, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y) / 200.  # 预测中有多少和真实值一样
        plt.text(
            1.5, -4, 'Accuracy=%.2f' %
            accuracy, fontdict={
                'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()  # 停止画图
plt.show()
