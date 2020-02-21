# 加速神经网络：SGD（每次只用一次样本）
# 传统参数更新：w+=Learning rate*dx(校正值）
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# hyper paprmeters 超参数，提前定义要用的数据
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# 数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# # 打印数据
# plt.scatter(x.numpy(),y.numpy())
# plt.show()

# 批处理
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    # num_workers=2
)

# 创建神经网络


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1,20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x,):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 通过使用不同的优化器，建立不同的神经网络
net_SDG = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SDG, net_Momentum, net_RMSprop, net_Adam]

# 建立不同的优化器
opt_SGD = torch.optim.SGD(net_SDG.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
# 记录误差
loss_his = [[], [], [], []]

for epoch in range(EPOCH):
    print(epoch)
    for step, (batch_x, batch_y) in enumerate(loader):
        b_x = Variable(batch_x)
        b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, loss_his):
            output = net(b_x)
            loss = loss_func(output, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            l_his.append(loss.data.numpy())    # 记录数据


# 打印
labeles = ['SDG', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(loss_his):
    plt.plot(l_his, label=labeles[i])
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()


# optimizer=torch.optim.SGD()
