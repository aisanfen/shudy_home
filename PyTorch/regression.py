import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# 建立数据集
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x=Variable(x)
y=Variable(y)
#画图
# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

# 建立神经网络

# 继承torch的module的功能
class Net(torch.nn.Module):
    # 输入值数量，每一层神经元的数量
    def __init__(self,n_features,n_hidden,n_output):
        super(Net,self).__init__()  #继承__init__功能
        # 定义每一层的样式
        self.hidden=torch.nn.Linear(n_features,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)

    # 前向传递,x为输入信息
    def forward(self,x):
        #激励函数嵌套隐藏层的输出信息
        x1=F.relu(self.hidden(x))
        x=self.predict(x1)
        return x

net=Net(1,10,1)
print(net)
# 可视化
plt.ion() #将图像变为实时变化的
plt.show()

# 优化器优化神经网络的参数
optimizer=torch.optim.SGD(net.parameters(),lr=0.2) # ls是学习效率=梯度下降的幅度=每一步前进的大小
# 计算误差 均方差处理回归问题
loss_func=torch.nn.MSELoss()

for t in range(100):
    prediction=net(x)

    loss=loss_func(prediction,y)

    optimizer.zero_grad() #梯度清零，避免爆炸
    loss.backward()  # 反向传递参数给每一个神经元
    optimizer.step() #优化梯度

    # 画图
    if t%5==0:
        plt.cla()
        plt.scatter(x.data.numpy(),y.data.numpy())
        plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

