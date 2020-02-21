# 激励函数就是将线性方程变为非线性方程的函数，例如卷积神经网络可以选择relu函数，循环神经网络选择relu 或者tanh函数]
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # x data (从-5到5的200个数据)
x = Variable(x)
x_np = x.data.numpy()  # plt数据只能使用numpy格式

y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()

# 画图
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')  # 给图像加上图例

plt.figure(1, (8, 6))
plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.figure(1, (8, 6))
plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.legend(loc='best')

plt.figure(1, (8, 6))
plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.legend(loc='best')

plt.show()
