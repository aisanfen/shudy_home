import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)

N_SAMPLES=20
N_HIDDEN=300

# train data
x=torch.unsqueeze(torch.linspace(-1,1,N_SAMPLES),1)
y=x+0.3*torch.normal(torch.zeros(N_SAMPLES,1),torch.ones(N_SAMPLES,1))

# test data
test_x=torch.unsqueeze(torch.linspace(-1,1,N_SAMPLES),1)
test_y=test_x+0.3*torch.normal(torch.zeros(N_SAMPLES,1),torch.ones(N_SAMPLES,1))

# show data
plt.scatter(x.data.numpy(),y.data.numpy(),c='magenta',s=50,alpha=0.5,label='train')
plt.scatter(test_x.data.numpy(),test_y.data.numpy(),c='cyan',alpha=0.5,label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5,2.5))
plt.show()

# 建立两个神经网络，一个具有处理过拟合的能力，一个没有
net_overfitting=torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),        # 输入层
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN), # 隐藏层
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1)         # 输出层
)

net_dropped=torch.nn.Sequential(
    torch.nn.Linear(1,N_HIDDEN),        # 输入层
    # 随机屏蔽50%的神经元
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,N_HIDDEN), # 隐藏层
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN,1)         # 输出层
)


# 打印
print(net_overfitting)
print(net_dropped)

# 优化器
optimizer_ofit=torch.optim.Adam(net_overfitting.parameters(),lr=0.01)
optimizer_drop=torch.optim.Adam(net_dropped.parameters(),lr=0.01)

loss_func=torch.nn.MSELoss()

for t in range(500):
    # 训练
    pred_ofit=net_overfitting(x)
    loss_ofit = loss_func(pred_ofit, y)

    pred_drop=net_dropped(x)
    loss_drop=loss_func(pred_drop,y)

    optimizer_ofit.zero_grad()
    loss_ofit.backward()
    optimizer_ofit.step()

    optimizer_drop.zero_grad()
    loss_drop.backward()
    optimizer_drop.step()




    if t % 10 == 0:
        # 预测模式，取消drop
        net_overfitting.eval()
        net_dropped.eval()
        test_pred_ofit=net_overfitting(test_x)
        test_pred_drop=net_dropped(test_x)

        # plotting
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(),
                 fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left');
        plt.ylim((-2.5, 2.5));
        plt.pause(0.1)
        # 训练模式，补回drop
        net_overfitting.train()
        net_dropped.train()
