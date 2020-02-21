import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

torch.manual_seed(1)

# 数据
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.2*torch.rand(x.size())
x,y=Variable(x,requires_grad=False),Variable(y,requires_grad=False)

# 保存
def save():
    # save net1
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    optimizer=torch.optim.SGD(net1.parameters(),lr=0.2)
    loss_func=torch.nn.MSELoss()

    for t in range(1,100):
        prediction=net1(x)
        loss=loss_func(prediction,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1,'net.pkl') # entire net
    torch.save(net1.state_dict(),'net_params.pkl') # 保留神经网络中所有的参数

    # 画图
    plt.figure(1,figsize=(10,3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)




# 提权神经网络

def restore_net():
    net2=torch.load('net.pkl')
    prediction=net2(x)
    # 画图
    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)



# 先建立网络，再复制参数
def restore_params():
    net3=torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction=net3(x)
    # 画图
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
    plt.show()
# save net1
save()

# 提取net2
restore_net()
#提取net3
restore_params()