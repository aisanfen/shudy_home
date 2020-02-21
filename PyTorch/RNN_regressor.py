# 循环神经网络（回归）
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 定义参数
TIME_STEP=10
INPUT_STEP=1
LR=0.02

# 数据
steps=np.linspace(0,np.pi*2,100,dtype=np.float32)
x_np=np.sin(steps)
y_np=np.cos(steps)
# 用sin曲线来预测cos曲线
# 画图
plt.plot(steps,y_np,'r-',label='target(cos)')
plt.plot(steps,x_np,'b-',label='input(sin)')
plt.legend(loc='best')
plt.show()

# 神经网络
class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn=nn.RNN(
            input_size=INPUT_STEP,#1
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out=nn.Linear(32,1)

    def forward(self, x,h_state):
        # h_state 为上一个神经网络学习到的
        # x shape(batch,time_step,input_step)
        # h_state shape(n_layers,batch,hidden_size)
        # r_out shape(batch, time_step, out_step=hidden_size)
        r_out,h_state=self.rnn(x,h_state)
        outs=[]
        # rnn动态图
        for time_step in range(r_out.size(1)):
            # 将每一个outs取time_step做一次hidden linear计算
            # 即每一个时间点都进行一次计算
            outs.append(self.out(r_out[:,time_step,:]))
        return  torch.stack(outs,dim=1),h_state

rnn=RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.MSELoss()

# 将最初的h_state为none
h_state=None

for step in range(60):
    start,end=step* np.pi,(step+1)*np.pi
    # use sin predicts cos
    steps=np.linspace(start,end,TIME_STEP,dtype=np.float32)
    x_np=np.sin(steps)
    y_np=np.cos(steps)

    # 将一维变为三维 shape(batch,time_step,input_size)
    x=Variable(torch.from_numpy(x_np[np.newaxis,:,np.newaxis]))     # 将一维变为三维 shape(batch,time_step,input_size)
    y=Variable(torch.from_numpy(y_np[np.newaxis,:,np.newaxis]))

    prediction,h_state=rnn(x,h_state)
    h_state=Variable(h_state.data)    #   在下次使用的时候需要将其变为variabel形式
    loss=loss_func(prediction,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


