import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as Data

# 参数
EPOCH = 1
BATCH_SIZE = 64    # 批处理数量
TIME_STEP = 28       # rnn time step /image height
INPUT_SIZE = 28      # rnn input step /image width
LR = 0.01            # learning rate
DWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root='./mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=DWNLOAD_MNIST)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_data = dsets.MNIST(
    root='./mnist/',
    train=False,
    transform=transforms.ToTensor())

with torch.no_grad():
    test_x = Variable(test_data.data).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

# 定义RNN


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,   # 可以增强计算能力
            batch_first=True,  # 表示输入数据batch所在的维度，FLASE表示（time_step,batch,input）  True表示batch第一个维度
        )
        # 10个为分类
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x的形式 （batch,time_step,input_size）
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])  # (batch,time_step,input)
        return out


rnn = RNN()
print(rnn)

optimizer=torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func=nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        b_x=Variable(x.view(-1,28,28))    # reshape x to (batch,time_step,input)
        b_y=Variable(y)
        output=rnn(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # 计算测试正确的数据的个数
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print(
                'Epoch:',
                epoch,
                '|train loss: %.4f' %
                loss.item(),
                '| test accuracy: %.2f' % accuracy)
test_output=rnn(test_x[:10].view(-1,28,28))
pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()
print(pred_y,'prediction number')
print(pred_y[:10],'real number')

