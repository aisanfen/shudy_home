# GAN生成对抗网络 能够通过随机数生成需要的数据
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

# 设置超参数
BATCH_SIZE=64
LR_G=0.0001               # learning rate for generator  新手画家    生成器
LR_D=0.0001               # learning rate for discriminator  新手鉴赏家    鉴赏器
N_IDEAS=5                 # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS=15         # it could be total point G can draw in the canvas
PAINT_POINTS=np.vstack(np.linspace(-1,1,ART_COMPONENTS) for _ in range(BATCH_SIZE))

# show our beautiful painting range
plt.plot(PAINT_POINTS[0],2*np.power(PAINT_POINTS[0],2)+1,c='#74BCFF',lw=3,label='upper bound')
plt.plot(PAINT_POINTS[0],1*np.power(PAINT_POINTS[0],2)+0,c='#FF9359',lw=3,label='upper bound')
plt.legend(loc='upper right')
plt.show()


#生成一批著名画家的画
def artist_works():
    # 随机生成一个数 二维
    a=np.random.uniform(1,2,size=BATCH_SIZE)[:,np.newaxis]
    # 生成随机曲线
    paintings=a* np.power(PAINT_POINTS,2)+(a-1)
    paintings=torch.from_numpy(paintings).float()
    return Variable(paintings)

# 新手画家通过N_IEADS个灵感，画出ART_COMPONENTS个线段，若画出来的线段和画家的画差不多，就表示新手画家学到了
G=nn.Sequential(
    nn.Linear(N_IDEAS,128),
    nn.ReLU(),
    nn.Linear(128,ART_COMPONENTS)
)

# 新手鉴赏家接受信息
D=nn.Sequential(
    nn.Linear(ART_COMPONENTS,128),
    nn.ReLU(),
    nn.Linear(128,1),
    # 判断接受到的信息是谁的
    nn.Sigmoid(),
)

# 优化器
opt_D=torch.optim.Adam(D.parameters(),lr=LR_D)
opt_G=torch.optim.Adam(G.parameters(),lr=LR_G)

plt.ion()

# 进行学习
for step in range(10000):
    # 接受著名画家的画
    artist_painting=artist_works()
    # 新手画家的灵感
    G_ideas=Variable(torch.randn(BATCH_SIZE,N_IDEAS))
    # 新手画家画画
    G_paintings=G(G_ideas)
    # 分辨画的类别
    prob_artist0=D(artist_painting)
    prob_artist1=D(G_paintings)

    # 误差
    # 减少新手画家的画的概率
    D_loss=-torch.mean(torch.log(prob_artist0)+torch.log(1-prob_artist1))
    # 增加新手画家的画的概率
    G_loss=torch.mean(torch.log(1-prob_artist1))

    opt_D.zero_grad()
    # 参数的意思为保留当前网络的参数，并传给下一个神经网络
    D_loss.backward(retain_graph=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()
    # 画图
    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()