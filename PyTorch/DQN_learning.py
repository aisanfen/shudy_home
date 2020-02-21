import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE=32
LR=0.01           # learning rate
EPSILON=0.9         # 最优选择动作百分比
GAMMA=0.9           # 奖励递减参数
TARGET_REPLACE_ITER=100         # target update frequency  Q现实网络更新频率
MEMORY_CAPACITY=2000            # 记忆库大小
# 导入实验场所
env=gym.make('CartPole-v0')
env=env.unwrapped
# 小车的动作
N_ACTIONS= env.action_space.n              # 杆子能做的动作
N_STATE=env.observation_space.shape[0]     # 杆子能获取的环境信息数

# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1=nn.Linear(N_STATE,10)
        self.fc1.weight.data.normal_(0,0.1)   # 随机生成初始值
        self.out=nn.Linear(10,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self, x):
        x=self.fc1(x)
        x=F.relu(x)
        actions_value=self.out(x)
        return  actions_value

class DQN(object):
    def __init__(self):
        self.eval_net,self.target_net=Net(),Net()
        # target updating
        self.learn_step_counter=0       # 用于target更新计时
        # storing memory
        self.memory_counter=0           # 记忆库计数
        # initialize memory
        self.memory=np.zeros((MEMORY_CAPACITY,N_STATE*2+2))         # 初始化记忆库
        self.optimizer=torch.optim.Adam(self.eval_net.parameters(),lr=LR)
        self.loss_func=nn.MSELoss()

    # 接受环境值，并作出相应的变化
    def choose_action(self,x):
        x=torch.unsqueeze(torch.FloatTensor(x),0)
        # 选取最优动作
        if np.random.uniform()<EPSILON:   # greedy 当概率小于0.9后，选取概率高的
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:   # random
            action=np.random.randint(0,N_ACTIONS)

        return action

    # 存储记忆
    def store_transition(self,s,a,r,s_):
        transition=np.hstack((s,[a,r],s_))
        # replace the old memory with new memory
        index=self.memory_counter%MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.memory_counter+=1

    def learn (self):
        # target net update
        if self.learn_step_counter % TARGET_REPLACE_ITER==0:
            self.target_net.load_state_dict(self.eval_net.state.dict())

        # 随机抽取记忆库中的批数据对env_net进行更新
        sample_index=np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
        b_memory=self.memory[sample_index,:]
        b_s=Variable(torch.FloatTensor(b_memory[:,:N_STATE]))
        b_a = Variable(torch.LongTensor(b_memory[:,N_STATE:N_STATE+1].astpe(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATE+1:N_STATE]+2))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATE:]))

        # 放入神经网络进行学习
        # 针对做过的动作b_a，来选取q_eval的值，（q_eval原本所有动作的值）
        q_eval=self.eval_net(b_s).gater(1,b_a)   # 计算动作的价值  shape(batch,1)
        # debatch() 表示不希望反向传递继续更新，因为这个动作已经完成了
        q_next=self.target_net(b_s).detach()    # shape(batch,1)
        # 选取最大值
        q_targrt=b_r+GAMMA*q_next.max(1)[0]
        loss=self.loss_func(q_eval,q_targrt)

        # 计算，更新eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn=DQN()

print('\n Collecting experience...')
for i_episode in range(400):
    # 现在的状态
    s=env.reset()
    while True:
        env.render()     # 显示实验动画
        a=dqn.choose_action(s)

        # take action，得到环境反馈
        s_,r,done,info=env.step(a)

        # modify the reward，使DQN快速学习
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s,a,r,s_)

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break
        s=s
