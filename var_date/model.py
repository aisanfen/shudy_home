import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torchvision.datasets as dsets
import data_processing


# 数据处理
data=data_processing.load_data()
new_data=data_processing.convert2onehot(data)
x=new_data.values[:,:21]
y=new_data.values[:,21:]
sep=int(0.7*len(new_data))
#torch_dataset=Data.TensorDataset(x,y)
train_loader=Data.DataLoader(
    dataset=new_data[:sep],
    batch_size=64,
    shuffle=True
)
test_loader=Data.DataLoader(
    dataset=new_data[sep:],
    batch_size=64,
    shuffle=True
)
print(len(test_loader))
# # 打乱顺序
# np.random.shuffle(new_data)
# # 划分数据

# train_data=new_data[:sep]
# test_data=new_data[sep:]


# 搭建神经网络
