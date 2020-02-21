import torch
import numpy as np

np_data=np.arange(6).reshape((2,3))
torch_data=torch.from_numpy(np_data)
tensor2array=torch_data.numpy()
print(
    'numpy',np_data,
    '\n torch',torch_data,  # 张量  数据+数据类型
    '\n tensor2array',tensor2array
)

# abs
data=[-1,-2,-3,-4]
tensor=torch.FloatTensor(data)

print(
    'abs:',
    '\n numpy:',np.abs(data),
    '\n tensor:',torch.abs(tensor)
)

# 矩阵相乘

data2=[[1,2],[3,4]]  # 矩阵
tensor2=torch.FloatTensor(data2)  #矩阵装换为张量

print(
    '\n numpy:',np.matmul(data2,data2),
    '\n tensor',torch.mm(tensor2,tensor2)
)
