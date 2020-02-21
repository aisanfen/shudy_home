import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
# 通过Variable建立图纸，作用是反向传播数据，后面参数表示是否计算梯度
variable = Variable(tensor,requires_grad=True)

t_out=torch.mean(tensor*tensor)
v_out=torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()   #（误差）反向传递的梯度 =求导
# v_out=1/4*sum(variable*variable)
# d(v_out)/d(variable)=1/4*2*variable=variable/2
print(variable.grad)
print('......')
print(variable)
print(variable.data)
print(variable.data.numpy())


