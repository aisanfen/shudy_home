import torch
import torch.utils.data as Data

BATCH_SIZE = 5
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 训练数据位data_tensor 计算误差（标签） target_tensor
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 定义是否要打断训练，进行随机抽样
    # num_workers=2,   # 多线程读数据
)

for epoch in range(3):
    # enumerate 每一步释放一小批数据进行处理
    #     for x in loader:
    #         print(x)
    for step, (batch_x, batch_y) in enumerate(loader):
        # 训练
        print('Epoch', epoch, '| Step',step, '| batch_x:', batch_x.numpy(),
              '| batch_y:', batch_y.numpy())
