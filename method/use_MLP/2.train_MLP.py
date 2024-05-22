import os
from random import seed

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from gpytorch.metrics import mean_absolute_error

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

def get_dataset():
    file_folder = '../../datasets'  # 文件夹名称
    file_name = 'all_data2.xlsx'  # 文件名
    file_path = os.path.join(file_folder, file_name)

    # 使用pandas读取数据
    df = pd.read_excel(file_path)
    # 将数据存储为字典
    data_dict = df.to_dict(orient='records')

    X = [[data[key] / 100 for key in list(data.keys())[:9]] for data in data_dict]
    Y = [data['Energy Density(Wh/kg)'] / 100 for data in data_dict]

    X = np.array(X)
    Y = np.array(Y)

    print("当前数据集长度为: ", len(X), " 特征长度为: ", len(X[0]))

    return X, Y

# 定义函数用于保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    # print(f'Model saved to {path}')

# 定义函数用于加载模型
def load_model(path):
    model = MLP()
    model.load_state_dict(torch.load(path))
    model.eval()
    # print(f'Model loaded from {path}')
    return model

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 4)  # 第一层：9维输入，4维输出
        self.fc2 = nn.Linear(4, 1)  # 第二层：4维输入，1维输出

    def forward(self, x):
        x = F.elu(self.fc1(x))  # 第一层使用ELU激活函数
        x = self.fc2(x)  # 第二层不使用激活函数
        return x

def train_mlp():
    # 构建模型实例
    model = MLP()

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early Stopping 参数
    best_mae = float('inf')
    counter = 0

    X_train, y_train = get_dataset()

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # 训练模型
    epochs = 806
    for epoch in range(epochs):
        model.train()
        # 前向传播
        outputs = model(X_train)
        outputs = outputs.squeeze()
        # print(outputs.shape,y_train.shape)
        loss = criterion(outputs, y_train)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印当前损失和验证集上的性能
        print(
            f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}')

    print("保存模型...")
    save_model(model, 'MLP_best_model.pth')


    print('训练完成！')


# 训练模型
train_mlp()
