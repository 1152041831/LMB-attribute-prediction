import math
import os
import time
from random import seed

import numpy as np
import pandas as pd
import torch
import gpytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

# 定义高斯过程回归模型
class MLP(nn.Module):
    def __init__(self, activation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 4)  # 第一层：9维输入，4维输出
        self.fc2 = nn.Linear(4, 1)  # 第二层：4维输入，1维输出
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.fc1(x))  # 第一层使用指定的激活函数
        x = self.fc2(x)  # 第二层不使用激活函数
        return x

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
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X = X.float()
    Y = Y.float()
    print("当前数据集长度为: ", len(X), " 特征长度为: ", len(X[0]))

    return X, Y

# LOO-CV 交叉验证函数
def loo_cv(hypers, X, Y, early_stopping_patience):
    print("当前超参数: ", hypers)

    loo = LeaveOneOut()
    all_mae = []
    all_mse = []

    for train_index, val_index in loo.split(X):
        train_x_fold, val_x_fold = X[train_index], X[val_index]
        train_y_fold, val_y_fold = Y[train_index], Y[val_index]

        train_x_fold = train_x_fold.clone().detach().float()
        val_x_fold = val_x_fold.clone().detach().float()
        train_y_fold = train_y_fold.clone().detach().float()
        val_y_fold = val_y_fold.clone().detach().float()

        optimizer = hypers['optimizer']
        activation = hypers['activation']
        lr = hypers['lr']

        model = MLP(activation)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()

        optimizer = optimizer(model.parameters(), lr=lr)

        best_loss = float('inf')  # mae
        patience = early_stopping_patience
        best_model_state_dict = model.state_dict()
        best_epoch = 0

        max_epochs = 100000

        for epoch in range(max_epochs):
            # if epoch == max_epochs - 1:
            #     print("max epoch!")
            model.train()
            output = model(train_x_fold)
            output = output.squeeze()
            # print("shape:",output.shape, train_y_fold.shape)
            loss = criterion(output, train_y_fold)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 使用验证集进行早停
            with torch.no_grad():
                model.eval()
                val_output = model(val_x_fold)
                val_output = val_output.squeeze()
                # val_loss = -mll(val_output, val_y_fold)
                val_loss = torch.mean(torch.abs(val_output - val_y_fold))

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state_dict = model.state_dict()
                    patience = early_stopping_patience
                    best_epoch = epoch
                else:
                    patience -= 1
                    if patience == 0:
                        break

        model.load_state_dict(best_model_state_dict)

        model.eval()
        with torch.no_grad():
            pred = model(val_x_fold)
            pred = pred.squeeze()
            mae = torch.mean(torch.abs(pred - val_y_fold))
            mse = torch.mean((pred - val_y_fold)**2)
            all_mae.append(mae)
            all_mse.append(mse)

    print("length mae&mse: ", len(all_mae), len(all_mse))
    print(f"验证集平均MAE: {np.mean(all_mae)}, 平均MSE: {np.mean(all_mse)}")
    return np.mean(all_mae), np.mean(all_mse)


# 准备数据
X, Y = get_dataset()

# 使用 LOO-CV 测试平均MSE和MAE
hypers = {
    'lr': 0.001,
    'activation': F.elu,
    'optimizer': optim.Adam,
}
# 对于每一种超参数组合，都会有一个score，score的计算方式为LOO-CV计算平均的MAE
mae, mse = loo_cv(hypers, X, Y, early_stopping_patience=10)


# 当前数据集长度为:  63  特征长度为:  9
# 当前超参数:  {'lr': 0.001, 'activation': <function elu at 0x00000264EFA9C7C0>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# length mae&mse:  63 63
# 验证集平均MAE: 0.12180954962968826, 平均MSE: 0.03905138745903969