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

def predict_energy_density(model, X):
    X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
    X = X.unsqueeze(0)
    # print(X)
    # 输入8组分比例+电流密度
    with torch.no_grad():
        prediction = model(X)
    # print(energy_density)
    return prediction.item()

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

# LOO-CV 交叉验证函数
def loo_cv(hypers, X, Y, early_stopping_patience):
    print("当前超参数: ", hypers)

    loo = LeaveOneOut()
    scores = []
    count = 1
    all_epoch = []

    for train_index, val_index in loo.split(X):
        train_x_fold, val_x_fold = X[train_index], X[val_index]
        train_y_fold, val_y_fold = Y[train_index], Y[val_index]

        train_x_fold = torch.tensor(train_x_fold, dtype=torch.float32)
        val_x_fold = torch.tensor(val_x_fold, dtype=torch.float32)
        train_y_fold = torch.tensor(train_y_fold, dtype=torch.float32)
        val_y_fold = torch.tensor(val_y_fold, dtype=torch.float32)

        optimizer = hypers['optimizer']
        activation = hypers['activation']
        lr = hypers['lr']

        model = MLP(activation)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()

        optimizer = optimizer(model.parameters(), lr=lr)


        best_loss = float('inf') # mae
        patience = early_stopping_patience
        best_model_state_dict = model.state_dict()
        best_epoch = 0

        max_epochs = 100000

        for epoch in range(max_epochs):
            if epoch == max_epochs-1:
                print("max epoch!")
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
                val_loss = torch.mean(torch.abs(val_output-val_y_fold))

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state_dict = model.state_dict()
                    patience = early_stopping_patience
                    best_epoch = epoch
                else:
                    patience -= 1
                    if patience == 0:
                        # print(f"LOO第{count}/{len(X)}组训练结束", "Best mae:", best_loss.item(), "最佳epoch为:", best_epoch)
                        all_epoch.append(best_epoch)
                        count += 1
                        break

        model.load_state_dict(best_model_state_dict)

        model.eval()
        with torch.no_grad():
            pred = model(val_x_fold)
            pred = pred.squeeze()
            mae = torch.mean(torch.abs(pred-val_y_fold))
            scores.append(mae)

    print(f"平均epoch: ", int(np.mean(all_epoch)))
    print("验证集平均MAE: ", np.mean(scores))
    return np.mean(scores)

# 准备数据
X, Y = get_dataset()

# 设置超参数
activations = [torch.relu, F.elu, nn.Tanh()]
optimizers = [optim.SGD, optim.Adam, optim.AdamW, optim.RMSprop]
lrs = [10 ** (-i) for i in range(1, 4)] # [0.1, 0.01, 0.001]

best_score = float('inf')
best_activation = None
best_optimizer = None
best_lr = None

# 使用 LOO-CV 寻找最优超参数
for lr in lrs:
    for activation in activations:
        for optimizer in optimizers:
            hypers = {
                'lr': lr,
                'activation': activation,
                'optimizer': optimizer,
            }
            # 对于每一种超参数组合，都会有一个score，score的计算方式为LOO-CV计算平均的MAE
            score = loo_cv(hypers, X, Y, early_stopping_patience=10)

            if score < best_score:
                best_score = score
                best_activation = activation
                best_optimizer = optimizer
                best_lr = lr

print(f"Best lr: {best_lr}, Best activation: {best_activation}, Best optimizer: {best_optimizer}, Best MAE: {best_score}")

# 当前数据集长度为:  63  特征长度为:  9
# 当前超参数:  {'lr': 0.1, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  22
# 验证集平均MAE:  0.78306764
# 当前超参数:  {'lr': 0.1, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  11
# 验证集平均MAE:  0.6552022
# 当前超参数:  {'lr': 0.1, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  13
# 验证集平均MAE:  0.6102049
# 当前超参数:  {'lr': 0.1, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  44
# 验证集平均MAE:  0.6569193
# 当前超参数:  {'lr': 0.1, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  20
# 验证集平均MAE:  0.818992
# 当前超参数:  {'lr': 0.1, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  10
# 验证集平均MAE:  0.7630701
# 当前超参数:  {'lr': 0.1, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  9
# 验证集平均MAE:  0.6905272
# 当前超参数:  {'lr': 0.1, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  19
# 验证集平均MAE:  0.5918419
# 当前超参数:  {'lr': 0.1, 'activation': Tanh(), 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  21
# 验证集平均MAE:  0.75810635
# 当前超参数:  {'lr': 0.1, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  6
# 验证集平均MAE:  0.9449086
# 当前超参数:  {'lr': 0.1, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  6
# 验证集平均MAE:  0.9886096
# 当前超参数:  {'lr': 0.1, 'activation': Tanh(), 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  5
# 验证集平均MAE:  0.7344863
# 当前超参数:  {'lr': 0.01, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  231
# 验证集平均MAE:  0.4434954
# 当前超参数:  {'lr': 0.01, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  81
# 验证集平均MAE:  0.5631595
# 当前超参数:  {'lr': 0.01, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  594
# 验证集平均MAE:  0.5690049
# 当前超参数:  {'lr': 0.01, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  25
# 验证集平均MAE:  0.3645798
# 当前超参数:  {'lr': 0.01, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  237
# 验证集平均MAE:  0.38708764
# 当前超参数:  {'lr': 0.01, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  92
# 验证集平均MAE:  0.51079047
# 当前超参数:  {'lr': 0.01, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  98
# 验证集平均MAE:  0.50384825
# 当前超参数:  {'lr': 0.01, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  25
# 验证集平均MAE:  0.40829685
# 当前超参数:  {'lr': 0.01, 'activation': Tanh(), 'optimizer': <class 'torch.optim.sgd.SGD'>}
# 平均epoch:  528
# 验证集平均MAE:  0.3727566
# 当前超参数:  {'lr': 0.01, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  49
# 验证集平均MAE:  0.5616252
# 当前超参数:  {'lr': 0.01, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  72
# 验证集平均MAE:  0.49007162
# 当前超参数:  {'lr': 0.01, 'activation': Tanh(), 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  28
# 验证集平均MAE:  0.50227433
# 当前超参数:  {'lr': 0.001, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# max epoch!
# 平均epoch:  3283
# 验证集平均MAE:  0.21735887
# 当前超参数:  {'lr': 0.001, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  961
# 验证集平均MAE:  0.17974932
# 当前超参数:  {'lr': 0.001, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  791
# 验证集平均MAE:  0.13535479
# 当前超参数:  {'lr': 0.001, 'activation': <built-in method relu of type object at 0x00007FF96B4612E0>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  481
# 验证集平均MAE:  0.20181717
# 当前超参数:  {'lr': 0.001, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.sgd.SGD'>}
# max epoch!
# 平均epoch:  2808
# 验证集平均MAE:  0.18661432
# 当前超参数:  {'lr': 0.001, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  806
# 验证集平均MAE:  0.120201595
# 当前超参数:  {'lr': 0.001, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  729
# 验证集平均MAE:  0.1356321
# 当前超参数:  {'lr': 0.001, 'activation': <function elu at 0x000002B18C920A40>, 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  364
# 验证集平均MAE:  0.236719
# 当前超参数:  {'lr': 0.001, 'activation': Tanh(), 'optimizer': <class 'torch.optim.sgd.SGD'>}
# max epoch!
# 平均epoch:  3700
# 验证集平均MAE:  0.17528886
# 当前超参数:  {'lr': 0.001, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adam.Adam'>}
# 平均epoch:  789
# 验证集平均MAE:  0.13500413
# 当前超参数:  {'lr': 0.001, 'activation': Tanh(), 'optimizer': <class 'torch.optim.adamw.AdamW'>}
# 平均epoch:  810
# 验证集平均MAE:  0.16099147
# 当前超参数:  {'lr': 0.001, 'activation': Tanh(), 'optimizer': <class 'torch.optim.rmsprop.RMSprop'>}
# 平均epoch:  477
# 验证集平均MAE:  0.16311446
# Best lr: 0.001, Best activation: <function elu at 0x000002B18C920A40>, Best optimizer: <class 'torch.optim.adam.Adam'>, Best MAE: 0.12020159512758255