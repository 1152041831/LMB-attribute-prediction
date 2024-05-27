import os
from random import seed

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut

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
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    X = X.float()
    Y = Y.float()
    print("当前数据集长度为: ", len(X), " 特征长度为: ", len(X[0]))

    return X, Y


# 准备数据
X, Y = get_dataset()

# 创建线性回归模型
model = LinearRegression()

# 创建 Leave-One-Out 交叉验证对象
loo = LeaveOneOut()

# 初始化列表来存储每次迭代的 MSE 和 MAE
mse_scores = []
mae_scores = []

# 进行 Leave-One-Out 交叉验证
for train_index, test_index in loo.split(X):
    # 划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # 训练模型
    model.fit(X_train, Y_train)

    # 预测
    Y_pred = model.predict(X_test)

    # 计算 MSE 和 MAE
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)

    # 将结果添加到列表中
    mse_scores.append(mse)
    mae_scores.append(mae)

print(len(mse_scores),len(mae_scores))
print(mse_scores)
# 计算平均 MSE 和 MAE
mean_mse = np.mean(mse_scores)
mean_mae = np.mean(mae_scores)

print("Mean MAE:", mean_mae)
print("Mean MSE:", mean_mse)


# Mean MAE: 0.30206813474023153
# Mean MSE: 0.1411896649603896
