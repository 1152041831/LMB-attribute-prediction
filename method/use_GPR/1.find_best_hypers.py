import math
import os
import time

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

# 定义高斯过程回归模型
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

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

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # print("原本的noise: ", likelihood.noise_covar.noise)
        likelihood.noise_covar.noise = hypers['likelihood.noise_covar.noise']
        # print("修改后的noise: ", likelihood.noise_covar.noise)

        model = ExactGPModel(train_x_fold, train_y_fold, likelihood)
        # print("原本的length: ",model.covar_module.base_kernel.lengthscale)
        model.covar_module.base_kernel.lengthscale = hypers['covar_module.base_kernel.lengthscale']
        # print("修改后的length: ",model.covar_module.base_kernel.lengthscale)


        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=hypers['lr'])

        best_loss = float('inf') # mae
        patience = early_stopping_patience
        best_model_state_dict = model.state_dict()
        best_epoch = 0

        model.train()
        likelihood.train()

        max_epochs = 100000

        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(train_x_fold)
            loss = -mll(output, train_y_fold)
            loss.backward()
            optimizer.step()

            # 使用验证集进行早停
            with torch.no_grad():
                model.eval()
                likelihood.eval()
                val_output = model(val_x_fold)
                # val_loss = -mll(val_output, val_y_fold)
                val_loss = mean_absolute_error(val_output.mean.numpy(), val_y_fold.numpy())

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_state_dict = model.state_dict()
                    patience = early_stopping_patience
                    best_epoch = epoch
                else:
                    patience -= 1
                    if patience == 0:
                        # current_time = time.strftime("%H:%M:%S", time.localtime())
                        # print('%s - Epoch %d/%d - Loss: %.3f  Val MAE: %.3f' % (
                        #     current_time, epoch + 1, max_epochs, loss.item(), val_loss.item()
                        # ))
                        # print(f'number {count}: Early stopping at epoch {epoch + 1}, best validation mae: {best_loss:.6f}')
                        all_epoch.append(best_epoch)
                        break
        count += 1

        model.load_state_dict(best_model_state_dict)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred_dist = likelihood(model(val_x_fold))
            mae = mean_absolute_error(pred_dist.mean, val_y_fold)
            scores.append(mae)

    print(f"平均epoch: ", int(np.mean(all_epoch)))
    print("验证集平均MAE: ", np.mean(scores))
    return np.mean(scores)

# 准备数据
torch.manual_seed(42)

X, Y = get_dataset()

# 设置超参数范围
lengthscale_range = [math.pow(10,i) for i in range(-4,2)]
noise_range = [math.pow(10,i) for i in range(-4,2)]
# [0.1, 0.01, 0.001]
lrs = [10 ** (-i) for i in range(1, 4)]

best_score = float('inf')
best_lengthscale = None
best_noise = None
best_lr = None

# 使用 LOO-CV 寻找最优超参数
for lr in lrs:
    for lengthscale in lengthscale_range:
        for noise in noise_range:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(noise),
                'covar_module.base_kernel.lengthscale': torch.tensor(lengthscale),
                'lr': lr,
            }
            # 对于每一种超参数组合，都会有一个score，score的计算方式为LOO-CV计算平均的MAE
            score = loo_cv(hypers, X, Y, early_stopping_patience=10)

            if score < best_score:
                best_score = score
                best_lengthscale = lengthscale
                best_noise = noise
                best_lr = lr


print(f"Best Lr: {best_lr}, Best Lengthscale: {best_lengthscale}, Best Noise: {best_noise}, Best MAE: {best_score}")
