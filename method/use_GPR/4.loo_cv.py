import math
import os
import time
from random import seed

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
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

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
    scores = []
    all_mae = []
    all_mse = []

    for train_index, val_index in loo.split(X):
        train_x_fold, val_x_fold = X[train_index], X[val_index]
        train_y_fold, val_y_fold = Y[train_index], Y[val_index]

        train_x_fold = train_x_fold.clone().detach().float()
        val_x_fold = val_x_fold.clone().detach().float()
        train_y_fold = train_y_fold.clone().detach().float()
        val_y_fold = val_y_fold.clone().detach().float()

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

        best_loss = float('inf')  # mae
        patience = early_stopping_patience
        best_model_state_dict = model.state_dict()

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
                else:
                    patience -= 1
                    if patience == 0:
                        break

        model.load_state_dict(best_model_state_dict)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            pred_dist = likelihood(model(val_x_fold))
            mae = mean_absolute_error(pred_dist.mean, val_y_fold)
            mse = mean_squared_error(pred_dist.mean, val_y_fold)
            all_mae.append(mae)
            all_mse.append(mse)

    print("length mae&mse: ", len(all_mae), len(all_mse))
    print(f"验证集平均MAE: {np.mean(all_mae)}, 平均MSE: {np.mean(all_mse)}")
    return np.mean(all_mae), np.mean(all_mse)


# 准备数据
X, Y = get_dataset()

# 使用 LOO-CV 测试平均MSE和MAE
hypers = {
    'likelihood.noise_covar.noise': 1.0,
    'covar_module.base_kernel.lengthscale': 0.01,
    'lr': 0.001,
}
# 对于每一种超参数组合，都会有一个score，score的计算方式为LOO-CV计算平均的MAE
mae, mse = loo_cv(hypers, X, Y, early_stopping_patience=10)


# 当前数据集长度为:  63  特征长度为:  9
# 当前超参数:  {'likelihood.noise_covar.noise': 1.0, 'covar_module.base_kernel.lengthscale': 0.01, 'lr': 0.001}
# length mae&mse:  63 63
# 验证集平均MAE: 0.1731795370578766, 平均MSE: 0.10999322682619095
