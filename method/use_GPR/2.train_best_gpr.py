from __future__ import annotations

import os
import time

import gpytorch
import numpy as np
import pandas as pd
import torch


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
    print("当前数据集长度为: ", len(X), " 特征长度为: ", len(X[0]))

    return X, Y

def train_with_ExGPR():
    # 设置随机种子
    torch.manual_seed(42)

    X, Y = get_dataset()

    X_train = torch.tensor(X, dtype=torch.float32)
    Y_train = torch.tensor(Y, dtype=torch.float32)

    max_epochs = 3390

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # print("原本的noise: ", likelihood.noise_covar.noise)
    likelihood.noise_covar.noise = 1.0
    # print("修改后的noise: ", likelihood.noise_covar.noise)
    model = ExactGPModel(X_train, Y_train, likelihood)
    # print("原本的length: ",model.covar_module.base_kernel.lengthscale)
    model.covar_module.base_kernel.lengthscale = 0.01
    # print("修改后的length: ",model.covar_module.base_kernel.lengthscale)

    print("开始训练...")
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, Y_train)
        loss.backward()
        optimizer.step()

        current_time = time.strftime("%H:%M:%S", time.localtime())
        print('%s - Epoch %d - Train Loss: %.3f ' % (
            current_time, epoch + 1, loss.item()
        ))

    best_model_state_dict = model.state_dict()

    torch.save(best_model_state_dict, f'GPR_best_model.pth')

train_with_ExGPR()









