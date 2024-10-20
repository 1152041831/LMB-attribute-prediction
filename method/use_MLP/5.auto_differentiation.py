import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 4)  # 第一层：9维输入，4维输出
        self.fc2 = nn.Linear(4, 1)  # 第二层：4维输入，1维输出

    def forward(self, x):
        x = F.elu(self.fc1(x))  # 使用ELU激活函数
        x = self.fc2(x)  # 第二层不使用激活函数
        return x


# 加载训练好的模型
def get_best_model():
    model = MLP()
    model.load_state_dict(torch.load('MLP_best_model.pth'))
    model.eval()  # 设置为评估模式
    return model


# 加载待预测数据集
all_combinations = np.loadtxt('../combinations.csv', delimiter=',')
all_combinations_tensor = torch.tensor(all_combinations, dtype=torch.float32)  # 转换为tensor

# 加载模型
model = get_best_model()

# 存储每条数据的偏导数
gradients = []

# 对每条数据逐条计算∂E/∂J（相对于电流密度，第9个特征的偏导数）
for i in range(all_combinations_tensor.shape[0]):
    # 获取当前数据并设置 requires_grad=True
    input_tensor = all_combinations_tensor[i:i + 1].clone().detach()
    input_tensor.requires_grad = True

    # 预测能量密度
    prediction = model(input_tensor)

    # 反向传播计算梯度
    prediction.backward()

    # 获取电流密度J的梯度（第9列特征）
    gradients.append(input_tensor.grad[:, 8].item())  # 取第9列的梯度，并转换为标量

# 输出偏导数结果
gradients = np.array(gradients)


print(len(gradients), np.mean(gradients))
# 自动微分 平均偏导数： -0.08518610071799881
# 有限差分 平均偏导数： -0.08518919170399859

# 计算 有限差分法得到偏导数 与 自动微分得到偏导数 的MAE：
all_current_derivative = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp_withoutE0.csv', delimiter=',')
derivative2 = all_current_derivative[:, -4]  # current delta : 1e-3

mae = np.mean(np.abs(derivative2 - gradients))
mse = np.mean((derivative2 - gradients) ** 2)

print(f"MAE: {mae:.11f}")
print(f"MSE: {mse:.11f}")

# 计算 自动微分 与 有限差分 所得到的偏导的MAE和MSE。
# MAE: 0.00005186174
# MSE: 0.00000000435
