import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# 获取数据
X, Y = get_dataset()

feature_names = ['Se', 'Sb', 'Cu', 'Bi', 'Pb', 'Sn', 'Te', 'Zn', 'J']

# 构建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, Y)

# 获取系数
coefficients = model.coef_
intercept = model.intercept_

# 打印公式
# 打印公式
terms = [f"{coefficients[0]:.3f}*{feature_names[0]}"]
for coef, feature in zip(coefficients[1:], feature_names[1:]):
    sign = " - " if coef < 0 else " + "
    terms.append(f"{sign}{abs(coef):.3f}*{feature}")
formula = "Energy Density = " + "".join(terms) + f" + {intercept:.3f}"

print("线性回归公式: ")
print(formula)

# 预测
# Y_pred = model.predict(X)
#
# # 评估模型性能
# mae = mean_absolute_error(Y, Y_pred)
# mse = mean_squared_error(Y, Y_pred)
# print(f"MAE: {mae} MSE: {mse}")