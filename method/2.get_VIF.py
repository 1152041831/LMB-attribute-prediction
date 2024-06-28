from __future__ import annotations

import os
from random import seed

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.outliers_influence import variance_inflation_factor


init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

def get_VIF():
    file_folder = '../../datasets'  # 文件夹名称
    file_name = 'all_data2.xlsx'  # 文件名
    file_path = os.path.join(file_folder, file_name)

    # 使用pandas读取数据
    df = pd.read_excel(file_path)
    # 获取前8列的数据作为特征
    Se_feature = df['Se mole fraction'].tolist()
    Sb_feature = df['Sb mole fraction'].tolist()
    Cu_feature = df['Cu mole fraction'].tolist()
    Bi_feature = df['Bi mole fraction'].tolist()
    Pb_feature = df['Pb mole fraction'].tolist()
    Sn_feature = df['Sn mole fraction'].tolist()
    Te_feature = df['Te mole fraction'].tolist()
    Zn_feature = df['Zn mole fraction'].tolist()

    Se = Se_feature.copy()
    Sb = Sb_feature.copy()
    Cu = Cu_feature.copy()
    Bi = Bi_feature.copy()
    Pb = Pb_feature.copy()
    Sn = Sn_feature.copy()
    Te = Te_feature.copy()
    Zn = Zn_feature.copy()

    # 将特征数组转为 DataFrame
    df = pd.DataFrame({
        'Se': Se,
        'Sb': Sb,
        'Cu': Cu,
        'Bi': Bi,
        'Pb': Pb,
        'Sn': Sn,
        'Te': Te,
        'Zn': Zn
    })

    # 计算每个特征的 VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    print("VIF for each feature:")
    print(vif_data)


get_VIF()

# VIF for each feature:
#   feature       VIF
# 0      Se  1.967054
# 1      Sb  1.524983
# 2      Cu  2.139381
# 3      Bi  1.216669
# 4      Pb  1.082355
# 5      Sn  1.090585
# 6      Te  1.093066
# 7      Zn  1.093233