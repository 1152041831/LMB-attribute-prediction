from __future__ import annotations

from random import seed

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import torch.nn.functional as F

init_seed = 42
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.cuda.manual_seed_all(init_seed)
np.random.seed(init_seed)  # 用于numpy的随机数
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(init_seed)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(9, 4)  # 第一层：9维输入，4维输出
        self.fc2 = nn.Linear(4, 1)  # 第二层：4维输入，1维输出

    def forward(self, x):
        x = F.elu(self.fc1(x))  # 第一层使用ELU激活函数
        x = self.fc2(x)  # 第二层不使用激活函数
        return x

def predict_energy_density(X):
    X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
    X = X.unsqueeze(0)
    # print(X)
    # 输入8组分比例+电流密度
    with torch.no_grad():
        prediction = model(X)
        # print(f"Current density: {X[0][8] * 100} Energy density: {energy_density * 100}")
    # print(energy_density)
    return prediction.item()

# 获得最佳模型
def get_best_model():
    model = MLP()
    model.load_state_dict(torch.load('MLP_best_model.pth'))
    model.eval()
    return model

# 定义计算偏导数的函数
def calculate_partial_derivative(input_features):
    # [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    current_delta_list = [10 ** (-i) for i in range(2, 7)]

    all_derivative = []

    energy_orgin = predict_energy_density(input_features)

    # 预测电流密度为0时能量密度
    input_features0 = input_features.copy()
    input_features0[-1] = 0
    # energy0 = predict_energy_density(input_features0)


    for current_delta in current_delta_list:
        input_features_add = input_features.copy()
        input_features_add[-1] = input_features[-1] + current_delta
        energy_add = predict_energy_density(input_features_add)
        input_features_sub = input_features.copy()
        input_features_sub[-1] = input_features[-1] - current_delta
        energy_sub = predict_energy_density(input_features_sub)
        energy_delta_add = energy_add - energy_orgin
        energy_delta_sub = energy_orgin - energy_sub

        # derivative1 = energy_delta_add / (current_delta * energy0)
        # derivative2 = energy_delta_sub / (current_delta * energy0)
        derivative1 = energy_delta_add / current_delta
        derivative2 = energy_delta_sub / current_delta
        derivative = 999.0

        if derivative1 == 0 or derivative2 == 0: # 偏导数为0
            derivative = 999.0
        else:
            # 同向
            if derivative1/derivative2 > 0:
                derivative = (derivative1 + derivative2) / 2
            else: # 反向
                derivative = -999.0

        all_derivative.append(derivative)

    # print("all_derivative: ", all_derivative)
    return all_derivative

# 计算所有组合下的偏导数
def get_all_derivative():
    # 读取 CSV 文件中的数组
    all_combinations = np.loadtxt('../combinations.csv', delimiter=',')
    # all_combinations = all_combinations[:22]

    all_current_derivative = []

    print(f"开始计算所有{len(all_combinations)}种组合的偏导数...")
    # 使用 tqdm 显示进度条
    for c in tqdm(all_combinations, desc="计算进度", unit="组合"):
        now_input_features = c.copy()
        all_derivative= calculate_partial_derivative(now_input_features)
        now_energy= predict_energy_density(now_input_features)
        now_input_features = np.append(now_input_features, now_energy)

        now_input_features = np.append(now_input_features, all_derivative[0])
        now_input_features = np.append(now_input_features, all_derivative[1])
        now_input_features = np.append(now_input_features, all_derivative[2])
        now_input_features = np.append(now_input_features, all_derivative[3])
        now_input_features = np.append(now_input_features, all_derivative[4])

        # print(now_input_features)
        # now_input_features[-1] = now_derivative
        all_current_derivative.append(now_input_features)


    print("计算完成！")
    # print(len(all_derivative))
    # print(all_derivative[0])
    all_current_derivative = np.array(all_current_derivative)

    print("开始保存所有电流密度和偏导数...")
    # 保存数组到 CSV 文件
    np.savetxt(f'../../predictions/all_current_derivative_by_mlp_withoutE0.csv', all_current_derivative, delimiter=',')
    print("保存成功！")
    # 读取 CSV 文件中的数组
    data_array_loaded = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp_withoutE0.csv', delimiter=',')

    print(data_array_loaded.shape)

# 计算组分与偏导数、能量密度的相关性系数
def calculating_correlation_no_data_preprocessing():
    all_current_derivative = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp_withoutE0.csv', delimiter=',')
    components = ['Se', 'Sb', 'Cu', 'Bi', 'Pb', 'Sn', 'Te', 'Zn']
    # 组分和偏导数的相关性系数
    correlation = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和能量密度的相关性系数
    correlation_en = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和偏导数的Spearman
    sp_de = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和能量密度的Spearman
    sp_en = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和偏导数的Spearman
    sp_de_p = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和能量密度的Spearman
    sp_en_p = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 所有偏导数均值
    all_partial = []

    error_0 = [0, 0, 0, 0, 0]  # 偏导数为0
    error_inversion = [0, 0, 0, 0, 0]  # 偏导数反向
    error_pos = [0, 0, 0, 0, 0]  # 偏导数大于0
    used = [0, 0, 0, 0, 0] # 实际所使用到的数据量
    for i, component in enumerate(components):
        now_com = all_current_derivative[:, i]
        derivative1 = all_current_derivative[:, -5]  # current delta : 1e-2
        derivative2 = all_current_derivative[:, -4]  # current delta : 1e-3
        derivative3 = all_current_derivative[:, -3]  # current delta : 1e-4
        derivative4 = all_current_derivative[:, -2]  # current delta : 1e-5
        derivative5 = all_current_derivative[:, -1]  # current delta : 1e-6
        energy = all_current_derivative[:, -6] # 能量密度

        all_derivative = [derivative1, derivative2, derivative3, derivative4, derivative5]

        for derivative in all_derivative:
            # 当前delta current density下偏导数平均值
            all_partial.append(np.mean(derivative))

        for j, derivative in enumerate(all_derivative):
            # 数据筛选
            indices_0 = np.where(derivative == 999) # 偏导数是0
            indices_useless = np.where(derivative == -999) # 偏导数反向
            indices_use_pos = np.where((derivative != -999) & (derivative != 999) & (derivative > 0))  # 偏导数同向但是大于0
            indices_use_neg = np.where((derivative != -999) & (derivative != 999) & (derivative < 0)) # 偏导数同向但是小于0

            error_0[j] = len(indices_0[0])
            error_inversion[j] = len(indices_useless[0])
            error_pos[j] = len(indices_use_pos[0])
            used[j] = len(indices_use_neg[0])

            new_now_com = now_com.copy()
            new_derivative = derivative.copy()
            new_energy = energy.copy()
            new_now_com = new_now_com[indices_use_neg]
            new_derivative = new_derivative[indices_use_neg]
            new_energy = new_energy[indices_use_neg]


            # 计算皮尔逊相关系数
            correlation_matrix = np.corrcoef(new_now_com, new_derivative)
            correlation_matrix_en = np.corrcoef(new_now_com, new_energy)

            pearson_correlation_coefficient = correlation_matrix[0, 1] # 偏导数的
            # pearson_correlation_coefficient = "{:.3f}".format(pearson_correlation_coefficient)
            pearson_correlation_coefficient_en = correlation_matrix_en[0, 1]
            # pearson_correlation_coefficient_en = "{:.3f}".format(pearson_correlation_coefficient_en)

            correlation[component].append(pearson_correlation_coefficient)
            correlation_en[component].append(pearson_correlation_coefficient_en)

            # 计算Spearman系数
            now_sp_de, now_p_de = spearmanr(new_now_com, new_derivative)
            now_sp_en, now_p_en = spearmanr(new_now_com, new_energy)
            sp_de[component].append(now_sp_de)
            sp_de_p[component].append(now_p_de)
            sp_en[component].append(now_sp_en)
            sp_en_p[component].append(now_p_en)



    print(f"偏导等于0的数量:", error_0)
    print(f"偏导反向的数量:", error_inversion)
    print(f"偏导同向且大于0的数量:", error_pos)
    print(f"偏导同向且小于0的数量(实际所使用的数据集):", used)
    print(f"所有偏导数的均值:", all_partial[:5])

    print("组分和偏导的皮尔逊相关系数:")
    for key, value in correlation.items():
        print(f"{key}: {value}")
    print("组分和能量密度的皮尔逊相关系数:")
    for key, value in correlation_en.items():
        print(f"{key}: {value}")
    print("============================================")
    print("组分和偏导的Spearman系数:")
    for key, value in sp_de.items():
        print(f"{key}: {value}")
    # print("组分和偏导的Spearman系数的p值:")
    # for key, value in sp_de_p.items():
    #     print(f"{key}: {value}")
    print("组分和能量密度的Spearman系数:")
    for key, value in sp_en.items():
        print(f"{key}: {value}")
    # print("组分和能量密度的Spearman系数的p值:")
    # for key, value in sp_en_p.items():
    #     print(f"{key}: {value}")
    print("============================================")

    return correlation, correlation_en, sp_de, sp_en

# 计算MI互信息
# 在计算互信息（MI）时，需要输入的是所有特征与目标变量之间的值分布情况，而不是某一个特征对应的所有数值。
# 互信息考虑了两个变量之间的整体关系，因此需要考虑所有特征与目标变量之间的值的分布情况。
def calculating_MI():
    # 从 CSV 文件加载数据
    data = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp.csv', delimiter=',')
    names = ['Se', 'Sb', 'Cu', 'Bi', 'Pb', 'Sn', 'Te', 'Zn']
    # 组分和偏导数的mi
    mi_de = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和能量密度的mi
    mi_en = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # KAN得到的公式中各个特征对应的导数（即偏导）
    kan_de = {'Se': 0.35, 'Sb': 0.33, 'Cu': 0.24, 'Bi': -0.03, 'Pb': -0.32, 'Sn': -0.24, 'Te': 0.1, 'Zn': -0.09, 'J': -0.15}

    # 提取特征和输出结果
    features = data[:, :8]  # 前8列：8组分
    energy = data[:, 9]  # 第10列：能量密度

    # 创建偏导数字典列表
    all_de_scores = []
    all_en_scores = []
    for i in range(1, 6):
        derivative = data[:, (i-6)]
        derivative = np.array(derivative)
        # 数据筛选，选择偏导数同向且<0的数据用于计算
        indices_use_neg = np.where((derivative != -999) & (derivative != 999) & (derivative < 0))[0]  # 偏导数同向但是小于0

        new_now_com = features.copy()
        new_derivative = derivative.copy()
        new_energy = energy.copy()

        new_now_com = new_now_com[indices_use_neg]
        new_derivative = new_derivative[indices_use_neg]
        new_energy = new_energy[indices_use_neg]

        # 能量密度的MI
        en_mi_scores = mutual_info_regression(new_now_com, new_energy)
        en_mi_dict = dict(zip(names, en_mi_scores))

        # 偏导数的MI
        de_mi_scores = mutual_info_regression(new_now_com, new_derivative)
        de_mi_dict = dict(zip(names, de_mi_scores))


        for key, value in de_mi_dict.items():
            new_list = mi_de[key]
            new_list.append(value)
            mi_de[key] = new_list
        for key, value in en_mi_dict.items():
            new_list = mi_en[key]
            new_list.append(value)
            mi_en[key] = new_list

        all_de_scores.append(de_mi_dict)
        all_en_scores.append(en_mi_dict)

    print("组分和偏导的MI:")
    for key, value in mi_de.items():
        print(f"{key}: {value}")
    print("组分和能量密度的MI:")
    for key, value in mi_en.items():
        print(f"{key}: {value}")


    print("============================================")

    return all_de_scores, all_en_scores


# 从大到小
def sort_components_descending(data, delta):
    return sorted(data.keys(), key=lambda x: -float(data[x][delta]))
# 从小到大
def sort_components_ascending(data, delta):
    return sorted(data.keys(), key=lambda x: float(data[x][delta]))

components = ['Se', 'Sb', 'Cu', 'Bi', 'Pb', 'Sn', 'Te', 'Zn']

model = get_best_model()

# 获取所有预测结果
# get_all_derivative()

# 计算皮尔逊系数和Spearman相关系数
correlation_de, correlation_en, sp_de, sp_en  = calculating_correlation_no_data_preprocessing()
# 计算MI
# all_de_scores, all_en_scores = calculating_MI()


print("偏导的皮尔逊系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(correlation_de, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("能量密度的皮尔逊系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(correlation_en, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("偏导的Spearman系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(sp_de, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("能量密度的Spearman系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(sp_en, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))

# MLP
# 开始计算所有213928种组合的偏导数...
# 计算进度: 100%|██████████| 213928/213928 [01:04<00:00, 3336.59组合/s]
# 计算完成！
# 开始保存所有电流密度和偏导数...
# 保存成功！
# (213928, 15)
# 偏导等于0的数量: [0, 0, 0, 18, 152085]
# 偏导反向的数量: [0, 0, 0, 0, 13563]
# 偏导同向且大于0的数量: [0, 0, 0, 0, 43]
# 偏导同向且小于0的数量(实际所使用的数据集): [213928, 213928, 213928, 213910, 48237]
# 所有偏导数的均值: [-0.08518769597368382, -0.08518919170399859, -0.08532747685929332, -5.043773304118406e-05, 646.8342049081824]
# 组分和偏导的皮尔逊相关系数:
# Se: [-0.009151790086114811, -0.009159657778738096, -0.009436665643491306, -0.007366988325819413, -0.26314703682818696]
# Sb: [-0.15090772019379547, -0.15106977174208608, -0.151033716703211, -0.11791662627595574, -0.2651621553814504]
# Cu: [-0.06832173190997477, -0.06835045609012548, -0.06896079502942547, -0.05060627095985196, -0.20253296424040892]
# Bi: [0.004170521388950588, 0.004212215838701976, 0.004317455467956563, -0.0003197760397388759, 0.05459323094323564]
# Pb: [0.2575477699782836, 0.2577487481807492, 0.25897229820868717, 0.2024981211076527, 0.47868163270263303]
# Sn: [0.08583852006263588, 0.08593434725570256, 0.0859803547606181, 0.06292919845124316, 0.31723865342143265]
# Te: [-0.014974333499615897, -0.015049186339878656, -0.015369092294196883, -0.009095053709137923, -0.29748281706592794]
# Zn: [-0.1042012357403682, -0.10426623932432702, -0.10446983876693809, -0.08010754869482428, -0.10281257167827682]
# 组分和能量密度的皮尔逊相关系数:
# Se: [0.31109016242848453, 0.31109016242848453, 0.31109016242848453, 0.31104364571715803, 0.325740214703662]
# Sb: [0.26941367394364335, 0.26941367394364335, 0.26941367394364335, 0.2695244389081858, 0.3097818046665671]
# Cu: [0.21045046608024695, 0.21045046608024695, 0.21045046608024695, 0.21045727283880858, 0.23521427896778407]
# Bi: [-0.1399432976008901, -0.1399432976008901, -0.1399432976008901, -0.13988164701923603, -0.029869181624765744]
# Pb: [-0.6331204920966423, -0.6331204920966423, -0.6331204920966423, -0.6330777324228033, -0.620924021276881]
# Sn: [-0.4506916479525001, -0.4506916479525001, -0.4506916479525001, -0.4506477356120215, -0.3795626059405074]
# Te: [0.3692539940185723, 0.3692539940185723, 0.3692539940185723, 0.36904852622267076, 0.3703464270240778]
# Zn: [0.06354714117908597, 0.06354714117908597, 0.06354714117908597, 0.06358438863774192, 0.1352608643190197]
# ============================================
# 组分和偏导的Spearman系数:
# Se: [-0.02928141036102333, -0.01960923855617034, -0.028538817309385785, -0.007823221748505605, -0.22927258281119256]
# Sb: [-0.1277659490976801, -0.13094736066108345, -0.12912927937922103, -0.09167827499685487, -0.2322890993373784]
# Cu: [-0.06523211168720126, -0.05829018875919084, -0.06873918510933111, -0.03749040460913729, -0.18064741554458874]
# Bi: [0.01923338900380358, 0.02037455848189602, 0.01987044237812096, -0.008297308609953164, 0.03667066821903236]
# Pb: [0.16423255439359827, 0.1617488858324811, 0.17220828288124443, 0.10729876647666084, 0.47952490882416005]
# Sn: [0.05903731761455393, 0.053848469816218214, 0.05827086283811013, 0.020186911640177035, 0.298441391360765]
# Te: [-0.03400727812975627, -0.031246439401190797, -0.0362474798468085, -0.008415843650294458, -0.2568314750787887]
# Zn: [-0.06842665449778897, -0.06529059155930443, -0.06742120419067582, -0.05823246024755189, -0.10137657857857946]
# 组分和能量密度的Spearman系数:
# Se: [0.28287070701895217, 0.28287070701895217, 0.28287070701895217, 0.2828326850346414, 0.2859897947417209]
# Sb: [0.24354206658940622, 0.24354206658940622, 0.24354206658940622, 0.24361982495749882, 0.2703454597671138]
# Cu: [0.18912629791988603, 0.18912629791988603, 0.18912629791988603, 0.1891059666558541, 0.21189923281202322]
# Bi: [-0.14450404604948391, -0.14450404604948391, -0.14450404604948391, -0.1444345434722539, -0.015914643357707807]
# Pb: [-0.5800282883230825, -0.5800282883230825, -0.5800282883230825, -0.5799770472003489, -0.5854223480980763]
# Sn: [-0.4260757865377441, -0.4260757865377441, -0.4260757865377441, -0.4260145610035132, -0.358702383615292]
# Te: [0.33575112548070274, 0.33575112548070274, 0.33575112548070274, 0.33562360150517817, 0.3170280709474243]
# Zn: [0.049510764088406385, 0.049510764088406385, 0.049510764088406385, 0.04952917823687413, 0.13497870856150648]
# ============================================
# 偏导的皮尔逊系数:
# current delta = 1e-2, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-3, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-4, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-5, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-6, sorted components: Pb>Sn>Bi>Zn>Cu>Se>Sb>Te
# 能量密度的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-6, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# 偏导的Spearman系数:
# current delta = 1e-2, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-3, sorted components: Pb>Sn>Bi>Se>Te>Cu>Zn>Sb
# current delta = 1e-4, sorted components: Pb>Sn>Bi>Se>Te>Zn>Cu>Sb
# current delta = 1e-5, sorted components: Pb>Sn>Se>Bi>Te>Cu>Zn>Sb
# current delta = 1e-6, sorted components: Pb>Sn>Bi>Zn>Cu>Se>Sb>Te
# 能量密度的Spearman系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
# current delta = 1e-6, sorted components: Te>Se>Sb>Cu>Zn>Bi>Sn>Pb
