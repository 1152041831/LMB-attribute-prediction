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
    energy0 = predict_energy_density(input_features0)


    for current_delta in current_delta_list:
        input_features_add = input_features.copy()
        input_features_add[-1] = input_features[-1] + current_delta
        energy_add = predict_energy_density(input_features_add)
        input_features_sub = input_features.copy()
        input_features_sub[-1] = input_features[-1] - current_delta
        energy_sub = predict_energy_density(input_features_sub)
        energy_delta_add = energy_add - energy_orgin
        energy_delta_sub = energy_orgin - energy_sub

        derivative1 = energy_delta_add / (current_delta * energy0)
        derivative2 = energy_delta_sub / (current_delta * energy0)
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
    np.savetxt(f'../../predictions/all_current_derivative_by_mlp.csv', all_current_derivative, delimiter=',')
    print("保存成功！")
    # 读取 CSV 文件中的数组
    data_array_loaded = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp.csv', delimiter=',')

    print(data_array_loaded.shape)

# 计算组分与偏导数、能量密度的相关性系数
def calculating_correlation_no_data_preprocessing():
    all_current_derivative = np.loadtxt(f'../../predictions/all_current_derivative_by_mlp.csv', delimiter=',')
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
all_de_scores, all_en_scores = calculating_MI()


print("偏导的皮尔逊系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(correlation_de, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("能量密度的皮尔逊系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(correlation_en, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))

# 绝对值
print("|偏导|的皮尔逊系数: ")
for delta_value in range(5):
    abs_p_de = {key: [abs(num) for num in value] for key, value in correlation_de.items()}
    sorted_components = sort_components_descending(abs_p_de, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("|能量密度|的皮尔逊系数: ")
for delta_value in range(5):
    abs_p_en = {key: [abs(num) for num in value] for key, value in correlation_en.items()}
    sorted_components = sort_components_descending(abs_p_en, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))

print("偏导的MI:")
for i, de_mi_dict in enumerate(all_de_scores):
    print(f"current delta = 1e-{i+2}, sorted components:",
          ">".join([feature for feature, _ in sorted(de_mi_dict.items(), key=lambda x: x[1], reverse=True)]))
print("能量密度的MI:")
for i, en_mi_dict in enumerate(all_en_scores):
    print(f"current delta = 1e-{i+2}, sorted components:",
          ">".join([feature for feature, _ in sorted(en_mi_dict.items(), key=lambda x: x[1], reverse=True)]))

print("偏导的Spearman系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(sp_de, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))
print("能量密度的Spearman系数: ")
for delta_value in range(5):
    sorted_components = sort_components_descending(sp_en, delta_value)
    print(f"current delta = 1e-{delta_value + 2}, sorted components:", ">".join(sorted_components))

# 偏导等于0的数量: [0, 0, 0, 13, 198230]
# 偏导反向的数量: [0, 0, 0, 0, 29153]
# 偏导同向且大于0的数量: [10956, 10956, 10956, 10955, 4019]
# 偏导同向且小于0的数量(实际所使用的数据集): [289377, 289377, 289377, 289365, 68931]
# 所有偏导数的均值: [-0.016509502724984186, -0.01650419714676748, -0.016615316676797765, 0.025813215809237334, 562.3991541998298]
# 组分和偏导的皮尔逊相关系数:
# Se: [0.1107926681158315, 0.11075557196625531, 0.11046933196258976, 0.11183730693752329, 0.11845960457509443]
# Sb: [0.09997147969038421, 0.09993325863318418, 0.09968608925208641, 0.10132292138748598, 0.11629847597196746]
# Cu: [0.0971406387979728, 0.09710895242296513, 0.09686978742142233, 0.0987815921211055, 0.11271609986237785]
# Bi: [0.03373719799348082, 0.033725353828604324, 0.03352553603613848, 0.03587252876027125, 0.09848761130642002]
# Pb: [-0.1322423100786291, -0.13224050354491357, -0.1321221973882136, -0.12681558582504004, -0.024683509215864936]
# Sn: [-0.6678381750405153, -0.6679797295224725, -0.6688420449019303, -0.6640949974413265, -0.44144653397746264]
# Te: [0.4077062282559885, 0.40785719429938644, 0.408840432042355, 0.40065136697112236, 0.26968522266891504]
# Zn: [0.07950118640730276, 0.0794686822904473, 0.07923003355970588, 0.08151010997956427, 0.12503010540349]
# 组分和能量密度的皮尔逊相关系数:
# Se: [-0.0009335168759585812, -0.0009335168759585812, -0.0009335168759585812, -0.0009650779033983075, 0.04018015674423104]
# Sb: [-0.014557825568476299, -0.014557825568476299, -0.014557825568476299, -0.014562935735915708, 0.03466189416382393]
# Cu: [-0.023570325242181556, -0.023570325242181556, -0.023570325242181556, -0.023596256725827793, 0.028076223226879644]
# Bi: [-0.18400696305925307, -0.18400696305925307, -0.18400696305925307, -0.18399419146769166, -0.08654421343150102]
# Pb: [-0.3743182050991092, -0.3743182050991092, -0.3743182050991092, -0.3742913610906112, -0.28395859953029573]
# Sn: [-0.7646484227104686, -0.7646484227104686, -0.7646484227104686, -0.7646492951740902, -0.757215802338492]
# Te: [0.7920767170265711, 0.7920767170265711, 0.7920767170265711, 0.7920834173678346, 0.8524101893459665]
# Zn: [-0.09487806040018835, -0.09487806040018835, -0.09487806040018835, -0.09488511052975368, -0.016690677664825286]
# ============================================
# 组分和偏导的Spearman系数:
# Se: [-0.10390581305523947, -0.10402829687859669, -0.10465961470550648, -0.10815883450014208, 0.10108733324247884]
# Sb: [-0.13295766234745146, -0.13306237890570896, -0.13341807185759738, -0.1361082097782326, 0.09040381524716562]
# Cu: [-0.1374325320830936, -0.13753685570673754, -0.13815539151437994, -0.14044389257559695, 0.09657602847013191]
# Bi: [-0.2741462645461842, -0.2743222059129422, -0.2743798242047271, -0.2698997032302767, 0.0765575597886341]
# Pb: [-0.4176090758035333, -0.41785111045227236, -0.417784594146942, -0.40495370880471016, 0.011712773343165278]
# Sn: [-0.7427962456112811, -0.7431702245714806, -0.743124404715761, -0.7362019949507407, -0.5842962730130693]
# Te: [0.8132182669871133, 0.8136554909950606, 0.8139228739754407, 0.8075428514660785, 0.5462933047896991]
# Zn: [-0.19513524204164212, -0.19523801459946422, -0.19541558971993994, -0.19544010135563955, 0.10389026696134857]
# 组分和能量密度的Spearman系数:
# Se: [-0.12058852225262864, -0.12058852225262864, -0.12058852225262864, -0.12063247599517449, 0.06516834582833918]
# Sb: [-0.13294599977002333, -0.13294599977002333, -0.13294599977002333, -0.1329558078249491, 0.060337835637068196]
# Cu: [-0.14717648594497487, -0.14717648594497487, -0.14717648594497487, -0.14721200067608964, 0.047082903680925006]
# Bi: [-0.2817524903936552, -0.2817524903936552, -0.2817524903936552, -0.28174311874935415, -0.03519612287576884]
# Pb: [-0.415562763691147, -0.415562763691147, -0.415562763691147, -0.41553904180061735, -0.19883638967258455]
# Sn: [-0.7347354403168628, -0.7347354403168628, -0.7347354403168628, -0.7347363736245763, -0.8013869247850949]
# Te: [0.8167786324635051, 0.8167786324635051, 0.8167786324635051, 0.8167716870764141, 0.789872680994638]
# Zn: [-0.19504260303037496, -0.19504260303037496, -0.19504260303037496, -0.1950540617560476, 0.029164386960634933]
# ============================================
# 组分和偏导的MI:
# Se: [0.6509798907637836, 0.6286650202069359, 0.6379163694128529, 0.7483202708362491, 0.4248936438668882]
# Sb: [0.6623812244853857, 0.6497250176226621, 0.6604750719134662, 0.76512391791579, 0.4314351337143614]
# Cu: [0.670250213515089, 0.6439755760711825, 0.6556323375253905, 0.7671681963717538, 0.48485959247365207]
# Bi: [0.6885412636396602, 0.6788506292526568, 0.6847672330538686, 0.7828359812069889, 0.5478870728398135]
# Pb: [0.6878850451777265, 0.6638289876594734, 0.6670287864789866, 0.7406131938087017, 0.7307959383233782]
# Sn: [0.9794344370669501, 0.9646180985731454, 0.9662478251836313, 1.048660759187943, 1.074084408710171]
# Te: [1.151635709052084, 1.127966509511496, 1.1422017685253807, 1.2473574415311632, 0.8772085521061475]
# Zn: [0.6735170398850614, 0.663960025040427, 0.6715411380656198, 0.7839617019734444, 0.4997176120499569]
# 组分和能量密度的MI:
# Se: [0.249146899101083, 0.25051924326066244, 0.25012954527230313, 0.25039698521531717, 0.23841013369645525]
# Sb: [0.23465917595055252, 0.2347725975495103, 0.23456305313047343, 0.23339902933564538, 0.23557649552648385]
# Cu: [0.2523303306429119, 0.2506240703111744, 0.2519508065534022, 0.25320559214373217, 0.27406384429273123]
# Bi: [0.36001709623722356, 0.35864636557681706, 0.36132081389025084, 0.3601253750632041, 0.2969351108567482]
# Pb: [0.40978950262035463, 0.40861532217367635, 0.40774647413260556, 0.40880601768692726, 0.505563558273908]
# Sn: [0.6143815384146838, 0.618791746076043, 0.6192909138850733, 0.6210512971577571, 0.7843261993583486]
# Te: [0.7553072104245109, 0.7548899467044294, 0.7525975824345261, 0.754349870224567, 0.7245499431602749]
# Zn: [0.3508598230023736, 0.3494857361708834, 0.34903879158111284, 0.35171167788943514, 0.27819818948579256]
# ============================================
# 偏导的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Zn>Se>Sb>Cu>Bi>Pb>Sn
# 能量密度的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# |偏导|的皮尔逊系数:
# current delta = 1e-2, sorted components: Sn>Te>Pb>Se>Sb>Cu>Zn>Bi
# current delta = 1e-3, sorted components: Sn>Te>Pb>Se>Sb>Cu>Zn>Bi
# current delta = 1e-4, sorted components: Sn>Te>Pb>Se>Sb>Cu>Zn>Bi
# current delta = 1e-5, sorted components: Sn>Te>Pb>Se>Sb>Cu>Zn>Bi
# current delta = 1e-6, sorted components: Sn>Te>Zn>Se>Sb>Cu>Bi>Pb
# |能量密度|的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Sb>Se
# current delta = 1e-3, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Sb>Se
# current delta = 1e-4, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Sb>Se
# current delta = 1e-5, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Sb>Se
# current delta = 1e-6, sorted components: Te>Sn>Pb>Bi>Se>Sb>Cu>Zn
# 偏导的MI:
# current delta = 1e-2, sorted components: Te>Sn>Bi>Pb>Zn>Cu>Sb>Se
# current delta = 1e-3, sorted components: Te>Sn>Bi>Zn>Pb>Sb>Cu>Se
# current delta = 1e-4, sorted components: Te>Sn>Bi>Zn>Pb>Sb>Cu>Se
# current delta = 1e-5, sorted components: Te>Sn>Zn>Bi>Cu>Sb>Se>Pb
# current delta = 1e-6, sorted components: Sn>Te>Pb>Bi>Zn>Cu>Sb>Se
# 能量密度的MI:
# current delta = 1e-2, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Se>Sb
# current delta = 1e-3, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Se>Sb
# current delta = 1e-4, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Se>Sb
# current delta = 1e-5, sorted components: Te>Sn>Pb>Bi>Zn>Cu>Se>Sb
# current delta = 1e-6, sorted components: Sn>Te>Pb>Bi>Zn>Cu>Se>Sb
# 偏导的Spearman系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Zn>Se>Cu>Sb>Bi>Pb>Sn
# 能量密度的Spearman系数:
# current delta = 1e-2, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Se>Sb>Cu>Zn>Bi>Pb>Sn