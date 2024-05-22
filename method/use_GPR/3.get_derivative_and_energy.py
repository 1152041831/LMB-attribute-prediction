from __future__ import annotations


import os
import gpytorch
import pandas as pd
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

def predict_energy_density(X):
    X = torch.tensor(X, requires_grad=True, dtype=torch.float32)
    X = X.unsqueeze(0)
    # print(X)
    # 输入8组分比例+电流密度
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X))
        energy_density = observed_pred.mean
        # print(f"Current density: {X[0][8] * 100} Energy density: {energy_density * 100}")
    # print(energy_density)
    return energy_density

# 获得最佳模型
def get_best_model():
    # 设置随机种子
    torch.manual_seed(42)

    X, Y = get_dataset()

    X_train = torch.tensor(X, dtype=torch.float32)
    Y_train = torch.tensor(Y, dtype=torch.float32)

    state_dict = torch.load(f'GPR_best_model.pth')
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPModel(X_train, Y_train, likelihood)
    model.load_state_dict(state_dict)

    model.eval()
    likelihood.eval()

    return model, likelihood

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
    np.savetxt(f'../../predictions/all_current_derivative_by_gpr.csv', all_current_derivative, delimiter=',')
    print("保存成功！")
    # 读取 CSV 文件中的数组
    data_array_loaded = np.loadtxt(f'../../predictions/all_current_derivative_by_gpr.csv', delimiter=',')

    print(data_array_loaded.shape)

# 计算组分与偏导数、能量密度的相关性系数
def calculating_correlation_no_data_preprocessing():
    all_current_derivative = np.loadtxt(f'../../predictions/all_current_derivative_by_gpr.csv', delimiter=',')
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

def calculating_MI():
    # 从 CSV 文件加载数据
    data = np.loadtxt(f'../../predictions/all_current_derivative_by_gpr.csv', delimiter=',')
    names = ['Se', 'Sb', 'Cu', 'Bi', 'Pb', 'Sn', 'Te', 'Zn']
    # 组分和偏导数的mi
    mi_de = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}
    # 组分和能量密度的mi
    mi_en = {'Se': [], 'Sb': [], 'Cu': [], 'Bi': [], 'Pb': [], 'Sn': [], 'Te': [], 'Zn': []}

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


model, likelihood = get_best_model()

# 获取所有预测结果
# get_all_derivative()


# 计算皮尔逊系数和Spearman相关系数
correlation_de, correlation_en, sp_de, sp_en = calculating_correlation_no_data_preprocessing()
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



# 偏导等于0的数量: [1, 21, 193, 5042, 81104]
# 偏导反向的数量: [481, 37, 52, 798, 19904]
# 偏导同向且大于0的数量: [145095, 145302, 145210, 140593, 82596]
# 偏导同向且小于0的数量(实际所使用的数据集): [154756, 154973, 154878, 153900, 116729]
# 所有偏导数的均值: [-1.6212325632176894, -0.07782691754006502, 0.44441341840166426, 14.092048065266262, 203.54997271315233]
# 组分和偏导的皮尔逊相关系数:
# Se: [0.05280507213157963, 0.05327510286277414, 0.05298342994322351, 0.05294285468429043, 0.03181369464321742]
# Sb: [-0.014291191127772537, -0.013569906588384919, -0.013983913282963233, -0.01320102609258331, -0.052286101346469674]
# Cu: [0.05382710936352182, 0.05460006814067702, 0.05432209651824903, 0.05330009164716853, 0.019088806517353257]
# Bi: [0.0944837805818268, 0.09465998505723752, 0.09453758612658594, 0.09396363079588471, 0.05965210430130279]
# Pb: [-0.03976056534827421, -0.03993077624852711, -0.03982098834885048, -0.04048989735615786, -0.09921839458819094]
# Sn: [-0.2397494952659161, -0.23947647918313283, -0.24039489977966172, -0.2414529226983209, -0.34482737653374246]
# Te: [0.11069850076767493, 0.10954882408354144, 0.11060188475845782, 0.11235049521134766, 0.2608322559128497]
# Zn: [0.05175912742527378, 0.05282377020175246, 0.052452413664506786, 0.052096480699250935, 0.02419910331686074]
# 组分和能量密度的皮尔逊相关系数:
# Se: [0.09798166379440662, 0.0980087836383545, 0.09807524543338536, 0.09835566672957546, 0.0601540265884281]
# Sb: [0.12944776273226774, 0.12959513623744126, 0.12956858131363083, 0.12914904791954482, 0.062416390993932334]
# Cu: [0.09967117216387517, 0.09977076383812104, 0.09976144004921517, 0.09985517913314369, 0.046746237267775874]
# Bi: [-0.013187916102179906, -0.01323531835058915, -0.013217435892307013, -0.013276630992080776, -0.05087303311453992]
# Pb: [-0.26055826394035453, -0.26051753317606835, -0.2605587951848174, -0.26082593105254653, -0.22263709104482354]
# Sn: [-0.46600198767628104, -0.4661012880876642, -0.4660381796947943, -0.4652429935080648, -0.4557782842561311]
# Te: [0.3474245132238666, 0.34722105629804956, 0.3472337408632208, 0.3473807455817098, 0.38285841213352756]
# Zn: [0.09134404115954498, 0.09134997935647224, 0.09142843914937385, 0.09109498177422069, 0.03949163192890301]
# ============================================
# 组分和偏导的Spearman系数:
# Se: [-0.016450285606035377, -0.015133967829310058, -0.015931006033482853, -0.019613920658824235, -0.03609607476635219]
# Sb: [-0.06558570815251855, -0.0638436773115908, -0.06469595817673435, -0.06722469843731811, -0.11841714697985321]
# Cu: [-0.03027139086630554, -0.02854251686005737, -0.029324402325048085, -0.033312073309207844, -0.06483565124477932]
# Bi: [-0.006200396418047697, -0.00510235665319677, -0.0058178633282574286, -0.0106536491333134, -0.04706941134176015]
# Pb: [-0.13045530585407258, -0.12955729896478607, -0.12995422728079026, -0.13283790476023355, -0.15004550685148826]
# Sn: [-0.12721184895143, -0.1256777418968635, -0.12670854557486655, -0.14470992679670883, -0.30303186280653416]
# Te: [0.1941508972984342, 0.19240064121079095, 0.193417302401262, 0.2093704994223918, 0.36544120673942004]
# Zn: [-0.029367362858907547, -0.02767953574933802, -0.028407735106320733, -0.03131328222239105, -0.054556716325486344]
# 组分和能量密度的Spearman系数:
# Se: [0.1334183638941338, 0.13365377337827047, 0.1336815690854715, 0.13347533164324368, 0.07110047918608553]
# Sb: [0.18023486325113294, 0.18056101426893353, 0.18054933259896352, 0.17989756619556457, 0.09709824103893315]
# Cu: [0.14509313152968537, 0.14537233101482042, 0.14535339742401293, 0.14514198672373818, 0.06753100535639905]
# Bi: [0.059139614449070704, 0.059062299482290186, 0.05913977782751656, 0.05918450396889031, -0.0009882054582494746]
# Pb: [-0.17537540433448373, -0.17506442162570068, -0.17512626992570246, -0.17528905549027485, -0.1870894473512767]
# Sn: [-0.41532460173290603, -0.4152678402758548, -0.41506792619485505, -0.41335289760784133, -0.45340911718510735]
# Te: [0.39560340444038905, 0.3950758733174736, 0.39517293917144203, 0.39626821008268587, 0.4532839313333655]
# Zn: [0.1490181974936644, 0.14922159672430085, 0.14930252194837101, 0.14866137385436654, 0.07966857380497279]
# ============================================
# 组分和偏导的MI:
# Se: [0.2575716575225173, 0.2605490527926442, 0.26012123124785447, 0.2637467408943448, 0.3019944817259077]
# Sb: [0.2546312194408942, 0.25307257087510493, 0.2512560541894535, 0.256242637411527, 0.27425748072673617]
# Cu: [0.2696866257672359, 0.26770273865663885, 0.2685052177046856, 0.27392841181173244, 0.2921864223679762]
# Bi: [0.30238752717386763, 0.30033440011732715, 0.3043301482041576, 0.3144393393803466, 0.33759099358142297]
# Pb: [0.4350889832207696, 0.4356570166938054, 0.4398142681973405, 0.44275037737563805, 0.43335581692250225]
# Sn: [0.5571469790192105, 0.5584238583336876, 0.5618126405584869, 0.564617844380856, 0.5813527055212018]
# Te: [0.765539527725656, 0.7659178958150394, 0.7687338117082492, 0.7751226062727916, 0.8156447694558682]
# Zn: [0.24786218892547662, 0.24958088100309794, 0.24709837629052345, 0.25880474875142, 0.28159518576869047]
# 组分和能量密度的MI:
# Se: [0.25006138847060333, 0.251783277623582, 0.25387434543156573, 0.2538538669291621, 0.25743800194688093]
# Sb: [0.3992308199089849, 0.40126779909602517, 0.3977099482190929, 0.40023142898826114, 0.404609548598601]
# Cu: [0.2672129742024101, 0.2680714800555255, 0.26908340585811397, 0.27197583328437736, 0.265036900700375]
# Bi: [0.4418941697667407, 0.43901362064658, 0.4421284492354589, 0.441582555581455, 0.43304328644708256]
# Pb: [0.5804594710138344, 0.5806548277324826, 0.5820693926725067, 0.5816304345146399, 0.5865873648297972]
# Sn: [0.7134741293636151, 0.7111504113342595, 0.71299706138337, 0.7123239327418371, 0.6961964703544323]
# Te: [0.7475235794657884, 0.7483799542631848, 0.7476434710593121, 0.7507804247663108, 0.7660970770826081]
# Zn: [0.3920282503588641, 0.3953503299391423, 0.39476469029848316, 0.39530070298560016, 0.40497667780487623]
# ============================================
# 偏导的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Bi>Cu>Se>Zn>Sb>Pb>Sn
# current delta = 1e-3, sorted components: Te>Bi>Cu>Se>Zn>Sb>Pb>Sn
# current delta = 1e-4, sorted components: Te>Bi>Cu>Se>Zn>Sb>Pb>Sn
# current delta = 1e-5, sorted components: Te>Bi>Cu>Se>Zn>Sb>Pb>Sn
# current delta = 1e-6, sorted components: Te>Bi>Se>Zn>Cu>Sb>Pb>Sn
# 能量密度的皮尔逊系数:
# current delta = 1e-2, sorted components: Te>Sb>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Sb>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Sb>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Sb>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Sb>Se>Cu>Zn>Bi>Pb>Sn
# |偏导|的皮尔逊系数:
# current delta = 1e-2, sorted components: Sn>Te>Bi>Cu>Se>Zn>Pb>Sb
# current delta = 1e-3, sorted components: Sn>Te>Bi>Cu>Se>Zn>Pb>Sb
# current delta = 1e-4, sorted components: Sn>Te>Bi>Cu>Se>Zn>Pb>Sb
# current delta = 1e-5, sorted components: Sn>Te>Bi>Cu>Se>Zn>Pb>Sb
# current delta = 1e-6, sorted components: Sn>Te>Pb>Bi>Sb>Se>Zn>Cu
# |能量密度|的皮尔逊系数:
# current delta = 1e-2, sorted components: Sn>Te>Pb>Sb>Cu>Se>Zn>Bi
# current delta = 1e-3, sorted components: Sn>Te>Pb>Sb>Cu>Se>Zn>Bi
# current delta = 1e-4, sorted components: Sn>Te>Pb>Sb>Cu>Se>Zn>Bi
# current delta = 1e-5, sorted components: Sn>Te>Pb>Sb>Cu>Se>Zn>Bi
# current delta = 1e-6, sorted components: Sn>Te>Pb>Sb>Se>Bi>Cu>Zn
# 偏导的MI:
# current delta = 1e-2, sorted components: Te>Sn>Pb>Bi>Cu>Se>Sb>Zn
# current delta = 1e-3, sorted components: Te>Sn>Pb>Bi>Cu>Se>Sb>Zn
# current delta = 1e-4, sorted components: Te>Sn>Pb>Bi>Cu>Se>Sb>Zn
# current delta = 1e-5, sorted components: Te>Sn>Pb>Bi>Cu>Se>Zn>Sb
# current delta = 1e-6, sorted components: Te>Sn>Pb>Bi>Se>Cu>Zn>Sb
# 能量密度的MI:
# current delta = 1e-2, sorted components: Te>Sn>Pb>Bi>Sb>Zn>Cu>Se
# current delta = 1e-3, sorted components: Te>Sn>Pb>Bi>Sb>Zn>Cu>Se
# current delta = 1e-4, sorted components: Te>Sn>Pb>Bi>Sb>Zn>Cu>Se
# current delta = 1e-5, sorted components: Te>Sn>Pb>Bi>Sb>Zn>Cu>Se
# current delta = 1e-6, sorted components: Te>Sn>Pb>Bi>Zn>Sb>Cu>Se
# 偏导的Spearman系数:
# current delta = 1e-2, sorted components: Te>Bi>Se>Zn>Cu>Sb>Sn>Pb
# current delta = 1e-3, sorted components: Te>Bi>Se>Zn>Cu>Sb>Sn>Pb
# current delta = 1e-4, sorted components: Te>Bi>Se>Zn>Cu>Sb>Sn>Pb
# current delta = 1e-5, sorted components: Te>Bi>Se>Zn>Cu>Sb>Pb>Sn
# current delta = 1e-6, sorted components: Te>Se>Bi>Zn>Cu>Sb>Pb>Sn
# 能量密度的Spearman系数:
# current delta = 1e-2, sorted components: Te>Sb>Zn>Cu>Se>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Te>Sb>Zn>Cu>Se>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Te>Sb>Zn>Cu>Se>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Te>Sb>Zn>Cu>Se>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Sb>Zn>Se>Cu>Bi>Pb>Sn