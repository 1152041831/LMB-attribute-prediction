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

    for current_delta in current_delta_list:
        input_features_add = input_features.copy()
        input_features_add[-1] = input_features[-1] + current_delta
        energy_add = predict_energy_density(input_features_add)
        input_features_sub = input_features.copy()
        input_features_sub[-1] = input_features[-1] - current_delta
        energy_sub = predict_energy_density(input_features_sub)
        energy_delta_add = energy_add - energy_orgin
        energy_delta_sub = energy_orgin - energy_sub

        derivative1 = energy_delta_add / (current_delta)
        derivative2 = energy_delta_sub / (current_delta)
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
    np.savetxt(f'../../predictions/all_current_derivative_by_gpr_withoutE0.csv', all_current_derivative, delimiter=',')
    print("保存成功！")
    # 读取 CSV 文件中的数组
    data_array_loaded = np.loadtxt(f'../../predictions/all_current_derivative_by_gpr_withoutE0.csv', delimiter=',')

    print(data_array_loaded.shape)

# 计算组分与偏导数、能量密度的相关性系数
def calculating_correlation_no_data_preprocessing():
    all_current_derivative = np.loadtxt(f'../../predictions/all_current_derivative_by_gpr_withoutE0.csv', delimiter=',')
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

# GPR
# 当前数据集长度为:  63  特征长度为:  9
# 开始计算所有213928种组合的偏导数...
# 计算进度: 100%|██████████| 213928/213928 [17:17<00:00, 206.14组合/s]
# 计算完成！
# 开始保存所有电流密度和偏导数...
# 保存成功！
# (213928, 15)
# 偏导等于0的数量: [1, 25, 262, 2663, 50956]
# 偏导反向的数量: [629, 44, 67, 673, 15033]
# 偏导同向且大于0的数量: [105928, 106209, 106077, 104338, 70170]
# 偏导同向且小于0的数量(实际所使用的数据集): [107370, 107650, 107522, 106254, 77769]
# 所有偏导数的均值: [-2.9743062845566017, -0.13040943735125243, 0.8689813920975552, 9.251659261940047, 167.7144402927862]
# 组分和偏导的皮尔逊相关系数:
# Se: [-0.058914092189647674, -0.05897734154456809, -0.05893978291688205, -0.05814697162414292, -0.05969542987784592]
# Sb: [-0.1034299690784485, -0.10317672763617468, -0.10335281005009403, -0.10151114379836212, -0.11142638935436035]
# Cu: [0.01013351027304205, 0.010571402511852812, 0.010549056246004074, 0.009225411916509227, 0.006049631028526544]
# Bi: [0.10727466358258655, 0.10665221814497425, 0.10701359564142109, 0.10793451987112493, 0.1304144614318314]
# Pb: [-0.009378334140419985, -0.009870877825831936, -0.009762446788776507, -0.005144800642152968, 0.05326610556692424]
# Sn: [0.1316733809990721, 0.13211933611808505, 0.13133378407862714, 0.1239883719459693, 0.05563038715455915]
# Te: [-0.09255088856138621, -0.09320150326881373, -0.09271199504577575, -0.09158396735410564, -0.09058100750470931]
# Zn: [0.015049040655706078, 0.015624719468318185, 0.015692211520180188, 0.01650186327237839, 0.025082388781147626]
# 组分和能量密度的皮尔逊相关系数:
# Se: [0.12140044859606756, 0.12143280632456493, 0.12142495450400162, 0.12094065594641482, 0.1285576736078588]
# Sb: [0.18869405735467332, 0.18887696169328494, 0.18873521399922694, 0.1868892217018577, 0.13127712057945695]
# Cu: [0.12182675407736697, 0.12207708846592208, 0.12195272931164643, 0.12109971967985851, 0.0953303532346529]
# Bi: [-0.06950237250731875, -0.06958235954535459, -0.06964257331808779, -0.07084217809348518, -0.0742211762756477]
# Pb: [-0.2678157450262941, -0.26757573273730517, -0.2678085245110309, -0.2697600081840829, -0.2831239081295016]
# Sn: [-0.3098369660753396, -0.31051532962578154, -0.310035511818041, -0.3043896768285078, -0.23122116923664504]
# Te: [0.13581771991254998, 0.13579708277550015, 0.13577966485695392, 0.13583999614255307, 0.1580444601237529]
# Zn: [0.08277658377279604, 0.08282698696434923, 0.08283289087791212, 0.0810986321363167, 0.0672532606756499]
# ============================================
# 组分和偏导的Spearman系数:
# Se: [-0.04097230508757534, -0.040971405733345734, -0.04102702916559276, -0.04061339243967717, -0.017358547200014986]
# Sb: [-0.09554806977204924, -0.09476239557839049, -0.09517381208557488, -0.09200662028891683, -0.14392203926994984]
# Cu: [-0.019934038533199624, -0.019238292275078048, -0.019316624809747336, -0.020324753658404138, -0.03162383223306696]
# Bi: [0.06772354123141863, 0.0670448874478176, 0.06730441831631243, 0.06864236217114661, 0.09254694267612262]
# Pb: [-0.082023629114292, -0.0821059414003682, -0.08196461491087785, -0.07745083138813859, -0.00851886050625976]
# Sn: [0.14688339376893397, 0.14762080797916577, 0.14658624469688367, 0.13454239228249135, 0.02405894152454605]
# Te: [-0.04616862377464599, -0.047006774333529074, -0.04638524846617649, -0.04510769933726091, -0.02064530557222718]
# Zn: [-0.005896009701442908, -0.005520019699607006, -0.005415580480616702, -0.003434934769234431, 0.012169269960750411]
# 组分和能量密度的Spearman系数:
# Se: [0.10631805306386972, 0.10646793611119186, 0.10639763439854526, 0.10535048494421322, 0.10494630095403598]
# Sb: [0.20660800212815134, 0.2067356959970446, 0.20663589561901388, 0.2049484219217122, 0.1735608888092153]
# Cu: [0.12182701134353595, 0.12215655441705622, 0.12201082367348247, 0.1206750421288309, 0.09154247691743853]
# Bi: [-0.039768110453168304, -0.03996465610838857, -0.040000337466231904, -0.04137001036757878, -0.04270850526503392]
# Pb: [-0.23858430929223942, -0.23809220575911413, -0.238431196881221, -0.24101444537272737, -0.27896845860417396]
# Sn: [-0.2938451880342675, -0.29459517449221956, -0.29394758210541333, -0.28760940363499743, -0.23272310234294996]
# Te: [0.10644583183054016, 0.10637804193894126, 0.10640086779245805, 0.10629568453590166, 0.11667313575673495]
# Zn: [0.10846324374961909, 0.10855461330588304, 0.10855422614002237, 0.106780943071276, 0.10080234119938208]
# ============================================
# 偏导的皮尔逊系数:
# current delta = 1e-2, sorted components: Sn>Bi>Zn>Cu>Pb>Se>Te>Sb
# current delta = 1e-3, sorted components: Sn>Bi>Zn>Cu>Pb>Se>Te>Sb
# current delta = 1e-4, sorted components: Sn>Bi>Zn>Cu>Pb>Se>Te>Sb
# current delta = 1e-5, sorted components: Sn>Bi>Zn>Cu>Pb>Se>Te>Sb
# current delta = 1e-6, sorted components: Bi>Sn>Pb>Zn>Cu>Se>Te>Sb
# 能量密度的皮尔逊系数:
# current delta = 1e-2, sorted components: Sb>Te>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Sb>Te>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Sb>Te>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Sb>Te>Cu>Se>Zn>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Te>Sb>Se>Cu>Zn>Bi>Sn>Pb
# 偏导的Spearman系数:
# current delta = 1e-2, sorted components: Sn>Bi>Zn>Cu>Se>Te>Pb>Sb
# current delta = 1e-3, sorted components: Sn>Bi>Zn>Cu>Se>Te>Pb>Sb
# current delta = 1e-4, sorted components: Sn>Bi>Zn>Cu>Se>Te>Pb>Sb
# current delta = 1e-5, sorted components: Sn>Bi>Zn>Cu>Se>Te>Pb>Sb
# current delta = 1e-6, sorted components: Bi>Sn>Zn>Pb>Se>Te>Cu>Sb
# 能量密度的Spearman系数:
# current delta = 1e-2, sorted components: Sb>Cu>Zn>Te>Se>Bi>Pb>Sn
# current delta = 1e-3, sorted components: Sb>Cu>Zn>Se>Te>Bi>Pb>Sn
# current delta = 1e-4, sorted components: Sb>Cu>Zn>Te>Se>Bi>Pb>Sn
# current delta = 1e-5, sorted components: Sb>Cu>Zn>Te>Se>Bi>Pb>Sn
# current delta = 1e-6, sorted components: Sb>Te>Se>Zn>Cu>Bi>Sn>Pb
