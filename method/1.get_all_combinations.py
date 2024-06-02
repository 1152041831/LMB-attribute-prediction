import itertools

import numpy as np


def get_all_combinations():
    current_density_range = list(range(0, 1001, 100))
    current_density_range = [c for c in current_density_range]
    print("电流密度选取范围:",current_density_range) # [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # 计算组分可能的值，每个组分都是0.1的倍数
    component_values = [round(i * 0.1, 1) for i in range(11)]
    print("组分选取范围:",component_values) # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combinations = []
    # for c1 in component_values:
    #     for c2 in component_values:
    #         for c3 in component_values:
    #             for c4 in component_values:
    #                 for c5 in component_values:
    #                     for c6 in component_values:
    #                         for c7 in component_values:
    #                             for c8 in component_values:
    #                                 if abs((c1+c2+c3+c4+c5+c6+c7+c8)-1.0) < 1e-3:
    #                                     for d in current_density_range:
    #                                         # 这里电流密度d除以100是因为训练模型时的数据集的电流密度也除去了100，保持输入格式一致
    #                                         now_com = [c1,c2,c3,c4,c5,c6,c7,c8,d/100]
    #                                         # if now_com not in combinations:
    #                                         combinations.append(now_com)
    for combination in itertools.product(component_values, repeat=8):
        c1, c2, c3, c4, c5, c6, c7, c8 = combination
        if abs((c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8) - 1.0) < 1e-3:
            for d in current_density_range:
                # 这里电流密度d除以100是因为训练模型时的数据集的电流密度也除去了100，保持输入格式一致
                now_com = [c1, c2, c3, c4, c5, c6, c7, c8, d / 100]
                combinations.append(now_com)

    combinations = np.array(combinations)
    print(combinations.shape)
    np.savetxt('combinations.csv', combinations, delimiter=',')


get_all_combinations()
