import numpy as np

def cal_neighbor_C(g, C_ratio_list, index_own):
    neighbor_list = np.array(list(g.neighbors(index_own)))

    ratio_list = C_ratio_list[neighbor_list, 0]
    total_number_list = C_ratio_list[neighbor_list, 1]
    number_C = np.sum(np.multiply(ratio_list, total_number_list))
    total_number = np.sum(total_number_list)

    return number_C / total_number

def update_C_ratio_list(C_ratio_list, index1, index2, action_1_str, action_2_str):
    if action_1_str == "C":
        new_ratio_1 = (C_ratio_list[index1, 0] * C_ratio_list[index1, 1] + 1) / (C_ratio_list[index1, 1] + 1)
    else:
        new_ratio_1 = (C_ratio_list[index1, 0] * C_ratio_list[index1, 1]) / (C_ratio_list[index1, 1] + 1)

    if action_2_str == "C":
        new_ratio_2 = (C_ratio_list[index2, 0] * C_ratio_list[index2, 1] + 1) / (C_ratio_list[index2, 1] + 1)
    else:
        new_ratio_2 = (C_ratio_list[index2, 0] * C_ratio_list[index2, 1]) / (C_ratio_list[index2, 1] + 1)

    C_ratio_list[index1, 0] = new_ratio_1
    C_ratio_list[index1, 1] += 1

    C_ratio_list[index2, 0] = new_ratio_2
    C_ratio_list[index2, 1] += 1

    return C_ratio_list

C_test = np.array([[0.5, 6], [0.3, 10]])

print(update_C_ratio_list(C_test, 0, 1, "C", "C"))