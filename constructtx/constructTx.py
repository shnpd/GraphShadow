import numpy as np
from matplotlib import pyplot as plt

from sampleaddressdegree import AddressCentralDegreeSampler
from plotgraph.main import BitcoinTransactionGraph
from sampletxinout import TxInOutSampler
from samplediameter import DiameterSampler
from utils import generate_random_address


def init_TxInOutSampler():
    # 离散分布
    discrete_date = {(1, 2): 15796, (1, 1): 11722}

    # 连续分布
    continuous_params = {
        'A': 391696615.17,
        'alpha': 14.04, 'beta': 17.33,
        'lam1': 1.7308, 'lam2': 3.1599, 'lam3': 12.5264
    }

    # 离散分布权重
    lambda_mix = (15796 + 11722) / (15796 + 11722 + 3394)
    sampler = TxInOutSampler(discrete_date, continuous_params, lambda_mix, max_range=5)
    return sampler


def init_AddressCentralDegreeSampler():
    params = {
        'A': 397541.13,
        'alpha': -5.39
    }

    sampler = AddressCentralDegreeSampler(params, min_x=2, max_x=100)

    return sampler


def init_DiameterSampler():
    csv_file_path = "path_length_probability_matrix.xlsx"
    diameter_sampler = DiameterSampler(csv_file_path)
    return diameter_sampler


def calculate_utxos(msgSizePerBit):
    return msgSizePerBit / 256


def calculate_txinout_expectations(sampler):
    """
    计算交易输入输出数量的数学期望
    :param sampler: 已经初始化的 TxInOutSampler 实例
    :return: 元组 (expected_n_inputs, expected_m_outputs)
    """
    # 1. 计算离散部分 (Discrete Part) 的期望
    # 提取 n 和 m 的列表
    # discrete_points 结构如: [(1, 1), (1, 2), ...]
    disc_n_vals = np.array([p[0] for p in sampler.discrete_points])
    disc_m_vals = np.array([p[1] for p in sampler.discrete_points])
    # 计算加权和: sum( value * prob )
    e_n_disc = np.sum(disc_n_vals * sampler.discrete_probs)
    e_m_disc = np.sum(disc_m_vals * sampler.discrete_probs)
    # 2. 计算连续部分的期望
    p_n_marginal = np.sum(sampler.cont_probs_grid, axis=1)
    p_m_marginal = np.sum(sampler.cont_probs_grid, axis=0)
    # 计算期望: sum( value * marginal_prob )
    # cont_n_vals / cont_m_vals 是我们在类中保存的坐标轴数值 [1, 2, ..., max_range]
    e_n_cont = np.sum(sampler.cont_n_vals * p_n_marginal)
    e_m_cont = np.sum(sampler.cont_m_vals * p_m_marginal)
    # 3. 混合两部分 (Mixture)
    lam = sampler.lambda_mix
    expected_n = lam * e_n_disc + (1 - lam) * e_n_cont
    expected_m = lam * e_m_disc + (1 - lam) * e_m_cont
    return expected_n, expected_m


def calculate_addresscentrality_expectations(sampler):
    """
    计算地址中心度的数学期望 (平均中心度)
    :param sampler: 已经初始化的 AddressCentralDegreeSampler (或 CentralitySampler) 实例
    :return: 平均中心度 (float)
    """
    expected_centrality = np.sum(sampler.x_vals * sampler.probs)
    return expected_centrality


def init_center_addresses(N_center, N_comm, centrality_sampler):
    """
        构造中心地址集并初始化状态

        :param N_center: 中心地址总数
        :param N_comm: 通信地址数量 (配额)
        :param centrality_sampler: 初始化的 AddressCentralDegreeSampler 实例
        :return: 中心地址状态列表
        """
    # 1. 生成地址 (使用 set 确保唯一性，防止极其罕见的碰撞)
    central_address_list = []
    while len(central_address_list) < N_center:
        central_address_list.append(generate_random_address())

    # 2. 采样目标中心度
    # 直接采样 N_center 个数据
    central_address_degree_list = []
    while len(central_address_degree_list) < N_center:
        sample_degree = int(centrality_sampler.sample()[0])
        # 采样为2的中心度优先分配给通信地址
        if sample_degree == 2 and N_comm > 0:
            N_comm -= 1
        else:
            central_address_degree_list.append(sample_degree)

    # 3. 初始化中心地址状态集合
    central_address_states = []
    for i in range(len(central_address_degree_list)):
        state = {
            'address': central_address_list[i],
            'd_cur': 0,
            'd_tgt': central_address_degree_list[i],
            'bal': 1
        }
        central_address_states.append(state)
    return central_address_states


if __name__ == "__main__":
    #  normal
    # 计算utxo数量
    N_utxo = calculate_utxos(1024 * 8)  # 1kB
    # 计算交易期望输入输出数量
    txInOutSampler = init_TxInOutSampler()
    En, Em = calculate_txinout_expectations(txInOutSampler)
    # 计算交易数量
    N_1 = N_utxo / Em
    N_2 = N_utxo / En
    # 计算中心地址集总度数需求
    D_total = (N_1 * En + N_2 * Em) * 1.2
    # 计算地址平均度
    addrCentralDegreeSampler = init_AddressCentralDegreeSampler()
    Ed = calculate_addresscentrality_expectations(addrCentralDegreeSampler)
    # 计算地址数量
    N_comm = N_utxo
    N_center = (D_total + (2 - Ed) * N_comm) / Ed
    # 计算交易图直径阈值
    Size = N_1 + N_2 + N_comm + N_center
    diameterSampler = init_DiameterSampler()
    D_Thres = diameterSampler.sample(Size, 1)
    # 初始化中心地址集状态
    Central_addresses_state = init_center_addresses(N_center, N_comm, addrCentralDegreeSampler)
    print(Central_addresses_state)