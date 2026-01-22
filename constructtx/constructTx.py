import numpy as np
from matplotlib import pyplot as plt

from sampleaddressdegree import AddressCentralDegreeSampler
from plotgraph.main import BitcoinTransactionGraph
from sampletxinout import TxInOutSampler
from samplediameter import DiameterSampler


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


if __name__ == "__main__":
    txInOutSampler = init_TxInOutSampler()
    Nutxo = calculate_utxos(1024 * 8)  # 1kB
    En, Em = calculate_txinout_expectations(txInOutSampler)
    N1 = Nutxo / Em
    N2 = Nutxo / En
    Dtotal = (N1 * En + N2 * Em) * 1.2
    addrCentralDegreeSampler = init_AddressCentralDegreeSampler()
    Ed = calculate_addresscentrality_expectations(addrCentralDegreeSampler)
    Ncomm = Nutxo
    Ncenter = (Dtotal + (2 - Ed) * Ncomm) / Ed
    Size = N1 + N2 + Ncomm + Ncenter
    diameterSampler = init_DiameterSampler()
    DThres = diameterSampler.sample(Size,1)
    print(DThres)
