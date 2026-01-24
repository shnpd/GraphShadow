import numpy as np
from matplotlib import pyplot as plt

from sampleaddressdegree import AddressCentralDegreeSampler
from txgraph.main import BitcoinTransactionGraph
from sampletxinout import TxInOutSampler
from samplediameter import DiameterSampler
import utils


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
        central_address_list.append(utils.generate_random_address())

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


def construct_token_distribute_transaction(tx_sampler, central_states, graph, diameter_threshold):
    """
    构造代币分配交易，中心地址->通信地址
    :param tx_sampler: TxInOutSampler 实例，用于采样 n, m
    :param central_states: 中心地址状态列表 (list of dict)，将被原地修改
                           结构: [{'address':.., 'bal':.., 'd_cur':.., 'd_tgt':..}, ...]
    :param graph: BitcoinTransactionGraph 实例，需支持 add/remove_transaction 和 calculate_diameter
    :param diameter_threshold: 图直径阈值 (int)
    :return: 成功返回交易字典 {'txid', 'inputs', 'outputs', 'diameter'}，失败返回 None
    """
    tx_id = utils.generate_tx_id()
    # 1. 采样输入输出数量 (n, m)
    n, m = tx_sampler.sample(size=1)[0]

    # 2. 筛选与动态加权选择发送地址 (Inputs)
    selected_indices = []

    # 用于记录本轮交易中状态的临时变更 (index, delta_bal, delta_d_cur)
    # 目的：在循环选择 n 个地址时，后续的选择必须感知到前面的选择导致的状态变化
    # 但在直径检查通过前，不能真正修改 central_states
    temp_changes = []

    # 选择n个中心地址作为发送地址
    for _ in range(n):
        candidates = []
        weights = []

        # 遍历所有中心地址计算权重
        for idx, state in enumerate(central_states):
            # 获取当前的基础状态
            curr_bal = state['bal']
            curr_d_cur = state['d_cur']

            # 叠加本轮已发生的临时变更
            for c_idx, d_bal, d_d_cur in temp_changes:
                if c_idx == idx:
                    curr_bal += d_bal
                    curr_d_cur += d_d_cur

            # 后选地址余额必须 > 0
            if curr_bal > 0:
                # 计算权重 W = d_tgt - d_cur
                w = state['d_tgt'] - curr_d_cur
                candidates.append(idx)
                weights.append(w)

        # 如果候选集为空（没有钱了），则本笔交易无法构建
        if not candidates:
            print(f"  [失败] 交易 {tx_id}: 可用中心地址不足，无法凑齐 {n} 个输入")
            return None

        # 归一化权重
        weights = np.array(weights)
        probs = weights / weights.sum()

        # 加权采样一个地址索引
        chosen_idx = np.random.choice(candidates, p=probs)
        selected_indices.append(chosen_idx)

        # 记录临时变更：选中后 bal-1, d_cur+1
        temp_changes.append((chosen_idx, -1, 1))

    # 3. 生成接收地址
    # 提取发送地址字符串
    inputs = [central_states[i]['address'] for i in selected_indices]
    # 生成新的接收地址
    outputs = [utils.generate_random_address() for _ in range(m)]

    # 4. 图更新与直径检查 (Check)
    # 4.1 预添加交易到图中 (Snapshot 思想)
    graph.add_transaction(tx_id, inputs, outputs)
    # 4.2 计算添加后的图直径
    current_diameter = graph.calculate_diameter()
    # 4.3 判断阈值
    if current_diameter > diameter_threshold:
        # print(f"  [回滚] 交易 {tx_id} 直径 {current_diameter} > 阈值 {diameter_threshold}")
        graph.remove_transaction(tx_id)
        return None

    else:
        # print(f"  [成功] 交易 {tx_id} 直径 {current_diameter}")
        # 【关键】真正提交状态更新到 central_states
        for idx, d_bal, d_d_cur in temp_changes:
            central_states[idx]['bal'] += d_bal  # 余额减少
            central_states[idx]['d_cur'] += d_d_cur  # 当前度增加
        return {
            'txid': tx_id,
            'inputs': inputs,
            'outputs': outputs,
        }


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
    btg = BitcoinTransactionGraph()
    for _ in range(10):
        construct_token_distribute_transaction(txInOutSampler, Central_addresses_state,btg, D_Thres)
    # btg.get_graph_info()
    btg.visualize()
