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
            'bal': 0
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
    :return: 成功返回交易字典 {'txid', 'inputs', 'outputs', 'diameter'}
        UTXO用完返回 'Done'
    """
    tx_id = utils.generate_tx_id()
    # 1. 采样输入输出数量 (n, m)
    n, m = tx_sampler.sample(size=1)[0]

    # 统计当前所有中心地址的可用 UTXO 总数 (bal 之和)
    # 筛选条件：bal > 0 且 权重 > 0 (即 d_cur < d_tgt，度数未满)
    available_candidates_indices = [
        i for i, s in enumerate(central_states)
        if s['bal'] > 0 and (s['d_tgt'] - s['d_cur']) > 0
    ]

    # 计算总 UTXO 数量
    total_available_utxos = sum(central_states[i]['bal'] for i in available_candidates_indices)

    # 如果完全没有可用 UTXO，直接退出
    if total_available_utxos == 0:
        print(f"  [终止] 所有中心地址UTXO耗尽。")
        return 'Done'

    # 【关键优化】将 n 限制为总可用 UTXO 数，而不是地址数
    if n > total_available_utxos:
        print(f"  [调整] 采样输入 {n} > 可用UTXO {total_available_utxos}，调整 n = {total_available_utxos}")
        n = total_available_utxos

    # ---------------------------------------------------------
    # 2. 定义加权选择函数
    # ---------------------------------------------------------
    def select_inputs(count, banned_indices):
        """
        从 central_states 中选择 count 个输入，避开 banned_indices
        """
        # 选择count个地址
        selection = []
        for _ in range(count):
            candidates = []
            weights = []
            for idx, state in enumerate(central_states):
                # 1. 跳过黑名单
                if idx in banned_indices:
                    continue
                curr_bal = state['bal']
                curr_d_cur = state['d_cur']
                # 2. 扣除本轮已选带来的临时变更 (允许同一个地址在 selection 中出现多次，只要余额够)
                current_selection_count = selection.count(idx)
                curr_bal -= current_selection_count
                curr_d_cur += current_selection_count
                # 计算权重
                w = state['d_tgt'] - curr_d_cur
                # 3. 筛选：还有余额 且 还有度数配额
                if curr_bal > 0 and w > 0:
                    candidates.append(idx)
                    weights.append(w)
            # 没得选了，退出内层循环，能选多少选多少
            if not candidates:
                break
            # 加权采样
            weights = np.array(weights)
            probs = weights / weights.sum()
            chosen = np.random.choice(candidates, p=probs)
            selection.append(chosen)
        return selection

    # 生成接收地址 (Outputs) - 只需要生成一次
    outputs = [utils.generate_random_address() for _ in range(m)]

    # ---------------------------------------------------------
    # 3. 尝试构造与重试循环(如果构造交易超出直径限制则重新选择中心地址)
    # ---------------------------------------------------------
    banned_indices_for_this_tx = set()  # 本次交易生成的全局黑名单
    # 尝试选择发送地址（中心地址）构造交易
    while True:
        # 3.1 尝试选择 n 个输入 (中心地址)
        selected_central_indices = select_inputs(n, banned_indices_for_this_tx)
        # 中心地址由于直径限制都被ban了，没有可选的中心地址，无法构造交易，退出循环
        if len(selected_central_indices) == 0:
            print('没有中心地址可供选择')
            return None
        # 3.2 直径验证
        inputs = [central_states[i]['address'] for i in selected_central_indices]
        graph.add_transaction(tx_id, inputs, outputs)
        current_diameter = graph.calculate_diameter()
        if current_diameter <= diameter_threshold:
            # === 成功 ===
            print(f"  [成功] 交易 {tx_id} 直径 {current_diameter}")
            # 提交状态更新
            for idx in selected_central_indices:
                central_states[idx]['bal'] -= 1
                central_states[idx]['d_cur'] += 1
            return {
                'txid': tx_id,
                'inputs': inputs,
                'outputs': outputs,
                'diameter': current_diameter
            }
        else:
            # === 失败 (直径超标) ===
            # 1. 回滚图状态
            graph.remove_transaction(tx_id)

            # 查找接收地址中存在的导致直径超标的地址
            # TODO：与输入也有关，比如输入两个输出一个，那么也会导致超标，理论上不超标的选择方式只有单输入单输出成环
            def find_harmful_senders(tx_id, selected_indices, outputs, graph, diameter_threshold):
                harmful = set()
                for idx in selected_indices:
                    # 只保留这个接收地址
                    test_input = [central_states[idx]['address']]
                    graph.add_transaction(tx_id, test_input, outputs)
                    d = graph.calculate_diameter()
                    graph.remove_transaction(tx_id)
                    # 如果单独使用它仍然超标，说明它本身就有问题
                    if d > diameter_threshold:
                        harmful.add(idx)
                return harmful

            harmful = find_harmful_senders(tx_id, selected_central_indices, outputs, graph, diameter_threshold)
            print(f"  [重试] 直径 {current_diameter} > {diameter_threshold}。拉黑 {len(harmful)} 个地址")
            banned_indices_for_this_tx.update(harmful)
            # 3. 循环继续 -> 下一次循环会调用 select_inputs，自动避开刚才这批人
        # 达到最大重试次数仍未成功
    return None


def construct_message_communication_transaction(tx_sampler, comm_addresses, central_states, graph, diameter_threshold):
    """
    构造消息通信交易，通信地址->中心地址
    :param tx_sampler: TxInOutSampler 实例
    :param comm_addresses: 可用的通信地址列表 (list of str)，作为资金来源
    :param central_states: 中心地址状态列表，作为接收方
    :param graph: 交易图实例
    :param diameter_threshold: 直径阈值
    :return: 成功返回交易字典，失败返回 None
    发送完毕返回'Done'
    """
    tx_id = utils.generate_tx_id()
    # 1. 采样输入输出数量 (n, m)
    n, m = tx_sampler.sample(size=1)[0]
    # 2. 选择发送地址
    # 检查可用通信地址是否足够，如果通信地址数量少于n，则将剩余通信地址用完即可
    available_comm_count = len(comm_addresses)
    # 适配输入数量
    if n > available_comm_count:
        n = available_comm_count

    # 从可用列表中随机选择 n 个索引 (无放回)
    selected_comm_indices = np.random.choice(available_comm_count, n, replace=False)
    inputs = [comm_addresses[i] for i in selected_comm_indices]

    # ---------------------------------------------------------
    # 2. 定义选择接收方函数 (内部复用)
    # ---------------------------------------------------------
    def select_outputs(count, banned_indices):
        """
        选择 count 个中心地址作为接收方，避开黑名单
        """
        selection = []
        for _ in range(count):
            candidates = []
            weights = []
            for idx, state in enumerate(central_states):
                # 1. 跳过黑名单
                if idx in banned_indices:
                    continue
                curr_d_cur = state['d_cur']
                # 2. 叠加本轮已选带来的变更
                curr_d_cur += selection.count(idx)
                # 计算权重
                w = state['d_tgt'] - curr_d_cur
                # 3. 筛选：只要还有度数配额就能接收
                if w > 0:
                    candidates.append(idx)
                    weights.append(w)
            # 没得选了，退出内层循环，能选多少选多少
            if not candidates:
                break
            # 加权采样
            weights = np.array(weights)
            probs = weights / weights.sum()
            chosen = np.random.choice(candidates, p=probs)
            selection.append(chosen)
        return selection

    # ---------------------------------------------------------
    # 3. 尝试构造与重试循环
    # ---------------------------------------------------------
    banned_central_indices = set()  # 本次交易针对输出端的黑名单

    # 尝试选择接收地址（中心地址）构造交易
    while True:
        # 3.1 加权 m 个接收方 (中心地址)
        selected_central_indices = select_outputs(m, banned_central_indices)
        # 中心地址由于直径限制都被ban了，没有可选的中心地址，无法构造交易，退出循环
        if len(selected_central_indices) == 0:
            print('没有中心地址可供选择')
            break
        outputs = [central_states[i]['address'] for i in selected_central_indices]
        # 3.2 直径验证
        graph.add_transaction(tx_id, inputs, outputs)
        current_diameter = graph.calculate_diameter()
        if current_diameter <= diameter_threshold:
            # === 成功 ===
            print(f"  [成功] 交易 {tx_id} 直径 {current_diameter}")
            # 1. 更新中心地址状态 (接收方: bal+1, d_cur+1)
            for idx in selected_central_indices:
                central_states[idx]['bal'] += 1
                central_states[idx]['d_cur'] += 1
            # 2. 移除已使用的通信地址，从后往前删，防止索引错位
            for idx in sorted(selected_comm_indices, reverse=True):
                del comm_addresses[idx]
            return {
                'txid': tx_id,
                'inputs': inputs,
                'outputs': outputs,
                'diameter': current_diameter
            }
        else:
            # === 失败 ===
            # 1. 回滚图状态
            graph.remove_transaction(tx_id)

            # 查找接收地址中存在的导致直径超标的地址
            # TODO：与输入也有关，比如输入两个输出一个，那么也会导致超标，理论上不超标的选择方式只有单输入单输出成环
            def find_harmful_receivers(tx_id, inputs, selected_indices, graph, diameter_threshold):
                harmful = set()
                for idx in selected_indices:
                    # 只保留这个接收地址
                    test_output = [central_states[idx]['address']]
                    graph.add_transaction(tx_id, inputs, test_output)
                    d = graph.calculate_diameter()
                    graph.remove_transaction(tx_id)
                    # 如果单独使用它仍然超标，说明它本身就有问题
                    if d > diameter_threshold:
                        harmful.add(idx)
                return harmful

            harmful = find_harmful_receivers(tx_id, inputs, selected_central_indices, graph, diameter_threshold)
            print(f"  [重试] 直径 {current_diameter} > {diameter_threshold}。拉黑 {len(harmful)} 个接收地址...")
            banned_central_indices.update(harmful)
            # 3. 循环继续 -> 重新 select_outputs
    return None


def init_parameter():
    global N_utxo, txInOutSampler, D_Thres, central_addresses_state, btg
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
    central_addresses_state = init_center_addresses(N_center, N_comm, addrCentralDegreeSampler)
    central_addresses_state[0]['bal'] = 5
    btg = BitcoinTransactionGraph()


if __name__ == "__main__":
    init_parameter()

    # 模拟主循环 (多轮次)
    # NUM_ROUNDS = 3  # 模拟轮数
    round_idx = 0
    # 如果 N_utxo > 0 说明消息为传输完成（消息传输还需要N_utxo个交易），则继续下一轮
    while N_utxo > 0:
        print(f"\n{'=' * 20} 开始第 {round_idx + 1} 轮模拟 {'=' * 20}")
        # -------------------------------------------------
        # Phase 1: 代币分配阶段 (Token Distribution)
        # 中心地址 -> 通信地址
        # -------------------------------------------------
        print(f"\n[Phase 1] 代币分配: 直到中心地址余额耗尽...")
        round_comm_addresses = []  # 本轮产生的所有通信地址
        dist_tx_count = 0
        # 如果 N_utxo > 0 说明仍需要utxo分配，继续创建代币分配交易
        while N_utxo > 0:
            # 尝试构造交易
            tx = construct_token_distribute_transaction(
                txInOutSampler,
                central_addresses_state,
                btg,
                D_Thres,
            )
            # 所有中心地址UTXO耗尽。
            if tx == 'Done':
                break
            elif tx:
                # 成功
                dist_tx_count += 1
                round_comm_addresses.extend(tx['outputs'])
                # 更新剩余待分配的utxo数量
                N_utxo -= len(tx['outputs'])
            else:
                print(f" 第 {round_idx + 1} 轮模拟 Phase 1 失败。")
                break
        print(f"Phase 1 结束。共生成 {dist_tx_count} 笔交易，产生了 {len(round_comm_addresses)} 个通信地址 UTXO。")
        if dist_tx_count == 0:
            break
        # -------------------------------------------------
        # Phase 2: 消息通信阶段 (Message Communication)
        # 通信地址 -> 中心地址
        # -------------------------------------------------
        print(f"\n[Phase 2] 消息通信: 直到通信地址消耗殆尽...")
        comm_tx_count = 0
        # 只要还有通信地址可用，就继续生成
        while len(round_comm_addresses) > 0:
            # 1. 尝试构造交易
            tx = construct_message_communication_transaction(
                txInOutSampler,
                round_comm_addresses,
                central_addresses_state,
                btg,
                D_Thres,
            )
            if tx:
                comm_tx_count += 1
                fail_count = 0
                if comm_tx_count % 100 == 0:
                    print(f"  已生成 {comm_tx_count} 笔通信交易, 剩余通信地址: {len(round_comm_addresses)}")
            else:
                print(f" 第 {round_idx + 1} 轮模拟 Phase 2 失败。")
                break
        print(f"Phase 2 结束。共生成 {comm_tx_count} 笔交易，中心地址已回收资金。")
        round_idx += 1

    btg.get_graph_info()
    print(btg.calculate_diameter())
    btg.visualize()
