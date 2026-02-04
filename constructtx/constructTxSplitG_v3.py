import json
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from constructtx.sampleaddressdegree import AddressCentralDegreeSampler
from txgraph.main import BitcoinTransactionGraph
from constructtx.sampletxinout import TxInOutSampler
from constructtx.samplediameter import DiameterSampler
import constructtx.utils as utils
import pandas as pd
import re

# ==========================================
# 采样器初始化与辅助函数 (保持不变)
# ==========================================

def init_TxInOutSampler():
    discrete_date = {(1, 2): 15796, (1, 1): 11722}
    continuous_params = {
        'A': 391696615.17,
        'alpha': 14.04, 'beta': 17.33,
        'lam1': 1.7308, 'lam2': 3.1599, 'lam3': 12.5264
    }
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

def calculate_utxos(msg_size_B):
    return msg_size_B * 8 / 256

def calculate_txinout_expectations(sampler):
    disc_n_vals = np.array([p[0] for p in sampler.discrete_points])
    disc_m_vals = np.array([p[1] for p in sampler.discrete_points])
    e_n_disc = np.sum(disc_n_vals * sampler.discrete_probs)
    e_m_disc = np.sum(disc_m_vals * sampler.discrete_probs)
    
    p_n_marginal = np.sum(sampler.cont_probs_grid, axis=1)
    p_m_marginal = np.sum(sampler.cont_probs_grid, axis=0)
    e_n_cont = np.sum(sampler.cont_n_vals * p_n_marginal)
    e_m_cont = np.sum(sampler.cont_m_vals * p_m_marginal)
    
    lam = sampler.lambda_mix
    expected_n = lam * e_n_disc + (1 - lam) * e_n_cont
    expected_m = lam * e_m_disc + (1 - lam) * e_m_cont
    return expected_n, expected_m

def calculate_addresscentrality_expectations(sampler):
    expected_centrality = np.sum(sampler.x_vals * sampler.probs)
    return expected_centrality

# ==========================================
# 核心修改 1: 初始化中心地址时分配 Group ID
# ==========================================
def init_center_addresses(N_center, N_comm, centrality_sampler, num_groups=8):
    """
    构造中心地址集并初始化状态，增加 group_id 实现分量隔离
    :param num_groups: 期望的分量（帮派）数量，默认8个
    """
    central_address_list = []
    while len(central_address_list) < N_center:
        central_address_list.append(utils.generate_random_address())

    central_address_degree_list = []
    while len(central_address_degree_list) < N_center:
        sample_degree = int(centrality_sampler.sample()[0])
        if sample_degree == 2 and N_comm > 0:
            N_comm -= 1
        else:
            central_address_degree_list.append(sample_degree)

    central_address_states = []
    for i in range(len(central_address_degree_list)):
        state = {
            'address': central_address_list[i],
            'd_cur': 0,
            'd_tgt': central_address_degree_list[i],
            'bal': 0,
            # 【新增】均匀分配组 ID (0, 1, ..., num_groups-1)
            'group_id': i % num_groups
        }
        central_address_states.append(state)
    return central_address_states

# ==========================================
# 阈值获取函数 (保持不变)
# ==========================================
def get_window_size(diameter, excel_path="constructtx/直径窗口映射表_精简版.xlsx"):
    try:
        df = pd.read_excel(excel_path)
        thresholds = []
        windows = []
        for _, row in df.iterrows():
            interval_str = str(row['直径区间 (D)'])
            match = re.findall(r'\d+', interval_str)
            if match:
                thresholds.append(int(match[-1]))
                windows.append(row['推荐窗口大小 (W)'])
        for i, t in enumerate(thresholds):
            if diameter <= t:
                return windows[i]
        return f"Overflow (>{thresholds[-1]})"
    except Exception as e:
        # Fallback if file not found
        return 5

def get_diameter_threshold_by_window(target_window, excel_path="constructtx/直径窗口映射表_精简版.xlsx"):
    try:
        df = pd.read_excel(excel_path)
        window_to_threshold = {}
        valid_windows = []
        for _, row in df.iterrows():
            try:
                w_val = int(row['推荐窗口大小 (W)'])
            except: continue
            interval_str = str(row['直径区间 (D)'])
            match = re.findall(r'\d+', interval_str)
            if match:
                window_to_threshold[w_val] = int(match[-1])
                valid_windows.append(w_val)
        valid_windows.sort()
        if not valid_windows: return 10
        if target_window < valid_windows[0]: return window_to_threshold[valid_windows[0]]
        if target_window in window_to_threshold: return window_to_threshold[target_window]
        last_valid_w = valid_windows[0]
        for w in valid_windows:
            if w < target_window: last_valid_w = w
            else: break
        return window_to_threshold[last_valid_w]
    except Exception:
        return 10  # Fallback threshold

# ==========================================
# 核心修改 2: Phase 1 严格按组发币
# ==========================================
def construct_token_distribute_transaction(tx_sampler, central_states, graph, start_height, comm_group_map):
    """
    构造代币分配交易 (Best-of-N + 严格分组)
    :param comm_group_map: 全局字典，用于记录通信地址属于哪个组
    """
    tx_id = utils.generate_tx_id()
    n, m = tx_sampler.sample(size=1)[0]
    
    # 1. 随机锁定一个"有钱的帮派"
    groups_with_money = set(s['group_id'] for s in central_states if s['bal'] > 0)
    if not groups_with_money:
        return 'Done'
    
    # 随机选择一个组 ID
    target_group_id = np.random.choice(list(groups_with_money))
    
    # 修正 n: 不能超过该组的总余额
    group_balance = sum(s['bal'] for s in central_states if s['group_id'] == target_group_id)
    if n > group_balance:
        n = group_balance

    # 2. 定义输入选择函数 (增加了 group_id 过滤)
    def select_inputs(count, banned_indices):
        selection = []
        for _ in range(count):
            candidates = []
            weights = []
            for idx, state in enumerate(central_states):
                # 【约束】非本组勿扰
                if state['group_id'] != target_group_id:
                    continue
                if idx in banned_indices:
                    continue
                
                curr_bal = state['bal'] - selection.count(idx)
                if curr_bal > 0:
                    candidates.append(idx)
                    weights.append(state['d_tgt']) # 组内仍可按度数加权
            
            if not candidates: break
            
            weights = np.array(weights)
            probs = weights / weights.sum() if weights.sum() > 0 else None
            chosen = np.random.choice(candidates, p=probs)
            selection.append(chosen)
        return selection

    # 3. Best-of-N 构造循环
    banned_indices_for_this_tx = set()
    output_addrs = [utils.generate_random_address() for _ in range(m)]
    current_diameter = -1
    CANDIDATE_ATTEMPTS = 5

    while True:
        best_indices = []
        min_temp_diameter = float('inf')
        
        for _ in range(CANDIDATE_ATTEMPTS):
            temp_indices = select_inputs(n, banned_indices_for_this_tx)
            if len(temp_indices) < n: continue

            temp_input_addrs = [central_states[i]['address'] for i in temp_indices]
            
            # 试探
            graph.add_transaction(tx_id, temp_input_addrs, output_addrs)
            temp_d = graph.calculate_diameter()
            graph.remove_transaction(tx_id)
            
            if temp_d < min_temp_diameter:
                min_temp_diameter = temp_d
                best_indices = temp_indices
        
        selected_central_indices = best_indices
        
        if len(selected_central_indices) == 0:
            windows = get_window_size(current_diameter)
            print(f'直径超标或选不出地址，等待新区块{start_height+windows}！')
            return 'Exceed'

        input_addrs = [central_states[i]['address'] for i in selected_central_indices]
        graph.add_transaction(tx_id, input_addrs, output_addrs)
        current_diameter = graph.calculate_diameter()
        
        now_height = 934217 # TODO: 获取真实高度
        diameter_thres = get_diameter_threshold_by_window(now_height - start_height)
        
        if current_diameter <= diameter_thres:
            print(f"  [成功-Group{target_group_id}] Tx {tx_id} 直径 {current_diameter}")
            
            # 更新余额
            for idx in selected_central_indices:
                central_states[idx]['bal'] -= 1
            
            # 【核心记录】记录生成的通信地址归属
            for out_addr in output_addrs:
                comm_group_map[out_addr] = target_group_id
                
            return {'hash': tx_id, 'input_addrs': input_addrs, 'output_addrs': output_addrs, 'diameter': current_diameter}
        else:
            graph.remove_transaction(tx_id)
            banned_indices_for_this_tx.update(selected_central_indices)

# ==========================================
# 核心修改 3: Phase 2 严格按组回款
# ==========================================
def construct_message_communication_transaction(tx_sampler, comm_addresses, central_states, graph, start_height, comm_group_map):
    """
    构造消息通信交易 (Best-of-N + 严格分组)
    :param comm_group_map: 用于查找输入地址归属的组
    """
    tx_id = utils.generate_tx_id()
    n, m = tx_sampler.sample(size=1)[0]
    
    available_comm = len(comm_addresses)
    if available_comm == 0: return 'Done'

    # 1. 确定本次交易的归属组 (通过随机选一个种子地址)
    seed_idx = np.random.choice(available_comm)
    seed_addr = comm_addresses[seed_idx]
    
    target_group_id = comm_group_map.get(seed_addr)
    # 如果找不到组信息(异常情况)，跳过
    if target_group_id is None: return 'Done'
    
    # 2. 筛选同组输入
    candidate_indices = [i for i, addr in enumerate(comm_addresses) if comm_group_map.get(addr) == target_group_id]
    n = min(n, len(candidate_indices))
    selected_comm_indices = np.random.choice(candidate_indices, n, replace=False)
    input_addrs = [comm_addresses[i] for i in selected_comm_indices]

    # 3. 定义输出选择函数 (严格回流到本组中心地址)
    def select_outputs(count, banned_indices):
        selection = []
        for _ in range(count):
            candidates = []
            weights = []
            for idx, state in enumerate(central_states):
                # 【约束】必须回流到本组
                if state['group_id'] != target_group_id:
                    continue
                if idx in banned_indices:
                    continue
                
                candidates.append(idx)
                weights.append(state['d_tgt'])
            
            if not candidates: break
            
            weights = np.array(weights)
            probs = weights / weights.sum() if weights.sum() > 0 else None
            chosen = np.random.choice(candidates, p=probs)
            selection.append(chosen)
        return selection

    # 4. Best-of-N 构造循环
    banned_central_indices = set()
    current_diameter = -1
    CANDIDATE_ATTEMPTS = 5
    
    while True:
        best_indices = []
        min_temp_diameter = float('inf')
        
        for _ in range(CANDIDATE_ATTEMPTS):
            temp_indices = select_outputs(m, banned_central_indices)
            if len(temp_indices) < m: continue
            
            temp_output_addrs = [central_states[i]['address'] for i in temp_indices]
            
            graph.add_transaction(tx_id, input_addrs, temp_output_addrs)
            temp_d = graph.calculate_diameter()
            graph.remove_transaction(tx_id)
            
            if temp_d < min_temp_diameter:
                min_temp_diameter = temp_d
                best_indices = temp_indices
        
        selected_central_indices = best_indices
        if len(selected_central_indices) == 0:
            windows = get_window_size(current_diameter)
            print(f'直径超标或无可选地址，等待新区块{start_height+windows}！')
            return 'Exceed'
            
        output_addrs = [central_states[i]['address'] for i in selected_central_indices]
        
        graph.add_transaction(tx_id, input_addrs, output_addrs)
        current_diameter = graph.calculate_diameter()
        
        now_height = 934217 
        diameter_thres = get_diameter_threshold_by_window(now_height - start_height)
        
        if current_diameter <= diameter_thres:
            print(f"  [成功-Group{target_group_id}] Tx {tx_id} 直径 {current_diameter}")
            
            # 更新状态
            for idx in selected_central_indices:
                central_states[idx]['bal'] += 1
            
            # 移除已使用的通信地址和映射记录
            for idx in sorted(selected_comm_indices, reverse=True):
                addr_to_del = comm_addresses[idx]
                if addr_to_del in comm_group_map:
                    del comm_group_map[addr_to_del]
                del comm_addresses[idx]
                
            return {'hash': tx_id, 'input_addrs': input_addrs, 'output_addrs': output_addrs, 'diameter': current_diameter}
        else:
            graph.remove_transaction(tx_id)
            banned_central_indices.update(selected_central_indices)

# ==========================================
# 参数初始化与主程序
# ==========================================
def init_parameter():
    global N_utxo, txInOutSampler, D_Thres, central_addresses_state, btg, comm_group_map
    
    # 1. 计算参数
    N_utxo = calculate_utxos(1024 * 8)
    txInOutSampler = init_TxInOutSampler()
    En, Em = calculate_txinout_expectations(txInOutSampler)
    N_1 = N_utxo / Em
    N_2 = N_utxo / En
    D_total = (N_1 * En + N_2 * Em) * 1.2
    
    addrCentralDegreeSampler = init_AddressCentralDegreeSampler()
    Ed = calculate_addresscentrality_expectations(addrCentralDegreeSampler)
    
    N_comm = N_utxo
    N_center = (D_total + (2 - Ed) * N_comm) / Ed
    
    # 2. 初始化中心地址 (指定分为 8 个组)
    # 这样最终图会呈现 8 个独立的连通分量
    central_addresses_state = init_center_addresses(N_center, N_comm, addrCentralDegreeSampler, num_groups=9)
    
    # 手动设置初始余额 (示例)
    for i in range(min(9, len(central_addresses_state))):
        central_addresses_state[i]['bal'] = 2

    btg = BitcoinTransactionGraph()
    
    # 3. 初始化全局分组映射表
    comm_group_map = {} 




def generate_covert_transactions(message_size_B, start_height=934217-5, num_groups=9):
    """
    封装好的隐蔽交易生成方法。
    
    :param message_size_B: 隐蔽消息的大小 (单位: B)
    :param start_height: 模拟的起始区块高度
    :param num_groups: 中心地址的分组数量 (用于隔离拓扑结构)
    :return: (all_transactions, btg) 
             - all_transactions: 包含所有交易详情的列表
             - btg: 构建好的 BitcoinTransactionGraph 对象
    """
    
    # ==========================================
    # 1. 参数计算与初始化
    # ==========================================
    N_utxo = calculate_utxos(message_size_B) 
    
    # 初始化采样器
    txInOutSampler = init_TxInOutSampler()
    addrCentralDegreeSampler = init_AddressCentralDegreeSampler()
    
    # 计算期望与拓扑参数
    En, Em = calculate_txinout_expectations(txInOutSampler)
    N_1 = N_utxo / Em
    N_2 = N_utxo / En
    D_total = (N_1 * En + N_2 * Em) * 1.2
    
    Ed = calculate_addresscentrality_expectations(addrCentralDegreeSampler)
    
    N_comm = N_utxo
    N_center = (D_total + (2 - Ed) * N_comm) / Ed
    
    # 初始化中心地址状态 (带分组)
    central_addresses_state = init_center_addresses(
        N_center, 
        N_comm, 
        addrCentralDegreeSampler, 
        num_groups=num_groups
    )
    
    # 初始余额设置 (确保前 num_groups 个地址有钱，激活每个组)
    for i in range(min(num_groups, len(central_addresses_state))):
        central_addresses_state[i]['bal'] = 1

    # 初始化图对象和分组映射
    btg = BitcoinTransactionGraph()
    comm_group_map = {} 
    
    # 用于存储所有生成的交易列表
    all_transactions = []

    # ==========================================
    # 2. 模拟主循环
    # ==========================================
    round_idx = 0
    # 为了防止死循环，可以加一个最大轮次限制
    max_rounds = 100 
    
    print(f"开始生成隐蔽交易... 目标 UTXO: {N_utxo}, 消息大小: {message_size_B}B")

    while N_utxo > 0 and round_idx < max_rounds:
        print(f"\n--- Round {round_idx + 1} Start (剩余 UTXO: {N_utxo}) ---")
        
        # ---------------- Phase 1: 代币分配 ----------------
        round_comm_addresses = []
        phase1_tx_count = 0
        
        # 只要还有UTXO需要分配就尝试构造交易
        while N_utxo > 0:
            tx = construct_token_distribute_transaction(
                txInOutSampler,
                central_addresses_state,
                btg,
                start_height, # 传入隐蔽交易的其实区块高度
                comm_group_map    # 传入 Map
            )
            
            if tx == 'Done':
                # 中心地址UTXO分配完毕，无法再构造交易，跳出 Phase 1，进入 Phase 2 回笼资金
                break
            elif tx == 'Exceed':
                raise RuntimeError(f"Round {round_idx+1} Phase 1 直径超标，生成失败。")
            else:
                phase1_tx_count += 1
                round_comm_addresses.extend(tx['output_addrs'])
                N_utxo -= len(tx['output_addrs'])
                all_transactions.append(tx) # 记录交易

        print(f"  Phase 1 完成: 生成 {phase1_tx_count} 笔交易")
        
        # 如果 Phase 1 没生成任何交易 (通常是因为还没分完但中心地址没钱了)，
        # 且 Phase 2 也没法跑 (没有通信地址)，说明彻底卡死了
        if phase1_tx_count == 0 and len(round_comm_addresses) == 0:
            print("警告: 无法生成更多交易且无待回收资金，提前终止。")
            break

        # ---------------- Phase 2: 消息通信 ----------------
        phase2_tx_count = 0
        
        while len(round_comm_addresses) > 0:
            tx = construct_message_communication_transaction(
                txInOutSampler,
                round_comm_addresses,
                central_addresses_state,
                btg,
                start_height,
                comm_group_map # 传入 Map
            )
            
            if tx == 'Done':
                # 可能剩下几个零星地址因为分组匹配不上，跳过
                break 
            elif tx == 'Exceed':
                 raise RuntimeError(f"Round {round_idx+1} Phase 2 直径超标，生成失败。")
            else:
                phase2_tx_count += 1
                all_transactions.append(tx) # 记录交易

        print(f"  Phase 2 完成: 生成 {phase2_tx_count} 笔交易 (资金回笼)")
        round_idx += 1

    # ==========================================
    # 3. 结果返回
    # ==========================================
    print(f"\n生成结束。总轮次: {round_idx}, 总交易数: {len(all_transactions)}")
    print(f"最终图直径: {btg.calculate_diameter()}")
    
    return all_transactions, btg



if __name__ == "__main__":
    # try:
    #     # 假设我们要传输 1KB 的数据
    #     covert_tx_list, tx_graph = generate_covert_transactions(1024)
        
    #      # 2. 获取交易列表用于后续分析或写入文件
    #     print(f"一共生成了 {len(covert_tx_list)} 笔隐蔽交易")

    #     # 3. 获取交易 ID 列表，用于在可视化中高亮
    #     covert_tx_ids = [tx['hash'] for tx in covert_tx_list]

    #     # 4. 使用之前定义的 visualize 方法绘制
    #     # (假设你已经把 visualize 定义在 btg 对象里了)
    #     tx_graph.visualize(covert_tx_ids=covert_tx_ids)
    # except Exception as e:
    #     print(f"程序执行出错: {e}")
    # # init_parameter()
    
    # 获取隐蔽交易
    covert_tx, _ = generate_covert_transactions(
        message_size_B=4096, 
        num_groups=18 # 使用之前的分组策略
    )
     # Save file
    output_filename = "constructtx/GraphShadow_transactions.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(covert_tx, f, indent=4)
        
    print(f"\n[✓] Successfully saved {len(covert_tx)} transactions to {output_filename}")
    # round_idx = 0
    # while N_utxo > 0:
    #     print(f"\n{'=' * 20} 开始第 {round_idx + 1} 轮模拟 {'=' * 20}")
        
    #     # ---------------- Phase 1 ----------------
    #     print(f"\n[Phase 1] 代币分配...")
    #     round_comm_addresses = []
    #     dist_tx_count = 0
        
    #     while N_utxo > 0:
    #         tx = construct_token_distribute_transaction(
    #             txInOutSampler,
    #             central_addresses_state,
    #             btg,
    #             934217 - 5,
    #             comm_group_map # 传入 Map
    #         )
            
    #         if tx == 'Done':
    #             break
    #         elif tx == 'Exceed':
    #             print(f" 第 {round_idx + 1} 轮模拟 Phase 1 失败。")
    #             sys.exit()
    #         else:
    #             dist_tx_count += 1
    #             round_comm_addresses.extend(tx['output_addrs'])
    #             N_utxo -= len(tx['output_addrs'])

    #     print(f"Round {round_idx + 1} Phase 1 结束，产生了 {dist_tx_count} 笔交易。")
    #     if dist_tx_count == 0:
    #         break
    #     # btg.visualize() 
        
    #     # ---------------- Phase 2 ----------------
    #     print(f"\n[Phase 2] 消息通信...")
    #     comm_tx_count = 0
        
    #     while len(round_comm_addresses) > 0:
    #         tx = construct_message_communication_transaction(
    #             txInOutSampler,
    #             round_comm_addresses,
    #             central_addresses_state,
    #             btg,
    #             934217 - 5,
    #             comm_group_map # 传入 Map
    #         )
            
    #         if tx == 'Done':
    #             break # 可能有些地址因为分组问题没法匹配，跳过
    #         elif tx == 'Exceed':
    #             print(f" 第 {round_idx + 1} 轮模拟 Phase 2 失败。")
    #             sys.exit()
    #         else:
    #             comm_tx_count += 1

    #     print(f"Round {round_idx + 1} Phase 2 结束, 剩余{N_utxo}个消息片待传输。")
    #     round_idx += 1
        
    #     # 可视化 (需要确保类中有定义)
        
    #     btg.visualize()

    # print("\n模拟结束。")
    # print(f"最终直径: {btg.calculate_diameter()}")