import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections

import pandas as pd
from graphanalysis.sample_transaction import load_transactions_from_file, load_graph_cache
from constructtx.constructTxSplitG_v3 import generate_covert_transactions
from txgraph.main import BitcoinTransactionGraph
import constructtx.utils as utils
import random
import normalandcovertgraph
import copy
from scipy.spatial.distance import jensenshannon

def constuct_graph(tx_list):
    btg = BitcoinTransactionGraph()
    for tx in tx_list:
        btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
    return btg

def merge_graphs(G_base_nx, G_inject_nx):
    G_mixed_nx = copy.deepcopy(G_base_nx)
    G_mixed_nx = nx.compose(G_mixed_nx, G_inject_nx)
    return G_mixed_nx

# === 2. 核心：计算概率分布 (PMF) ===
def get_pmf_data(G):
    """
    计算度数的概率分布 P(K=k)
    """
    # 提取地址节点度数
    degrees = [d for n, d in G.degree() if G.nodes[n].get('node_type') == 'address']
    
    if not degrees:
        return [], []
        
    total_nodes = len(degrees)
    degree_counts = collections.Counter(degrees)
    
    # 获取所有出现的度数，并排序
    x = sorted(degree_counts.keys())
    
    # 计算每个度数的出现概率 (Count / Total)
    y = [degree_counts[k] / total_nodes for k in x]
        
    return x, y

# === 3. 绘图函数 (优化版) ===
def plot_multi_scenario_pmf_linear(normal_wrapper, covert_dict):
    """
    绘制多组混合图的 PMF 对比 (线性坐标 - 美化版)
    包含：Jitter抖动、高对比度颜色、不同Marker
    """
    G_normal_nx = normal_wrapper.graph
    
    # 1. 计算基准线 (Baseline)
    x_base, y_base = get_pmf_data(G_normal_nx)
    
    # 设置图布风格，使用白色背景网格
    plt.figure(figsize=(12, 7)) #稍微加宽一点
    
    # --- 改进点1：基准线设为“背景真值” ---
    # 使用深灰色粗线，作为背景参考，zorder设为1保证它在最底层
    plt.plot(x_base, y_base, color='#333333', linestyle='-', linewidth=4, 
             marker='o', markersize=8, markerfacecolor='none', markeredgewidth=2,
             alpha=0.4, label='Baseline: Normal Traffic', zorder=1)
    
    # --- 改进点2：定义高对比度样式组 ---
    # 颜色选用：深红、深蓝、深绿、深紫 (避免荧光色)
    styles = [
        {'color': '#D62728', 'marker': '^', 'ls': '--'},  # 深红 + 虚线 + 三角
        {'color': '#1F77B4', 'marker': 's', 'ls': '-.'},  # 深蓝 + 点划线 + 方块
        {'color': '#2CA02C', 'marker': 'D', 'ls': ':'},   # 深绿 + 点线 + 菱形
        {'color': '#9467BD', 'marker': 'v', 'ls': '--'}   # 深紫 + 虚线 + 倒三角
    ]
    
    max_display_degree = 0
    num_scenarios = len(covert_dict)
    
    for i, (label_name, covert_wrapper) in enumerate(covert_dict.items()):
        G_covert_nx = covert_wrapper.graph
        G_mixed = merge_graphs(G_normal_nx, G_covert_nx)
        x_mix, y_mix = get_pmf_data(G_mixed)
        
        current_max = max(x_mix) if x_mix else 0
        max_display_degree = max(max_display_degree, current_max)
        
        # --- 改进点3：X轴微量抖动 (Jitter) ---
        # 如果不抖动，所有点都会叠在一起看不清
        # 偏移量计算：让点围绕整数刻度左右散开
        # 例如3个组：偏移 -0.15, 0, +0.15
        if num_scenarios > 1:
            offset = (i - (num_scenarios - 1) / 2) * 0.15 
        else:
            offset = 0
            
        x_jittered = [x + offset for x in x_mix]
        
        # 获取样式
        style = styles[i % len(styles)]
        
        # 绘制实验线 (zorder=2 保证画在基准线上面)
        plt.plot(x_jittered, y_mix, 
                 color=style['color'], 
                 linestyle=style['ls'], 
                 marker=style['marker'],
                 markersize=7, 
                 linewidth=2, 
                 alpha=0.9, # 不透明度高一点
                 label=f'Mixed: {label_name}',
                 zorder=2 + i) # 每一层都稍微高一点

    # 4. 坐标轴优化
    limit_x = min(max_display_degree * 1.5, 25) 
    plt.xlim(0.5, limit_x) 
    plt.ylim(-0.02, 1.05) # 稍微留点负空间给Marker展示
    
    # 设置刻度只显示整数 (因为度数是整数)
    plt.xticks(np.arange(1, int(limit_x) + 1, 1))
    
    plt.xlabel('Degree k (Linear Scale)', fontsize=14, fontweight='bold')
    plt.ylabel('Probability P(K=k)', fontsize=14, fontweight='bold')
    plt.title('Degree Distribution Comparison (PMF)', fontsize=16, pad=20)
    
    # 图例优化
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc='upper right')
    
    # 网格线优化
    plt.grid(True, which='major', linestyle='--', alpha=0.4, color='gray')
    
    plt.tight_layout()
    plt.show()
def get_structure_counts(G):
    """
    获取交易结构 (in, out) 的计数
    """
    tx_nodes = [n for n in G.nodes() if G.nodes[n].get('node_type') == 'transaction']
    if not tx_nodes:
        return {}
    
    structures = []
    for tx in tx_nodes:
        n_in = G.in_degree(tx)
        n_out = G.out_degree(tx)
        structures.append((n_in, n_out))
        
    return collections.Counter(structures)

def evaluate_and_plot_structures(normal_wrapper, covert_dict, top_k=8, output_dir='.'):
    """
    绘制 Top-K 结构对比图，并计算量化差异指标 (TVD, JSD)
    最后输出隐蔽性排行榜
    """
    G_normal_nx = normal_wrapper.graph
    
    # --- 1. 预处理基准分布 ---
    print("正在计算基准分布...")
    normal_counts = get_structure_counts(G_normal_nx)
    total_normal = sum(normal_counts.values())
    
    # 提取 Top-K 模式
    most_common = normal_counts.most_common(top_k)
    
    labels = []
    target_patterns = []
    
    # 构造基准概率向量 P
    P = []
    for (struct, count) in most_common:
        n_in, n_out = struct
        labels.append(f"{n_in}-in\n{n_out}-out")
        P.append(count / total_normal)
        target_patterns.append(struct)
        
    # 处理 Others
    others_count = total_normal - sum([c for _, c in most_common])
    labels.append("Others")
    target_patterns.append("Others")
    P.append(others_count / total_normal)
    
    # 转为 numpy array 方便计算
    P = np.array(P)
    
    # --- 2. 循环评估每个方案 ---
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    x = np.arange(len(labels))
    width = 0.35
    
    # 用于存储评分结果
    score_board = []

    for label_name, covert_wrapper in covert_dict.items():
        print(f"正在评估: {label_name} ...")
        
        G_covert_nx = covert_wrapper.graph
        G_mixed = merge_graphs(G_normal_nx, G_covert_nx)
        
        mixed_counts = get_structure_counts(G_mixed)
        total_mixed = sum(mixed_counts.values())
        
        # 构造混合概率向量 Q
        Q = []
        for pat in target_patterns:
            if pat == "Others":
                current_top_k = sum([mixed_counts[p] for p in target_patterns if p != "Others"])
                val = total_mixed - current_top_k
                Q.append(val / total_mixed)
            else:
                Q.append(mixed_counts.get(pat, 0) / total_mixed)
        
        Q = np.array(Q)
        
        # === 核心：计算量化指标 ===
        # 1. TVD (Total Variation Distance)
        tvd_score = 0.5 * np.sum(np.abs(P - Q))
        
        # 2. JSD (Jensen-Shannon Divergence)
        # scipy 的 jensenshannon 返回的是距离 (平方根后的)，范围 [0, 1]
        jsd_score = jensenshannon(P, Q)
        
        # 存入排行榜
        score_board.append({
            "Scheme": label_name,
            "TVD": tvd_score,
            "JSD": jsd_score
        })
        
        # === 绘图 ===
        plt.figure(figsize=(10, 6))
        
        plt.bar(x - width/2, P, width, label='Baseline: Normal', 
                color='#1F77B4', alpha=0.85, edgecolor='black', zorder=2)
        
        plt.bar(x + width/2, Q, width, label=f'Mixed: {label_name}', 
                color='#D62728', alpha=0.85, edgecolor='black', hatch='//', zorder=2)
        
        plt.ylabel('Probability', fontsize=12, fontweight='bold')
        
        # === 将分数写在标题里 ===
        title_str = (f'Transaction Structure Distribution\n'
                     f'Scheme: {label_name} | TVD: {tvd_score:.4f} (Lower is Better)')
        plt.title(title_str, fontsize=14, pad=15)
        
        plt.xticks(x, labels, fontsize=11)
        plt.legend(fontsize=11)
        plt.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)
        plt.tight_layout()
        
        safe_name = label_name.replace(" ", "_").replace(":", "")
        plt.savefig(os.path.join(output_dir, f"Struct_Diff_{safe_name}.pdf"), dpi=300)
        plt.close()

    # --- 3. 输出最终排行榜 ---
    print("\n" + "="*50)
    print("   隐蔽性评估排行榜 (值越小越好)")
    print("="*50)
    
    # 使用 Pandas 美化输出 (如果没有 pandas 可以直接 print 列表)
    df_scores = pd.DataFrame(score_board)
    # 按 TVD 从小到大排序 (越小越好)
    df_scores = df_scores.sort_values(by="TVD", ascending=True).reset_index(drop=True)
    
    print(df_scores.to_string(index=True))
    print("="*50)
    print(f"最佳方案是: {df_scores.iloc[0]['Scheme']}")

    return df_scores

if __name__ == "__main__":
    # 统计10个区块的正常交易
    normal_tx = []
    for i in range(923800, 923810):
        filename = f"dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        normal_tx.extend(random.sample(file_transactions, 25))


    GraphShadow_tx = load_transactions_from_file("constructtx/GraphShadow_transactions.json")
    DDSAC_tx = load_transactions_from_file("CompareMethod/DDSAC/DDSAC_transactions.json")
    GBCTD_tx = load_transactions_from_file("CompareMethod/GBCTD/GBCTD_transactions.json")
    BlockWhisper_tx = load_transactions_from_file("CompareMethod/BlockWhisper/BlockWhisper_transactions.json")

    scenarios = {
        "GraphShadow": constuct_graph(GraphShadow_tx),
        "DDSAC": constuct_graph(DDSAC_tx),
        "GBCTD": constuct_graph(GBCTD_tx),
        "BlockWhisper": constuct_graph(BlockWhisper_tx)
    }

    plot_multi_scenario_pmf_linear(constuct_graph(normal_tx), scenarios)
    # evaluate_and_plot_structures(constuct_graph(normal_tx), scenarios, 8, "Experiment/transactionnode")