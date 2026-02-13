import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import collections
import math
import pandas as pd
from graphanalysis.sample_transaction import (
    load_transactions_from_file,
    load_graph_cache,
)
from constructtx.constructTxSplitG_v3 import generate_covert_transactions
from txgraph.main import BitcoinTransactionGraph
import constructtx.utils as utils
import random
import copy
from scipy.spatial.distance import jensenshannon


def constuct_graph(tx_list):
    btg = BitcoinTransactionGraph()
    for tx in tx_list:
        btg.add_transaction(tx["hash"], tx["input_addrs"], tx["output_addrs"])
    return btg


def merge_graphs(G_base, G_inject):
    """
    合并基准图和注入图。
    注意：此函数现在主要在 main 中调用，用于预处理数据。
    """
    G_mixed_nx = copy.deepcopy(G_base)
    G_mixed_nx = nx.compose(G_mixed_nx, G_inject)
    # 返回的是 NetworkX 对象，如果后续需要 Wrapper，需注意封装，
    # 但根据你的代码逻辑，Scenarios 中存 NetworkX 对象即可，或者存 Wrapper 但 Wrapper.graph 是混合后的
    # 根据 main 中的逻辑：scenarios 存的是 NetworkX 对象
    return G_mixed_nx


# === 2. 核心：计算概率分布 (PMF) ===
def get_pmf_data(G):
    """
    计算度数的概率分布 P(K=k)
    """
    # 提取地址节点度数
    degrees = [d for n, d in G.degree() if G.nodes[n].get("node_type") == "address"]

    if not degrees:
        return [], []

    total_nodes = len(degrees)
    degree_counts = collections.Counter(degrees)

    # 获取所有出现的度数，并排序
    x = sorted(degree_counts.keys())

    # 计算每个度数的出现概率 (Count / Total)
    y = [degree_counts[k] / total_nodes for k in x]

    return x, y

def compare_address_degree(G_normal_nx, mixed_dict, max_x_limit=15, output_dir='experiment/'):
    """
    绘制多组混合图的 PMF 对比，计算量化指标 (TVD, JSD)，并保存为 PDF，
    最后在控制台格式化输出评估排行榜。
    """
    # 1. 设置字体家族为 macOS 内置的通用支持中文的字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 

    # 2. 解决负号 '-' 显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 3. 计算基准分布数据
    x_base, y_base = get_pmf_data(G_normal_nx)
    
    # 构造基准概率字典 {degree: prob}
    base_prob_map = dict(zip(x_base, y_base))

    # 设置画布
    plt.figure(figsize=(12, 7))

    # --- 绘制基准线 (Baseline) ---
    plt.plot(
        x_base,
        y_base,
        color="#333333",
        linestyle="-",
        linewidth=4,
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgewidth=2,
        alpha=0.4,
        label="Baseline",
        zorder=1,
    )

    # --- 样式定义 ---
    styles = [
        {"color": "#FF7F0E", "marker": "X", "ls": "-"},   # 橙色 (Normal Injection)
        {"color": "#D62728", "marker": "^", "ls": "--"},  # 深红
        {"color": "#1F77B4", "marker": "s", "ls": "-."},  # 深蓝
        {"color": "#2CA02C", "marker": "D", "ls": ":"},   # 深绿
        {"color": "#9467BD", "marker": "v", "ls": "--"},  # 深紫
    ]

    num_scenarios = len(mixed_dict)
    
    # === 新增：用于收集量化指标的列表 ===
    metrics_board = []

    for i, (label_name, G_mixed_nx) in enumerate(mixed_dict.items()):
        # 获取混合图 PMF
        x_mix, y_mix = get_pmf_data(G_mixed_nx)
        
        # === 计算量化指标 ===
        mix_prob_map = dict(zip(x_mix, y_mix))
        all_degrees = sorted(set(base_prob_map.keys()) | set(mix_prob_map.keys()))
        
        P = np.array([base_prob_map.get(k, 0) for k in all_degrees])
        Q = np.array([mix_prob_map.get(k, 0) for k in all_degrees])
        
        tvd = 0.5 * np.sum(np.abs(P - Q))
        jsd = jensenshannon(P, Q)
        
        # 将计算结果存入列表
        metrics_board.append({
            "方案 (Scheme)": label_name,
            "TVD": tvd,
            "JSD": jsd
        })
        
        # 图例保持清爽
        legend_label = f"{label_name}"

        # 计算 Jitter
        if num_scenarios > 1:
            offset = (i - (num_scenarios - 1) / 2) * 0.1
        else:
            offset = 0

        x_jittered = [x + offset for x in x_mix]

        # 获取样式
        style = styles[i % len(styles)]

        # --- 绘制实验线 ---
        plt.plot(
            x_jittered,
            y_mix,
            color=style["color"],
            linestyle=style["ls"],
            marker=style["marker"],
            markersize=7,
            linewidth=2,
            alpha=0.9,
            label=legend_label, 
            zorder=2 + i,
        )

    # --- 坐标轴与美化 ---
    plt.xlim(0.5, max_x_limit + 0.5)
    plt.ylim(-0.02, 1.05)
    plt.xticks(np.arange(1, max_x_limit + 1, 1))

    plt.xlabel("地址中心度", fontsize=14, fontweight="bold")
    plt.ylabel("分布概率", fontsize=14, fontweight="bold")
    plt.title("地址中心度分布概率对比", fontsize=16, pad=20)

    # 图例设置
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, loc="upper right")
    plt.grid(True, which="major", linestyle="--", alpha=0.4, color="gray")
    plt.tight_layout()

    # === 保存逻辑 ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "Degree_Distribution_Comparison.pdf")
    plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f"\n[完成] 分布图已保存至: {save_path}")

    # ==========================================
    # 新增：控制台格式化输出量化指标排行榜
    # ==========================================
    print("\n" + "=" * 55)
    print("   地址中心度分布差异评估 (量化指标排行榜)")
    print("=" * 55)
    
    # 转换为 DataFrame 进行美化和排序
    df_metrics = pd.DataFrame(metrics_board)
    
    # 根据 TVD 从小到大排序 (值越小，与基线越相似)
    df_metrics = df_metrics.sort_values(by="TVD", ascending=True).reset_index(drop=True)
    
    # 格式化浮点数，保留4位小数
    df_metrics['TVD'] = df_metrics['TVD'].map('{:.4f}'.format)
    df_metrics['JSD'] = df_metrics['JSD'].map('{:.4f}'.format)
    
    # 打印表格
    print(df_metrics.to_string(index=True))
    print("=" * 55 + "\n")

    plt.show()

# === 辅助函数：获取结构计数 ===
def get_structure_counts(G):
    tx_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "transaction"]
    if not tx_nodes:
        return {}
    structures = []
    for tx in tx_nodes:
        n_in = G.in_degree(tx)
        n_out = G.out_degree(tx)
        structures.append((n_in, n_out))
    return collections.Counter(structures)

# === 辅助函数：预处理概率分布 ===
def _compute_distributions(G_normal, mixed_dict, top_k=8):
    """
    内部逻辑提取：计算基准分布 P 和各个方案的分布 Q
    返回: labels, P, Q_dict, score_board
    """
    # 1. 计算基准 P
    normal_counts = get_structure_counts(G_normal)
    total_normal = sum(normal_counts.values())
    most_common = normal_counts.most_common(top_k)

    labels = []
    target_patterns = []
    P = []
    
    for struct, count in most_common:
        n_in, n_out = struct
        labels.append(f"{n_in}-in\n{n_out}-out")
        P.append(count / total_normal)
        target_patterns.append(struct)

    # 处理 Others
    others_count = total_normal - sum([c for _, c in most_common])
    labels.append("Others")
    target_patterns.append("Others")
    P.append(others_count / total_normal)
    P = np.array(P)

    # 2. 计算各个混合方案 Q
    Q_dict = {}
    score_board = []

    for label_name, mixed_graph_obj in mixed_dict.items():
        G_mixed = mixed_graph_obj

        mixed_counts = get_structure_counts(G_mixed)
        total_mixed = sum(mixed_counts.values())

        Q = []
        for pat in target_patterns:
            if pat == "Others":
                current_top_k = sum([mixed_counts[p] for p in target_patterns if p != "Others"])
                val = total_mixed - current_top_k
                Q.append(val / total_mixed)
            else:
                Q.append(mixed_counts.get(pat, 0) / total_mixed)
        Q = np.array(Q)
        
        # 计算指标
        tvd = 0.5 * np.sum(np.abs(P - Q))
        jsd = jensenshannon(P, Q)
        
        Q_dict[label_name] = Q
        score_board.append({"Scheme": label_name, "TVD": tvd, "JSD": jsd})

    return labels, P, Q_dict, score_board


def compare_transaction_degree(G_normal, mixed_dict, top_k=8, output_dir="."):
    """
    绘制交易结构分布对比图
    修改点：
    1. 统一纵坐标范围 (Y-Axis Limit)
    2. 图例汉化 (Chinese Legends)
    """
    # 1. 设置字体 (确保中文正常显示)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 调用公共计算逻辑获取数据
    labels, P, Q_dict, score_board = _compute_distributions(G_normal, mixed_dict, top_k)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = np.arange(len(labels))
    width = 0.35

    # === 关键修改 1：计算统一的纵坐标上限 ===
    # 找出基准 P 和所有混合方案 Q 中的最大概率值
    max_p = np.max(P)
    max_q = 0
    for q_arr in Q_dict.values():
        current_max = np.max(q_arr)
        if current_max > max_q:
            max_q = current_max
    
    # 设置上限为最大值的 1.15 倍，留出空间给图例
    y_limit = max(max_p, max_q) * 1.15

    for item in score_board:
        label_name = item['Scheme']
        Q = Q_dict[label_name]
        tvd = item['TVD']
        jsd = item['JSD']
        
        plt.figure(figsize=(10, 6))

        # 绘制基准 (中文图例)
        plt.bar(x - width/2, P, width, label="baseline", 
                color="#1F77B4", alpha=0.85, edgecolor="black", zorder=2)

        # 绘制混合 (中文图例 + 指标)
        # 格式化字符串：将方案名和指标分行显示
        legend_label = f"{label_name}"
        
        plt.bar(x + width/2, Q, width, label=legend_label,
                color="#D62728", alpha=0.85, edgecolor="black", hatch="//", zorder=2)

        # === 关键修改 2：应用统一纵坐标 ===
        plt.ylim(0, y_limit)

        plt.ylabel("概率", fontsize=12, fontweight="bold")
        plt.title(f"交易结构分布对比 ({label_name})", fontsize=14, pad=15)
        plt.xticks(x, labels, fontsize=11)
        
        # 图例位置优化
        plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc="upper right")
        plt.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
        plt.tight_layout()

        safe_name = label_name.replace(" ", "_").replace(":", "")
        plt.savefig(os.path.join(output_dir, f"Struct_Diff_{safe_name}.pdf"), dpi=300)
        plt.close()

    # 打印排行榜
    df_scores = pd.DataFrame(score_board).sort_values(by="TVD").reset_index(drop=True)
    print("\n=== 隐蔽性评估排行榜 (独立文件已保存) ===")
    print(df_scores.to_string())
    return df_scores

    

if __name__ == "__main__":
    base_tx = []
    added_normal_tx = [] # 新增：用于混合的正常交易组
    random.seed(42)    
    for i in range(923800, 923810):
        filename = f"dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        # base_tx.extend( file_transactions)
        base_tx.extend( random.sample(file_transactions, min(250, len(file_transactions))))

    for i in range(923840, 923850):
        filename = f"dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        added_normal_tx.extend(random.sample(file_transactions, min(80, len(file_transactions))))


    # 加载各种隐蔽交易数据
    GraphShadow_tx = load_transactions_from_file(
        "constructtx/GraphShadow_transactions.json"
    )
    DDSAC_tx = load_transactions_from_file(
        "CompareMethod/DDSAC/DDSAC_transactions.json"
    )
    GBCTD_tx = load_transactions_from_file(
        "CompareMethod/GBCTD/GBCTD_transactions.json"
    )
    BlockWhisper_tx = load_transactions_from_file(
        "CompareMethod/BlockWhisper/BlockWhisper_transactions.json"
    )
    
    base_graph = constuct_graph(base_tx).graph
    added_normal_graph = constuct_graph(added_normal_tx).graph
    graphshadow_graph = constuct_graph(GraphShadow_tx).graph
    ddsac_graph = constuct_graph(DDSAC_tx).graph
    gbctd_graph = constuct_graph(GBCTD_tx).graph
    blockwhisper_graph = constuct_graph(BlockWhisper_tx).graph
    
    merge_normal_graph = merge_graphs(base_graph, added_normal_graph)
    merge_GraphShadow_graph = merge_graphs(base_graph, graphshadow_graph)
    merge_DDSAC_graph = merge_graphs(base_graph, ddsac_graph)
    merge_GBCTD_graph = merge_graphs(base_graph, gbctd_graph)
    merge_BlockWhisper_graph = merge_graphs(base_graph, blockwhisper_graph)
    
    scenarios = {
        # === 新增对照组 ===
        "Normal": merge_normal_graph,
        "GraphShadow": merge_GraphShadow_graph,
        "DDSAC": merge_DDSAC_graph,
        "GBCTD": merge_GBCTD_graph,
        "BlockWhisper": merge_BlockWhisper_graph,
    }
    
    # === 新增：打印节点数量与交易数量统计 ===
    print("\n" + "="*55)
    print("   图规模统计 (Nodes & Transactions)")
    print("="*55)
    
    # 定义一个临时函数用于计算交易节点数
    def get_tx_count(G):
        return sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "transaction")
    
    # 1. 打印 Baseline (注意：base_graph 是 Wrapper，需取 .graph)
    base_node_count = base_graph.number_of_nodes()
    base_tx_count = get_tx_count(base_graph)
    
    print(f"{'Normal (Baseline)':<20} : {base_node_count:<5} nodes | {base_tx_count:<5} txs")
    
    # 2. 打印 Scenarios (注意：scenarios 里的值已经是 NetworkX 对象)
    for name, G_mixed in scenarios.items():
        node_count = G_mixed.number_of_nodes()
        tx_count = get_tx_count(G_mixed)
        
        print(f"{name:<20} : {node_count:<5} nodes | {tx_count:<5} txs")
        
    print("="*55 + "\n")


    # print(">>> 开始绘制地址中心度分布对比图...")
    # compare_address_degree(base_graph, scenarios, 10, 'experiment/')

    print(">>> 开始绘制交易结构对比图...")
    compare_transaction_degree(base_graph, scenarios, 8, 'experiment/')
