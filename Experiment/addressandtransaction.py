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
import normalandcovertgraph
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
    G_base_nx = G_base.graph
    G_inject_nx = G_inject.graph
    G_mixed_nx = copy.deepcopy(G_base_nx)
    G_mixed_nx = nx.compose(G_mixed_nx, G_inject_nx)
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

def compare_address_degree(normal_wrapper, mixed_dict, max_x_limit=15, output_dir='experiment/'):
    """
    绘制多组混合图的 PMF 对比，计算量化指标 (TVD, JSD)，并保存为 PDF
    """
    # 1. 获取基准图 (根据 Main 逻辑，这里是 Wrapper)
    G_normal_nx = normal_wrapper.graph

    # 2. 计算基准分布数据
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
        label="Baseline: Normal Traffic",
        zorder=1,
    )

    # --- 样式定义 ---
    styles = [
        {"color": "#D62728", "marker": "^", "ls": "--"},  # 深红
        {"color": "#1F77B4", "marker": "s", "ls": "-."},  # 深蓝
        {"color": "#2CA02C", "marker": "D", "ls": ":"},   # 深绿
        {"color": "#9467BD", "marker": "v", "ls": "--"},  # 深紫
    ]

    num_scenarios = len(mixed_dict)

    for i, (label_name, G_mixed_nx) in enumerate(mixed_dict.items()):
        # 获取混合图 PMF
        x_mix, y_mix = get_pmf_data(G_mixed_nx)
        
        # === 计算量化指标 ===
        # 1. 数据对齐
        mix_prob_map = dict(zip(x_mix, y_mix))
        all_degrees = sorted(set(base_prob_map.keys()) | set(mix_prob_map.keys()))
        
        # 2. 构造向量
        P = np.array([base_prob_map.get(k, 0) for k in all_degrees])
        Q = np.array([mix_prob_map.get(k, 0) for k in all_degrees])
        
        # 3. 计算指标
        tvd = 0.5 * np.sum(np.abs(P - Q))
        jsd = jensenshannon(P, Q)
        
        # 4. 格式化图例
        legend_label = f"{label_name}\n(TVD={tvd:.4f}, JSD={jsd:.4f})"

        # 计算 Jitter
        if num_scenarios > 1:
            offset = (i - (num_scenarios - 1) / 2) * 0.15 
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

    plt.xlabel("Degree k (Linear Scale)", fontsize=14, fontweight="bold")
    plt.ylabel("Probability P(K=k)", fontsize=14, fontweight="bold")
    plt.title("Degree Distribution Comparison & Quantified Similarity", fontsize=16, pad=20)

    # 图例设置
    plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True, loc="upper right")
    plt.grid(True, which="major", linestyle="--", alpha=0.4, color="gray")
    plt.tight_layout()

    # === 保存逻辑 ===
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, "Degree_Distribution_Comparison.pdf")
    
    # bbox_inches='tight' 防止图例或标签被裁剪
    plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight')
    print(f"结果图已保存至: {save_path}")

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

# ==========================================
# 方法 1：修改版 - 独立绘图 (指标在图例)
# ==========================================
def compare_transaction_degree(normal_wrapper, mixed_dict, top_k=8, output_dir="."):
    G_normal = normal_wrapper.graph if hasattr(normal_wrapper, 'graph') else normal_wrapper
    
    # 调用公共计算逻辑
    labels, P, Q_dict, score_board = _compute_distributions(G_normal, mixed_dict, top_k)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = np.arange(len(labels))
    width = 0.35

    for item in score_board:
        label_name = item['Scheme']
        Q = Q_dict[label_name]
        tvd = item['TVD']
        jsd = item['JSD']
        
        plt.figure(figsize=(10, 6))

        # 绘制基准
        plt.bar(x - width/2, P, width, label="Baseline: Normal", 
                color="#1F77B4", alpha=0.85, edgecolor="black", zorder=2)

        # 绘制混合 (图例中包含指标)
        legend_label = f"Mixed: {label_name}\n(TVD={tvd:.4f}, JSD={jsd:.4f})"
        plt.bar(x + width/2, Q, width, label=legend_label,
                color="#D62728", alpha=0.85, edgecolor="black", hatch="//", zorder=2)

        plt.ylabel("Probability", fontsize=12, fontweight="bold")
        plt.title(f"Transaction Structure Distribution ({label_name})", fontsize=14, pad=15)
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

# ==========================================
# 方法 2：新增 - 统一绘图 (Subplots)
# ==========================================
def compare_transaction_degree_union(normal_wrapper, mixed_dict, top_k=8, output_dir="."):
    """
    将所有方案的对比绘制在同一张大图中 (子图网格形式)
    """
    G_normal = normal_wrapper.graph if hasattr(normal_wrapper, 'graph') else normal_wrapper
    
    # 调用公共计算逻辑
    labels, P, Q_dict, score_board = _compute_distributions(G_normal, mixed_dict, top_k)
    
    num_schemes = len(mixed_dict)
    
    # 动态计算网格布局 (例如 4个方案 -> 2x2, 3个方案 -> 1x3 或 2x2)
    cols = 2 if num_schemes > 1 else 1
    rows = math.ceil(num_schemes / cols)
    
    # 创建大图
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), sharey=True)
    axes = np.array(axes).reshape(-1) # 展平方便遍历
    
    x = np.arange(len(labels))
    width = 0.35
    
    # 遍历每个方案画子图
    for i, item in enumerate(score_board):
        ax = axes[i]
        label_name = item['Scheme']
        Q = Q_dict[label_name]
        tvd = item['TVD']
        jsd = item['JSD']
        
        # 绘制
        ax.bar(x - width/2, P, width, label="Baseline", 
               color="#1F77B4", alpha=0.85, edgecolor="black", zorder=2)
        
        legend_label = f"{label_name}\n(TVD={tvd:.4f}, JSD={jsd:.4f})"
        ax.bar(x + width/2, Q, width, label=legend_label,
               color="#D62728", alpha=0.85, edgecolor="black", hatch="//", zorder=2)
        
        # 设置子图属性
        ax.set_title(f"vs. {label_name}", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
        ax.legend(fontsize=9, loc="upper right")
        
        if i % cols == 0:
            ax.set_ylabel("Probability", fontweight='bold')

    # 隐藏多余的空子图 (如果有)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle(f"Transaction Structure Comparison (Top-{top_k})", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # 保存
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "Structure_Comparison_All_In_One.pdf")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Success] 统一对比图已保存至: {save_path}")
    

if __name__ == "__main__":
    # 统计10个区块的正常交易
    normal_tx = []
    # 这里加载数据用于演示
    for i in range(923800, 923801):
        try:
            filename = f"dataset/transactions_block_{i}.json"
            file_transactions = load_transactions_from_file(filename)
            # normal_tx.extend(file_transactions)
            normal_tx.extend(random.sample(file_transactions, 600))
        except Exception as e:
            print(f"Skip {filename}: {e}")

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
    
    normal_graph = constuct_graph(normal_tx)

    scenarios = {
        "GraphShadow": merge_graphs(normal_graph, constuct_graph(GraphShadow_tx)),
        "DDSAC": merge_graphs(normal_graph, constuct_graph(DDSAC_tx)),
        "GBCTD": merge_graphs(normal_graph, constuct_graph(GBCTD_tx)),
        "BlockWhisper": merge_graphs(normal_graph, constuct_graph(BlockWhisper_tx)),
    }
    
    # === 新增：打印节点数量统计 ===
    print("\n" + "="*40)
    print("   图规模统计 (Node Counts)")
    print("="*40)
    
    # normal_graph 是 Wrapper 类，需要访问 .graph 属性
    print(f"{'Normal (Baseline)':<20} : {normal_graph.graph.number_of_nodes()} nodes")
    
    # scenarios 中的值是 merge_graphs 返回的 NetworkX 对象，直接调用 number_of_nodes()
    for name, G_mixed in scenarios.items():
        print(f"{name:<20} : {G_mixed.number_of_nodes()} nodes")
    print("="*40 + "\n")

    # print(">>> 开始绘制地址中心度分布对比图...")
    # compare_address_degree(normal_graph, scenarios, 15, 'experiment/')

    print(">>> 开始绘制交易结构对比图...")
    compare_transaction_degree_union(normal_graph, scenarios, 8, 'experiment/')
