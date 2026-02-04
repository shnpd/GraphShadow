import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
import os
import random

# 引入项目模块 (确保这些路径在你的项目中存在)
from txgraph.main import BitcoinTransactionGraph
from graphanalysis.sample_transaction import load_transactions_from_file

# ==========================================
# 1. 基础构建函数
# ==========================================
def constuct_graph(tx_list):
    """根据交易列表构建图"""
    btg = BitcoinTransactionGraph()
    for tx in tx_list:
        btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
    return btg

def merge_graphs(G_base_nx, G_inject_nx):
    """合并正常图与隐蔽图"""
    G_mixed_nx = copy.deepcopy(G_base_nx)
    G_mixed_nx = nx.compose(G_mixed_nx, G_inject_nx)
    return G_mixed_nx

# ==========================================
# 2. 核心算法：近似有向直径计算
# ==========================================
def calculate_approx_directed_diameter(G, sample_size=100):
    """
    计算有向图的近似直径（忽略不可达节点）
    
    原理：
    1. 随机选取 sample_size 个起始节点（探测器）。
    2. 对每个起点，计算它能到达的所有节点的最短路径。
    3. 找到所有可达路径中的最大值。
    
    :param G: NetworkX DiGraph
    :param sample_size: 采样次数，建议 50-100，越大越准但越慢
    :return: 估计的最大跳数 (Integer)
    """
    nodes = list(G.nodes())
    total_nodes = len(nodes)
    
    # 如果图很小，全量计算；否则采样
    # if total_nodes <= sample_size:
        # target_sources = nodes
    # else:
        # target_sources = random.sample(nodes, sample_size)
    target_sources = nodes
    max_dist = 0
    
    # 开始探测
    for start_node in target_sources:
        try:
            # nx.single_source_shortest_path_length 会自动忽略不可达的节点
            # 它返回一个字典: {target_node: distance}
            # 这里的 distance 是跳数
            lengths = nx.single_source_shortest_path_length(G, start_node)
            
            # 如果这是一个孤立点（只能到自己，距离为0），跳过
            if not lengths:
                continue
                
            # 获取从当前 start_node 出发能走到的最远距离
            current_max = max(lengths.values())
            
            # 更新全局最大值
            if current_max > max_dist:
                max_dist = current_max
                
        except Exception:
            continue
            
    return max_dist

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_directed_diameter_comparison(normal_wrapper, covert_dict, output_dir='experiment_results'):
    """
    绘制有向直径对比柱状图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(">>> 正在计算 [基准: Normal] 的有向直径...")
    G_normal_nx = normal_wrapper.graph
    # 正常图采样 100 次足以收敛
    d_norm = calculate_approx_directed_diameter(G_normal_nx, sample_size=100)
    print(f"    Normal Diameter: {d_norm}")
    
    # 准备绘图数据
    labels = ['Normal\n(Baseline)']
    diameters = [d_norm]
    # 为不同方案分配颜色
    colors = ['#1F77B4'] # 基准蓝
    scheme_colors = ['#D62728', '#2CA02C', '#FF7F0E', '#9467BD', '#8C564B'] # 红绿橙紫棕
    
    # 循环处理每个隐蔽方案
    for i, (label_name, covert_wrapper) in enumerate(covert_dict.items()):
        print(f">>> 正在计算 [混合: {label_name}] 的有向直径...")
        
        G_covert_nx = covert_wrapper.graph
        
        # 计算混合图直径
        d_mix = calculate_approx_directed_diameter(G_covert_nx, sample_size=100)
        print(f"    {label_name} Diameter: {d_mix}")
        
        labels.append(label_name)
        diameters.append(d_mix)
        colors.append(scheme_colors[i % len(scheme_colors)])
        
    # === 开始绘图 ===
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(labels))
    width = 0.5
    
    # 绘制柱子
    bars = plt.bar(x, diameters, width, color=colors, alpha=0.85, edgecolor='black', zorder=3)
    
    # 设置 Y 轴范围 (稍微留高一点放数字)
    plt.ylim(0, max(diameters) * 1.25)
    
    # 标签与标题
    plt.ylabel('Max Reachable Distance (Hops)', fontsize=14, fontweight='bold')
    plt.title('Directed Graph Diameter Comparison\n(Max Path Length Ignoring Infinity)', fontsize=16, pad=20)
    plt.xticks(x, labels, fontsize=11, fontweight='bold')
    
    # 网格线
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    # 在柱子上标注数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', 
                 fontsize=14, fontweight='bold')
        
    # 如果直径变化很大，可以加一个红色虚线表示基准线
    plt.axhline(y=d_norm, color='#1F77B4', linestyle='--', linewidth=1.5, alpha=0.6, zorder=4)

    plt.tight_layout()
    
    # 保存结果
    save_path = os.path.join(output_dir, "Directed_Diameter_Comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[完成] 结果图已保存至: {save_path}")
    plt.show()

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    print(">>> 正在加载正常交易数据...")
    normal_tx = []
    # 加载10个区块的数据
    for i in range(923800, 923810):
        filename = f"dataset/transactions_block_{i}.json"
        try:
            file_transactions = load_transactions_from_file(filename)
            # 随机采样以控制图规模，确保实验速度
            if len(file_transactions) > 25:
                normal_tx.extend(random.sample(file_transactions, 25))
            else:
                normal_tx.extend(file_transactions)
        except FileNotFoundError:
            print(f"Warning: 文件 {filename} 未找到，跳过。")

    # 构建正常图
    G_normal_wrapper = constuct_graph(normal_tx)

    print(">>> 正在加载对比方案数据...")
    try:
        # 加载各方案数据 (请确保路径正确)
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

        # 运行对比
        plot_directed_diameter_comparison(G_normal_wrapper, scenarios, output_dir='experiment_results')
        
    except FileNotFoundError as e:
        print(f"Error: 找不到数据文件 - {e}")
        print("请检查 constructtx/ 或 CompareMethod/ 下的文件路径。")