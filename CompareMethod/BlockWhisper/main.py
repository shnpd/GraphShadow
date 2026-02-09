import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from collections import defaultdict
import hashlib
import time
import math
# ==========================================
# 模块 A: 完善后的交易生成器
# ==========================================
class BlockWhisperGraph:
    def __init__(self, address_pool, alpha_out=1.0, alpha_in=1.0):
        self.address_pool = address_pool
        self.n = len(address_pool)
        
        # 构建头部地址列表 (Top Address Lists)
        self.source_top_list = address_pool.copy()
        random.shuffle(self.source_top_list)
        self.dest_top_list = address_pool.copy()
        random.shuffle(self.dest_top_list)
        
        # 预计算 Zipf 概率
        self.prob_out = self._calculate_zipf_probabilities(self.n, alpha_out)
        self.prob_in = self._calculate_zipf_probabilities(self.n, alpha_in)

    def _calculate_zipf_probabilities(self, n, alpha):
        ranks = np.arange(1, n + 1)
        denominator = np.sum(1 / (ranks ** alpha))
        return (1 / (ranks ** alpha)) / denominator

    def construct_transaction(self):
        """
        生成一笔完整的交易对象，包含 hash、input_addrs 和 output_addrs
        """
        while True:
            # 1. 基于 Zipf 分布选择地址
            from_idx = np.random.choice(self.n, p=self.prob_out)
            addr_from = self.source_top_list[from_idx]
            
            to_idx = np.random.choice(self.n, p=self.prob_in)
            addr_to = self.dest_top_list[to_idx]
            
            # 2. 约束检查：发送方 != 接收方
            if addr_from != addr_to:
                # 3. 【核心修改】在此处直接生成交易哈希
                # 使用时间戳 + 随机数 + 地址组合来确保 Hash 唯一性
                timestamp = time.time()
                nonce = random.random()
                raw_string = f"{addr_from}{addr_to}{timestamp}{nonce}"
                tx_hash = hashlib.sha256(raw_string.encode('utf-8')).hexdigest()
                
                # 4. 返回完整构造好的字典对象
                return {
                    "hash": tx_hash,
                    "input_addrs": [addr_from],  # 封装为列表以匹配格式
                    "output_addrs": [addr_to]    # 封装为列表以匹配格式
                }






# ==========================================
# 模块 B: 交易图绘制器 (只负责可视化)
# 入参: transaction_list (列表)
# ==========================================
def plot_transaction_graph(transaction_list, title="Transaction Graph"):
    """
    根据交易列表绘制网络图（已适配 JSON 字典格式）
    """
    # ==========================================
    # 1. 数据适配（核心修复点）
    # ==========================================
    edge_list = []
    for tx in transaction_list:
        # 兼容两种格式：
        # 情况 A: 如果是元组 (u, v) -> 直接使用
        if isinstance(tx, (tuple, list)):
            edge_list.append(tx)
        
        # 情况 B: 如果是字典 {'input_addrs': [...], 'output_addrs': [...]} -> 提取地址
        elif isinstance(tx, dict):
            # 注意：我们的生成器是单对单，所以直接取列表第一个元素
            # 如果是真实比特币数据（多对多），这里需要双重循环
            inputs = tx.get("input_addrs", [])
            outputs = tx.get("output_addrs", [])
            
            # 建立所有输入到所有输出的连边
            for src in inputs:
                for dst in outputs:
                    edge_list.append((src, dst))

    # ==========================================
    # 2. 构建图 (使用处理好的 edge_list)
    # ==========================================
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    
    # 3. 布局计算
    pos = nx.spring_layout(G, k=0.15, iterations=20, seed=42)
    
    # 4. 节点样式分析
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    total_degrees = dict(G.degree())
    
    # 识别 Top 节点
    if not G.nodes():
        print("图为空，无法绘制")
        return

    top_k = max(3, int(len(G.nodes()) * 0.1))
    sorted_nodes = sorted(G.nodes(), key=lambda x: total_degrees[x], reverse=True)
    top_nodes = set(sorted_nodes[:top_k])
    
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        deg = total_degrees.get(node, 0)
        node_sizes.append(50 + 30 * deg)
        
        if node in top_nodes:
            if out_degrees.get(node, 0) > in_degrees.get(node, 0):
                node_colors.append('#ff7f0e') # 橙色: 发送大户
            else:
                node_colors.append('#1f77b4') # 蓝色: 接收大户
        else:
            node_colors.append('#A0A0A0') # 灰色: 普通
            
    # 5. 绘图
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, arrowsize=10, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    
    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.show()

def calculate_and_fit_alpha(transactions):
    """
    读取交易数据，计算入度出度，并拟合 Zipf 分布参数 Alpha
    :param transactions: 交易列表，格式 [{'hash': '...', 'input_addrs': [...], 'output_addrs': [...]}]
    :return: (alpha_out, alpha_in)
    """
    
    # 1. 统计每个地址的出度 (Out-degree) 和 入度 (In-degree)
    # 使用 defaultdict 方便计数
    out_degree_counts = defaultdict(int)
    in_degree_counts = defaultdict(int)

    for tx in transactions:
        # 处理输入地址 -> 贡献出度 (发送)
        # 注意：Bitcoin 中同一个地址可能在同一笔交易的 inputs 中出现多次（花费多个 UTXO）
        # 这里我们按出现次数统计，也可以按“参与交易次数”统计（去重）。
        # 论文通常关注交互频率，因此统计出现次数较符合“度”的定义。
        for addr in tx.get('input_addrs', []):
            if addr: # 确保地址非空
                out_degree_counts[addr] += 1
        
        # 处理输出地址 -> 贡献入度 (接收)
        for addr in tx.get('output_addrs', []):
            if addr:
                in_degree_counts[addr] += 1

    # 2. 定义拟合逻辑
    def get_alpha_from_degrees(degree_dict, label="Data"):
        """
        内部辅助函数：给定度数计数字典，拟合 Alpha
        """
        if not degree_dict:
            print(f"[{label}] No data to fit.")
            return 0.0

        # 提取所有的度数值 (Frequencies)
        # 例如：地址A出现10次，地址B出现5次 -> [10, 5, ...]
        frequencies = np.array(list(degree_dict.values()))
        
        # 排序：从大到小 (Zipf Law: 频率越高，排名越靠前)
        sorted_freq = np.sort(frequencies)[::-1]
        
        # 归一化为概率 (Probability)
        # P(r) = count / total_count
        prob = sorted_freq / np.sum(sorted_freq)
        
        # 生成排名 (Rank): 1, 2, 3, ... N
        ranks = np.arange(1, len(prob) + 1)
        
        # 定义 Zipf 函数 (对数形式)
        # log(P) = -alpha * log(rank) + C
        # 我们使用 curve_fit 在对数空间进行拟合，这样更稳定
        def zipf_log_func(rank, alpha, intercept):
            return -alpha * np.log(rank) + intercept

        # 准备数据 (避免 log(0) 错误，虽然 rank 从1开始且 prob > 0)
        x_data = ranks
        y_data = np.log(prob)
        
        # 拟合
        # 通常只拟合头部数据（如前 50% 或前 80%），因为尾部数据噪音大
        # 这里我们取全部数据或截断尾部（例如只取 freq > 1 的数据）
        # 论文中通常是对整个分布进行拟合
        try:
            popt, pcov = curve_fit(zipf_log_func, x_data, y_data)
            alpha_fitted = popt[0]
            return alpha_fitted
        except Exception as e:
            print(f"[{label}] Fitting failed: {e}")
            return 0.0

    # 3. 分别计算 Alpha
    alpha_out = get_alpha_from_degrees(out_degree_counts, label="Out-degree")
    alpha_in = get_alpha_from_degrees(in_degree_counts, label="In-degree")
    
    return alpha_out, alpha_in

def fit_alpha():
    # 统计10个区块的正常交易
    normal_tx = []
    for i in range(923800, 923850):
        filename = f"dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        normal_tx.extend(random.sample(file_transactions, 100))

    # 2. 执行计算
    print(f"Processing {len(file_transactions)} transactions...")
    alpha_out_val, alpha_in_val = calculate_and_fit_alpha(file_transactions)
    print("-" * 30)
    print(f"Calculated Fit Parameters:")
    print(f"Alpha (Out-degree / Sender): {alpha_out_val:.4f}")
    print(f"Alpha (In-degree / Receiver): {alpha_in_val:.4f}")
    print("-" * 30)

def load_transactions_from_file(file_path):
    """
    从文件加载交易
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    print(f"✓ 从 {file_path} 成功加载 {len(transactions)} 笔交易")
    return transactions


def save_transactions_to_json(transaction_list, filename="my_transactions.json"):
    """
    直接将交易列表保存为 JSON 文件，不做额外处理
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transaction_list, f, indent=4)
        print(f"✓ 成功保存 {len(transaction_list)} 笔交易到文件: {filename}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")

if __name__ == "__main__":
    # 传输消息大小
    message_size_B = 1024
    
    # 1. 准备地址池
    pool = [f"1Addr_{i:03d}" for i in range(100)] 
    
    # 2. 初始化生成器
    # 使用你之前拟合的或者示例参数
    generator = BlockWhisperGraph(pool, alpha_out=0.4171, alpha_in=0.3484)
    

    # 计算交易数量
    num = math.ceil(message_size_B * 8 / 29)
    # 3. 生成交易数据 (此时列表里已经是完整的字典对象了)
    print(f"正在构造{num}个交易...")
    my_transactions = [generator.construct_transaction() for _ in range(num)]
    
    # 4. 保存为 JSON
    save_transactions_to_json(my_transactions, "CompareMethod/BlockWhisper/BlockWhisper_transactions.json")
    
    plot_transaction_graph(my_transactions)