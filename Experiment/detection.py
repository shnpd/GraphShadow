import networkx as nx
import numpy as np
from txgraph.main import BitcoinTransactionGraph
from graphanalysis.sample_transaction import (
    load_transactions_from_file,
    load_graph_cache,
)
from addressandtransaction import constuct_graph, merge_graphs
import random


# ---------------------------------------------------------
# 2. 论文核心检测算法实现
# ---------------------------------------------------------
class CovertTransactionDetector:
    def __init__(self, raw_graph_obj):
        """
        :param raw_graph_obj: BitcoinTransactionGraph 的实例
        """
        self.raw_graph = raw_graph_obj.graph

        # 论文 Table I 中定义的阈值 (Thresholds) [cite: 462]
        # 注意：论文逻辑是 |Th - D| <= epsilon 为隐蔽交易
        # 对于出度方差，正常交易方差大，隐蔽交易方差小(接近0)。
        # 论文设定 Out-degree Variance Th=1，这看起来是指正常交易的基准。
        # 但根据 Fig 4 和文字描述，隐蔽交易特征值落在特定范围内。
        # 这里的实现逻辑遵循：隐蔽交易趋向于链状，方差极低。
        self.THRESHOLDS = {
            "out_degree_variance": 1.0,
            "in_degree_variance": 0.01,  # 新增：入度方差阈值 (Table I)
            "path_ratio": 0.5,
        }

        # 论文 Section V.A.3 提到的偏移量，推荐使用 0.01 或 1 [cite: 458, 480]
        self.EPSILON = {
            "out_degree": 0.04,  # 论文推荐范围
            "in_degree": 0.04,  # 新增：入度方差的偏移量
            "path_ratio": 0.04,  # 根据 Fig 7(d) 调整
        }

    def _build_address_interaction_graph(self):
        """
        将原始的[地址-交易-地址]二部图转换为论文所需的[地址-地址]交互图。
        论文引用[cite: 194]: combine transaction bipartite graphs... to draw a transaction graph.
        """
        addr_graph = nx.DiGraph()

        # 遍历所有交易节点
        tx_nodes = [
            n
            for n, d in self.raw_graph.nodes(data=True)
            if d.get("node_type") == "transaction"
        ]

        for tx in tx_nodes:
            # 获取该交易的输入地址（前驱）和输出地址（后继）
            inputs = [u for u, v, d in self.raw_graph.in_edges(tx, data=True)]
            outputs = [v for u, v, d in self.raw_graph.out_edges(tx, data=True)]

            # 建立直接的资金流向边：Input Address -> Output Address
            for i_addr in inputs:
                for o_addr in outputs:
                    if i_addr != o_addr:  # 避免自环
                        addr_graph.add_edge(i_addr, o_addr)

        return addr_graph

    def _calculate_metrics(self, subgraph):
        """
        计算论文 Section III.C 提到的结构度量指标
        """
        num_nodes = subgraph.number_of_nodes()
        if num_nodes == 0:
            return None

        # 1. 计算出度方差 (Variance of Out-Degree) [cite: 237]
        out_degrees = [d for n, d in subgraph.out_degree()]
        var_out_degree = np.var(out_degrees)

        # 2. 新增：计算入度方差 (Variance of In-Degree) [cite: 225, 237]
        # 论文指出入度方差也是区分隐蔽交易的重要特征
        in_degrees = [d for n, d in subgraph.in_degree()]
        var_in_degree = np.var(in_degrees)

        # 3. 计算最长路径 (Longest Path Length) [cite: 231]
        longest_path = 0
        # 针对有向图计算所有节点间的最短路径长度
        # dict(nx.·(subgraph)) 返回迭代器，需要转换或遍历
        all_shortest_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
        for source, targets in all_shortest_paths.items():
            if targets:
                # 获取该源节点到所有其他可达节点的最短路径长度中的最大值
                max_dist_from_source = max(targets.values())
                if max_dist_from_source > longest_path:
                    longest_path = max_dist_from_source

        path_ratio = longest_path / num_nodes if num_nodes > 0 else 0
        return {
            "var_out_degree": var_out_degree,
            "var_in_degree": var_in_degree,  # 新增返回
            "path_ratio": path_ratio,
            "num_nodes": num_nodes,
            "num_edges": subgraph.number_of_edges(),
        }

    def detect(self):
        """
        执行检测流程，对应论文 Fig. 4
        修改：返回整张图的最终判定结果 ("Covert" 或 "Normal")
        判定标准：如果被判定为隐蔽的子图数量超过有效子图总数的 1/3，则整张图视为隐蔽。
        """
        # 1. 构建地址交互图
        G = self._build_address_interaction_graph()

        # 2. 分割连通子图
        # 注意：这里使用弱连通分量，适合有向图
        subgraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        # 初始化计数器
        valid_subgraphs_count = 0  # 满足边数要求的有效子图总数
        covert_subgraphs_count = 0  # 被判定为隐蔽特征的子图数

        for sub_g in subgraphs:
            # 过滤过小的噪声子图（通常边数太少无法计算统计特征）
            if sub_g.number_of_edges() < 3:
                continue

            metrics = self._calculate_metrics(sub_g)
            if not metrics:
                continue

            # 标记为一个有效子图
            valid_subgraphs_count += 1

            # 3. 判定逻辑 (Detection Logic)
            # 依据论文公式: |Th - D| <= epsilon

            # --- 指标 A: 出度方差 ---
            dist_out_degree = abs(
                self.THRESHOLDS["out_degree_variance"] - metrics["var_out_degree"]
            )
            is_covert_by_out = dist_out_degree <= self.EPSILON["out_degree"]

            # --- 指标 B: 入度方差 ---
            dist_in_degree = abs(
                self.THRESHOLDS["in_degree_variance"] - metrics["var_in_degree"]
            )
            is_covert_by_in = dist_in_degree <= self.EPSILON["in_degree"]

            # --- 指标 C: 路径/节点比率 ---
            dist_path = abs(self.THRESHOLDS["path_ratio"] - metrics["path_ratio"])
            is_covert_by_path = dist_path <= self.EPSILON["path_ratio"]

            # 4. 子图级综合判定 (Subgraph Verdict)
            is_subgraph_covert = False

            reasons = []
            if is_covert_by_out:
                reasons.append("Out-Degree")
            if is_covert_by_in:
                reasons.append("In-Degree")
            if is_covert_by_path:
                reasons.append("Path-Ratio")

            # 判定条件：
            # 条件1：满足2个及以上特征
            # 条件2：或者单满足最强的“出度方差”特征 (根据你之前的逻辑)
            if len(reasons) >= 2 or is_covert_by_out:
                is_subgraph_covert = True

            if is_subgraph_covert:
                covert_subgraphs_count += 1

        # 5. 图级最终判定 (Graph-level Verdict)
        # 防止除以零（如果图是空的或者全是碎片）
        if valid_subgraphs_count == 0:
            return "Normal"

        # 计算隐蔽子图占比
        ratio = covert_subgraphs_count / valid_subgraphs_count

        # 阈值判定：超过 1/3 (0.333...) 则判定为 Covert
        if ratio > (1 / 3):
            return "Covert"
        else:
            return "Normal"


# ---------------------------------------------------------
# 3. 使用示例
# ---------------------------------------------------------
# if __name__ == "__main__":
#     # 统计10个区块的正常交易
#     # normal_tx = []
#     # # 这里加载数据用于演示
#     # for i in range(923800, 923801):
#     #     filename = f"dataset/transactions_block_{i}.json"
#     #     file_transactions = load_transactions_from_file(filename)
#     #     # normal_tx.extend(file_transactions)
#     #     normal_tx.extend(random.sample(file_transactions, 600))

#     # 加载各种隐蔽交易数据
#     GraphShadow_tx = load_transactions_from_file(
#         "constructtx/GraphShadow_transactions.json"
#     )
#     DDSAC_tx = load_transactions_from_file(
#         "CompareMethod/DDSAC/DDSAC_transactions.json"
#     )
#     GBCTD_tx = load_transactions_from_file(
#         "CompareMethod/GBCTD/GBCTD_transactions.json"
#     )
#     BlockWhisper_tx = load_transactions_from_file(
#         "CompareMethod/BlockWhisper/BlockWhisper_transactions.json"
#     )
    
#     covert_filename = f"CompareMethod/DDSAC/DDSAC_transactions.json"
#     for i in range(1, 101):  # 1到100
#         # 生成文件名
#         covert_filename = f"CompareMethod/DDSAC/dataset/DDSAC_transactions_{i}.json"

#         # covert_filename = f"constructtx/dataset/GraphShadow_transactions_{i}.json"
#         covert_tx = load_transactions_from_file(covert_filename)
#         graph = constuct_graph(covert_tx)
#         # 执行检测
#         detector = CovertTransactionDetector(graph)
#         detection_results = detector.detect()
#         print(f"隐蔽交易检测结果: {detection_results}")
        
        
#         normal_filename = f"dataset/transactions_block_{923800+i}.json"
#         normal_tx = load_transactions_from_file(normal_filename)
#         graph = constuct_graph(normal_tx)
#         # 执行检测
#         detector = CovertTransactionDetector(graph)
#         detection_results = detector.detect()
#         print(f"正常交易检测结果: {detection_results}")
        

        
  
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# ==========================================
# 假设引用的外部方法 (请确保在项目中可用)
# from your_module import load_transactions_from_file, constuct_graph, CovertTransactionDetector
# ==========================================

def run_comparative_evaluation():
    # 初始化存储真实标签和预测标签
    y_true = [] 
    y_pred = []
    
    # 统计计数器 (用于实时打印)
    stats = {
        'TP': 0, # Covert 被检测为 Covert (检测成功)
        'FN': 0, # Covert 被检测为 Normal (逃逸/漏报)
        'TN': 0, # Normal 被检测为 Normal (正确放行)
        'FP': 0  # Normal 被检测为 Covert (误报)
    }

    print("Starting Comparative Evaluation (Covert vs. Normal)...")
    print("-" * 60)

    # ==========================================
    # 循环 100 组 (每组包含 1个隐蔽 + 1个正常)
    # ==========================================
    for i in range(1, 101):
        # -------------------------------------------------
        # 1. 检测隐蔽交易样本 (Positive Sample, Label=1)
        # -------------------------------------------------
        covert_filename = f"constructtx/dataset/GraphShadow_transactions_{i}.json"
        tx_list = load_transactions_from_file(covert_filename)
        graph = constuct_graph(tx_list)
        detector = CovertTransactionDetector(graph)
        
        # result 预期为 "Covert" 或 "Normal"
        res = detector.detect()
        
        # 记录数据
        y_true.append(1) # 真实是隐蔽交易
        if res == "Covert":
            y_pred.append(1)
            stats['TP'] += 1
        else:
            y_pred.append(0)
            stats['FN'] += 1
           
 

        # -------------------------------------------------
        # 2. 检测正常交易样本 (Negative Sample, Label=0)
        # -------------------------------------------------
        normal_filename = f"dataset/transactions_block_{923900+i}.json"
        tx_list = load_transactions_from_file(normal_filename)
        graph = constuct_graph(tx_list)
        detector = CovertTransactionDetector(graph)
        res = detector.detect()
        # 记录数据
        y_true.append(0) # 真实是正常交易
        if res == "Covert":
            y_pred.append(1) # 误报
            stats['FP'] += 1
        else:
            y_pred.append(0) # 正确判断
            stats['TN'] += 1

        # 每10轮打印一次简报
        if i % 10 == 0:
            print(f"Progress {i}/100 | TP:{stats['TP']} FN:{stats['FN']} TN:{stats['TN']} FP:{stats['FP']}")

    # ==========================================
    # 3. 计算最终指标
    # ==========================================
    # 防止除零错误
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)       # 针对隐蔽交易的检出率
    precision = precision_score(y_true, y_pred, zero_division=0) # 检出的样本中有多少是真的隐蔽交易
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 计算误报率 (False Positive Rate) = FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print("\n" + "="*40)
    print("【最终实验结果统计】")
    print("="*40)
    print(f"Total Samples Processed : {len(y_true)}")
    print("-" * 40)
    print(f"Accuracy (准确率)       : {acc:.2%}")
    print(f"Recall (检出率)         : {recall:.2%}  (检测器抓住了多少隐蔽交易)")
    print(f"Precision (精确率)      : {precision:.2%}")
    print(f"F1 Score               : {f1:.4f}")
    print(f"False Positive Rate    : {fpr:.2%}  (误将正常交易判为隐蔽的概率)")
    print("="*40)

    # ==========================================
    # 4. 可视化 (混淆矩阵 + 柱状图)
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- 左图：混淆矩阵 ---
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0]) 
    # 注意：这里 labels=[1,0] 是为了让左上角显示 TP (隐蔽被抓)，右下角 TN (正常放行)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], annot_kws={"size": 16},
                xticklabels=['Pred: Covert', 'Pred: Normal'],
                yticklabels=['True: Covert', 'True: Normal'])
    axes[0].set_title('Confusion Matrix', fontsize=14)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('Ground Truth')

    # --- 右图：指标统计 ---
    metrics = {
        'Accuracy': acc,
        'Recall\n(Detection Rate)': recall,
        'Precision': precision,
        'F1 Score': f1
    }
    
    # 绘制柱状图
    colors = ['#d3d3d3', '#ff9999', '#66b3ff', '#99ff99']
    bars = axes[1].bar(metrics.keys(), metrics.values(), color=colors, edgecolor='black')
    
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title('Evaluation Metrics', fontsize=14)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_comparative_evaluation()