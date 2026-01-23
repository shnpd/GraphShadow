from datetime import datetime
import random

import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
from collections import defaultdict

import numpy as np
import pandas as pd

# 从数据文件或API加载数据的函数
def load_transactions_from_file(file_path):
    """
    从文件加载交易
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    print(f"✓ 从 {file_path} 成功加载 {len(transactions)} 笔交易")
    return transactions


class BitcoinTransactionGraph:
    def __init__(self):
        """初始化比特币交易图"""
        # 使用有向图
        self.graph = nx.DiGraph()

    def add_transaction(self, tx_id, inputs, outputs):
        """
        添加一笔交易
        :param tx_id: 交易ID
        :param inputs: 输入地址列表
        :param outputs: 输出地址列表
        """
        # 添加交易节点
        self.graph.add_node(tx_id, node_type='transaction')

        # 添加输入边（地址 -> 交易）
        for addr in inputs:
            self.graph.add_node(addr, node_type='address')
            self.graph.add_edge(addr, tx_id, direction='input')

        # 添加输出边（交易 -> 地址）
        for addr in outputs:
            self.graph.add_node(addr, node_type='address')
            self.graph.add_edge(tx_id, addr, direction='output')

    def remove_transaction(self, tx_id):
        """回滚交易：删除交易节点及相关边，清理孤立节点"""
        if tx_id not in self.graph:
            return

        # 记录邻居以便检查孤立节点
        neighbors = list(self.graph.predecessors(tx_id)) + list(self.graph.successors(tx_id))
        self.graph.remove_node(tx_id)

        # 清理孤立的地址节点 (度为0)
        for node in neighbors:
            if node in self.graph and self.graph.degree(node) == 0:
                self.graph.remove_node(node)


    def calculate_diameter(self):
        """使用 Floyd-Warshall 计算图直径 (包含你的代码逻辑)"""
        # 1. 边界情况
        if self.graph.number_of_nodes() == 0:
            return 0

        # 2. 计算全点对最短路径矩阵
        # 返回 numpy matrix, 不可达为 inf
        dist_matrix = nx.floyd_warshall_numpy(self.graph)

        # 3. 过滤有效值
        mask = np.isfinite(dist_matrix)
        finite_distances = dist_matrix[mask]

        # 4. 取最大值
        if len(finite_distances) > 0:
            return int(np.max(finite_distances))
        else:
            return 0



    def visualize(self):
        plt.figure(figsize=(8, 8))
        Grepulsiveforce =0.1
        Goverlap_scaling = 1
        pos = nx.nx_agraph.pygraphviz_layout(
            self.graph,
            prog="neato",
            args=f"""
            -Grepulsiveforce={Grepulsiveforce}
            -Goverlap=prism
            -Goverlap_scaling={Goverlap_scaling}
            -Gsmoothing=triangle
            """
        )

        # 分离不同类型的节点
        address_nodes = [n for n, attr in self.graph.nodes(data=True)
                         if attr.get('node_type') == 'address']
        transaction_nodes = [n for n, attr in self.graph.nodes(data=True)
                             if attr.get('node_type') == 'transaction']

        # 分离不同类型的边
        input_edges = [(u, v) for u, v, attr in self.graph.edges(data=True)
                       if attr.get('direction') == 'input']
        output_edges = [(u, v) for u, v, attr in self.graph.edges(data=True)
                        if attr.get('direction') == 'output']
        # 图形参数
        node_size = 0.6
        node_linewidths = 0.1
        node_alpha = 0.7
        edge_width = 0.1
        edge_alpha = 0.5
        # 绘制输入边
        nx.draw_networkx_edges(self.graph, pos, edgelist=input_edges, edge_color='#666666', arrows=True,
                               arrowsize=1, alpha=edge_alpha, width=edge_width, node_shape='o',
                               node_size=node_size, )

        # 绘制输出边
        nx.draw_networkx_edges(self.graph, pos, edgelist=output_edges, edge_color='#666666', arrows=True,
                               arrowsize=1, alpha=edge_alpha, width=edge_width, node_shape='o',
                               node_size=node_size, )

        # 绘制地址节点
        nx.draw_networkx_nodes(self.graph, pos, nodelist=address_nodes, node_color='#A6CEE3', node_shape='o',
                               node_size=node_size, linewidths=node_linewidths, alpha=node_alpha)

        # 绘制交易节点
        nx.draw_networkx_nodes(self.graph, pos, nodelist=transaction_nodes, node_color='#FDBF6F', node_shape='o',
                               node_size=node_size, linewidths=node_linewidths, alpha=node_alpha)

        # 添加图例
        address_patch = mpatches.Patch(color='#A6CEE3', label='地址节点', alpha=0.8)
        transaction_patch = mpatches.Patch(color='#FDBF6F', label='交易节点', alpha=0.8)
        plt.legend(handles=[address_patch, transaction_patch],
                   loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
        plt.title("比特币交易图", fontsize=12, fontweight='bold')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            f"bitcoin_tx_graph_{timestamp}.pdf",
            format="pdf", bbox_inches="tight"
        )

    def get_graph_info(self):
        """获取图的基本信息"""
        print("=== 比特币交易图信息 ===")
        print(f"节点总数: {self.graph.number_of_nodes()}")
        print(f"地址节点数: {len([n for n, attr in self.graph.nodes(data=True)
                                  if attr.get('node_type') == 'address'])}")
        print(f"交易节点数: {len([n for n, attr in self.graph.nodes(data=True)
                                  if attr.get('node_type') == 'transaction'])}")
        print(f"边总数: {self.graph.number_of_edges()}")
        print(f"输入边数: {len([1 for _, _, attr in self.graph.edges(data=True)
                                if attr.get('direction') == 'input'])}")
        print(f"输出边数: {len([1 for _, _, attr in self.graph.edges(data=True)
                                if attr.get('direction') == 'output'])}")
        print("=" * 30)

    def analyze_node_in_degree_distribution_to_excel(self, node_type=None):
        """
        统计节点入度分布，并将结果保存为 Excel
        第一行：入度
        第二行：对应数量

        :param node_type: 节点类型（如 "transaction" / "address"），None 表示全部节点
        :return: dict {in_degree: count}
        """
        in_degree_dist = defaultdict(int)

        for node, attr in self.graph.nodes(data=True):
            if node_type is not None and attr.get("node_type") != node_type:
                continue

            in_deg = self.graph.in_degree(node)
            in_degree_dist[in_deg] += 1

        # 排序后的入度和值
        degrees = sorted(in_degree_dist.keys())
        counts = [in_degree_dist[k] for k in degrees]

        # 构造 DataFrame（两行）
        df = pd.DataFrame([degrees, counts])

        # 文件名
        name = node_type if node_type is not None else "all"
        filename = f"{name}入度.xlsx"

        # 保存为 Excel（不保存 index 和 header）
        df.to_excel(filename, index=False, header=False)

        print(f"\n入度分布已保存为 Excel 文件：{filename}")

        return dict(in_degree_dist)

# 在创建图表前调用
if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题`

    all_transactions = []
    # 统计10个区块
    for i in range(928050, 928060):
        filename = f"../dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        all_transactions.extend(random.sample(file_transactions, 100))
    with open("sampled_elements.json", "w") as f:
        json.dump(all_transactions, f)
    # with open("sampled_elements.json", "r") as f:
    #     all_transactions = json.load(f)

    # all_transactions = load_transactions_from_file("dataset/transactions_block_928053.json")
    # all_transactions = all_transactions[:500]
    # plot
    print("\n正在生成交易图...")
    btg = BitcoinTransactionGraph()
    for tx in all_transactions:
        btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
    btg.get_graph_info()
    btg.visualize()
    # btg.analyze_degree_distribution()

