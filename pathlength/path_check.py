import numpy as np
import networkx as nx

from plotgraph.main import BitcoinTransactionGraph


def check(self):
    """
    使用 Floyd-Warshall 算法计算当前图的最长路径长度。
    定义：图中任意两个可达节点之间最短距离（最少边数）的最大值。
    """
    # 1. 边界情况：如果图中没有节点，直径为 0
    if self.graph.number_of_nodes() == 0:
        return 0

    # 2. 调用 Floyd-Warshall 算法计算全点对最短路径矩阵
    # nx.floyd_warshall_numpy 返回一个 Numpy 矩阵，不可达的节点对距离为 inf
    # 对于异构图，地址A到地址B如果经过一个交易，最短距离为 2
    dist_matrix = nx.floyd_warshall_numpy(self.graph)

    # 3. 过滤掉无穷大值（不可达的路径）
    # np.isfinite 会保留矩阵中所有的有效数值（包括自身距离 0）
    mask = np.isfinite(dist_matrix)
    finite_distances = dist_matrix[mask]

    # 4. 获取最短距离中的最大值即为图直径
    if len(finite_distances) > 0:
        max_shortest_path = np.max(finite_distances)
        return int(max_shortest_path)
    else:
        return 0


if __name__ == "__main__":
    btg = BitcoinTransactionGraph()

    # 构造一条长度为 4 的路径: a1 -> tx1 -> a2 -> tx2 -> a3
    btg.add_transaction("tx1", ["a1"], ["a2"])

    # 尝试成环：a3 -> tx3 -> a1
    btg.add_transaction("tx3", ["a3"], ["a1"])
    print(f"成环后交易图的最长路径长度为: {check(btg)}")