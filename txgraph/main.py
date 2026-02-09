from datetime import datetime
import random

import networkx as nx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import math

# 从数据文件或API加载数据的函数
def load_transactions_from_file(file_path):
    """
    从文件加载交易
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    print(f"✓ 从 {file_path} 成功加载 {len(transactions)} 笔交易")
    return transactions

VIS_CONFIG = {
    # 节点大小
    'SIZE_BG': 10,           # 背景正常节点
    'SIZE_COVERT': 15,       # 隐蔽节点 (Tx 和 Addr)
    
    # 颜色与透明度
    'COLOR_BG': '#C0C0C0',     # 背景灰色
    'COLOR_ADDR': '#1E90FF',   # 隐蔽地址蓝色
    'COLOR_TX': '#D62728',     # 隐蔽交易红色
    'COLOR_EDGE': '#FF4500',   # 隐蔽边橙红色
    
    'ALPHA_BG_NODE': 0.6,      # 背景节点透明度
    'ALPHA_BG_EDGE': 0.4,      # 背景边透明度
    'ALPHA_COVERT': 1.0,       # 隐蔽元素透明度
    
    # 线条与箭头
    'WIDTH_BG_EDGE': 0.5,      # 背景边宽
    'WIDTH_COVERT_EDGE': 0.8,  # 隐蔽边宽
    'ARROW_SIZE': 5,           # 箭头大小
    
    # 字体与画布
    'FONT_SIZE_TITLE': 14,
    'FONT_SIZE_LEGEND': 10,
    'DPI': 200,
    'FIG_SIZE': (12, 12)
}

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
        
    def calculate_diameter_nfx(self):
        """
        将图视为无向图，使用 Floyd-Warshall 计算图直径
        (计算的是最大弱连通分量的拓扑直径)
        """
        # 1. 边界情况
        if self.graph.number_of_nodes() == 0:
            return 0

        # 2. 转换为无向图 (核心修改)
        # to_undirected() 会创建一个新的图副本，其中所有的有向边都被转换为无向边
        # 这样 A->B 的连接，现在 B 也可以到达 A，距离为 1
        undirected_G = self.graph.to_undirected()

        # 3. 计算全点对最短路径矩阵
        # 注意：Floyd-Warshall 的复杂度是 O(N^3)，仅适用于节点数较少的情况 (<1000)
        dist_matrix = nx.floyd_warshall_numpy(undirected_G)

        # 4. 过滤有效值 (处理不连通的情况)
        # 在无向图中，如果不连通，距离依然是 inf，isfinite 会自动过滤掉
        mask = np.isfinite(dist_matrix)
        finite_distances = dist_matrix[mask]

        # 5. 取最大值
        if len(finite_distances) > 0:
            return int(np.max(finite_distances))
        else:
            return 0
    def visualize(self):
        # ✅ 1. 设置中文字体 (Windows下通常使用 SimHei)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        # ✅ 2. 解决负号 '-' 显示为方块的问题
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(8, 8))
        Grepulsiveforce = 0.1
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


    def visualize_GraphShadow_covert(self, covert_tx_ids=None):
        """
        针对 3000+ 节点优化的可视化方法。
        调整：增强背景正常交易的可见度，同时保持隐蔽交易的高亮。
        """
        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 画布大小保持
        plt.figure(figsize=(12, 12), dpi=200) 
        
        print(f"正在计算布局 (节点数: {self.graph.number_of_nodes()})...")

        # 2. 布局计算
        try:
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                # 稍微减小一点排斥力，让背景稍微聚拢一点，看起来更紧凑
                args="-Grepulsiveforce=1.2 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception as e:
            print(f"PyGraphviz layout failed, using spring_layout... ({e})")
            pos = nx.spring_layout(self.graph, k=0.04, iterations=100, seed=42)

        # ==========================
        # 3. 节点筛选与归类
        # ==========================
        
        # A. 找出隐蔽地址
        covert_related_addrs = set()
        for tx_id in covert_tx_ids:
            if tx_id in self.graph:
                neighbors = list(self.graph.successors(tx_id)) + list(self.graph.predecessors(tx_id))
                covert_related_addrs.update(neighbors)
                
        # B. 节点分类
        bg_nodes = []       # 背景噪音 (灰色)
        covert_addrs = []   # 隐蔽地址 (蓝色)
        covert_txs = []     # 隐蔽交易 (红色)
        
        for n, attr in self.graph.nodes(data=True):
            if n in covert_tx_ids:
                covert_txs.append(n)
            elif n in covert_related_addrs:
                covert_addrs.append(n)
            else:
                bg_nodes.append(n)

        # C. 边分类
        bg_edges = []
        covert_edges = []
        all_covert_nodes = covert_tx_ids.union(covert_related_addrs)
        
        for u, v in self.graph.edges():
            if u in all_covert_nodes and v in all_covert_nodes:
                covert_edges.append((u, v))
            else:
                bg_edges.append((u, v))

        print("开始绘制...")
        ax = plt.gca()

        # ==========================
        # 4. 精细化绘制 (尺寸调整)
        # ==========================
        
        # --- Layer 1: 背景 (增强可见度) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color=VIS_CONFIG['COLOR_BG'], 
                            node_size=VIS_CONFIG['SIZE_BG'], 
                            alpha=VIS_CONFIG['ALPHA_BG_NODE'],           
                            linewidths=0, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color=VIS_CONFIG['COLOR_BG'],
                            arrows=False, 
                            width=VIS_CONFIG['WIDTH_BG_EDGE'],           
                            alpha=VIS_CONFIG['ALPHA_BG_EDGE'],          
                            ax=ax)

        # --- Layer 2: 隐蔽链路 (Link Layer) ---
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color=VIS_CONFIG['COLOR_EDGE'], 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            arrowstyle='-|>', arrowsize=VIS_CONFIG['ARROW_SIZE'], 
                            width=VIS_CONFIG['WIDTH_COVERT_EDGE'], alpha=VIS_CONFIG['ALPHA_COVERT'], # 隐蔽连线设为完全不透明
                            ax=ax)

        # --- Layer 3: 隐蔽地址 (Blue Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color=VIS_CONFIG['COLOR_ADDR'], node_shape='o', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # --- Layer 4: 隐蔽交易 (Red Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color=VIS_CONFIG['COLOR_TX'], node_shape='^', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 5. 图例与保存
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        plt.title(f"GraphShadow交易拓扑图", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GraphShadow_mixed_graph_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()
        
    def visualize_GBCTD_covert(self, covert_tx_ids=None):
        """
        针对 3000+ 节点优化的可视化方法。
        调整：增强背景正常交易的可见度，同时对隐蔽交易进行局部布局优化（拉开距离）。
        """
        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 画布大小保持
        plt.figure(figsize=(12, 12), dpi=200) 
        
        print(f"正在计算布局 (节点数: {self.graph.number_of_nodes()})...")

        # ==========================
        # 2. 全局布局计算 (保持原有的背景布局)
        # ==========================
        try:
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                # 稍微减小一点排斥力，让背景稍微聚拢一点，看起来更紧凑
                args="-Grepulsiveforce=1.2 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception as e:
            print(f"PyGraphviz layout failed, using spring_layout... ({e})")
            pos = nx.spring_layout(self.graph, k=0.04, iterations=100, seed=42)

        # ==========================
        # 3. 节点筛选与归类
        # ==========================
        
        # A. 找出隐蔽地址
        covert_related_addrs = set()
        for tx_id in covert_tx_ids:
            if tx_id in self.graph:
                neighbors = list(self.graph.successors(tx_id)) + list(self.graph.predecessors(tx_id))
                covert_related_addrs.update(neighbors)
        
        all_covert_nodes = list(covert_tx_ids.union(covert_related_addrs))

        # ==========================
        # 【核心修改】 隐蔽交易局部布局优化
        # ==========================
        if len(all_covert_nodes) > 1:
            print("正在优化隐蔽交易拓扑布局...")
            # 1. 提取隐蔽子图
            covert_subgraph = self.graph.subgraph(all_covert_nodes)
            
            # 2. 计算子图的重心 (以便稍后放回原位)
            # 获取当前这些节点在全局图中的坐标平均值
            xs = [pos[n][0] for n in all_covert_nodes]
            ys = [pos[n][1] for n in all_covert_nodes]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            # 3. 对子图单独应用布局算法 (增加 k 值以拉开距离)
            # k 值越大，节点间斥力越大；iterations 增加以保证舒展
            sub_pos = nx.spring_layout(covert_subgraph, k=0.8, iterations=100, seed=101)
            
            # 4. 坐标变换：缩放并平移回原重心
            # spring_layout 生成的坐标通常在 (-1, 1) 之间，需要根据背景图的尺度进行缩放
            # 计算背景图的大致跨度
            all_xs = [c[0] for c in pos.values()]
            all_ys = [c[1] for c in pos.values()]
            span_x = max(all_xs) - min(all_xs)
            span_y = max(all_ys) - min(all_ys)
            
            # 缩放因子：让隐蔽子图占据大约背景图 1/5 到 1/4 的大小，确保能看清
            scale_factor = min(span_x, span_y) * 0.20 
            
            for node, coords in sub_pos.items():
                # 新坐标 = 重心 + (局部坐标 * 缩放因子)
                new_x = center_x + coords[0] * scale_factor
                new_y = center_y + coords[1] * scale_factor
                pos[node] = (new_x, new_y)

        # ==========================
        # 接下来继续原有的分类与绘制逻辑
        # ==========================

        # B. 节点分类
        bg_nodes = []       # 背景噪音 (灰色)
        covert_addrs = []   # 隐蔽地址 (蓝色)
        covert_txs = []     # 隐蔽交易 (红色)
        
        for n, attr in self.graph.nodes(data=True):
            if n in covert_tx_ids:
                covert_txs.append(n)
            elif n in covert_related_addrs:
                covert_addrs.append(n)
            else:
                bg_nodes.append(n)

        # C. 边分类
        bg_edges = []
        covert_edges = []
        
        # 将 set 转换为 frozen set 以加速查找 (或者直接用上面的 list)
        covert_node_set = set(all_covert_nodes)
        
        for u, v in self.graph.edges():
            if u in covert_node_set and v in covert_node_set:
                covert_edges.append((u, v))
            else:
                bg_edges.append((u, v))

        print("开始绘制...")
        ax = plt.gca()

        # ==========================
        # 4. 精细化绘制 (尺寸调整)
        # ==========================
        
        # --- 尺寸参数定义 (修改点) ---

        # --- Layer 1: 背景 (增强可见度) ---
        # 节点：颜色稍微加深，透明度提高
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color=VIS_CONFIG['COLOR_BG'], 
                            node_size=VIS_CONFIG['SIZE_BG'], 
                            alpha=VIS_CONFIG['ALPHA_BG_NODE'],           
                            linewidths=0, ax=ax)
        
        # 边：线条加粗，透明度提高
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color=VIS_CONFIG['COLOR_BG'],
                            arrows=False, 
                            width=VIS_CONFIG['WIDTH_BG_EDGE'],           
                            alpha=VIS_CONFIG['ALPHA_BG_EDGE'],          
                            ax=ax)

        # --- Layer 2: 隐蔽链路 (Link Layer) ---
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color=VIS_CONFIG['COLOR_EDGE'], 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            arrowstyle='-|>', arrowsize=VIS_CONFIG['ARROW_SIZE'], 
                            width=VIS_CONFIG['WIDTH_COVERT_EDGE'], alpha=VIS_CONFIG['ALPHA_COVERT'], # 隐蔽连线设为完全不透明
                            ax=ax)

        # --- Layer 3: 隐蔽地址 (Blue Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color=VIS_CONFIG['COLOR_ADDR'], node_shape='o', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # --- Layer 4: 隐蔽交易 (Red Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color=VIS_CONFIG['COLOR_TX'], node_shape='^', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 5. 图例与保存
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        plt.title(f"GBCTD交易拓扑图", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"GBCTD_mixed_graph_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()

    def visualize_DDSAC_covert(self, covert_tx_ids=None):
        """
        基于 Neato 布局 + 1输入2输出 Peeling Chain 结构优化。
        
        【样式约束】：
        背景节点和边的样式严格参照用户提供的 visualize_covert 函数参数。
        隐蔽节点采用几何重构以展示清晰的拓扑结构。
        """
        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 12), dpi=200) # 保持画布大小
        
        print(f"正在计算全局布局 (节点数: {self.graph.number_of_nodes()})...")

        # ==========================
        # 2. 全局布局 (Neato) - 参照您的参数
        # ==========================
        try:
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                # 使用您指定的参数：斥力 1.2
                args="-Grepulsiveforce=1.2 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception as e:
            print(f"PyGraphviz layout failed ({e}), fallback to spring...")
            pos = nx.spring_layout(self.graph, k=0.04, iterations=100, seed=42)

        # ==========================
        # 3. 链式结构优化 (几何重构逻辑)
        # ==========================
        print("正在执行 Peeling Chain 结构优化...")

        # A. 获取边界
        all_coords = np.array(list(pos.values()))
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        width, height = x_max - x_min, y_max - y_min
        
        # 定义安全区域
        margin = 0.05
        safe_x_min, safe_x_max = x_min + width*margin, x_max - width*margin
        safe_y_min, safe_y_max = y_min + height*margin, y_max - height*margin

        # B. 提取隐蔽子图
        covert_related_addrs = set()
        for tx_id in covert_tx_ids:
            if tx_id in self.graph:
                neighbors = list(self.graph.successors(tx_id)) + list(self.graph.predecessors(tx_id))
                covert_related_addrs.update(neighbors)
        all_covert_nodes = covert_tx_ids.union(covert_related_addrs)
        
        if all_covert_nodes:
            covert_subgraph = self.graph.subgraph(all_covert_nodes)
            
            # 寻找链头
            chain_heads = [n for n in covert_subgraph.nodes() if covert_subgraph.in_degree(n) == 0]
            if not chain_heads: chain_heads = [list(all_covert_nodes)[0]]

            # 识别 骨架 (Spine) 和 叶子 (Leaves)
            spines = []     
            leaf_map = {}   
            visited_global = set()

            for head in chain_heads:
                if head in visited_global: continue
                
                spine = []
                curr = head
                
                while True:
                    visited_global.add(curr)
                    spine.append(curr)
                    
                    succs = [n for n in covert_subgraph.successors(curr) if n not in visited_global]
                    
                    if not succs:
                        break 
                    
                    if len(succs) == 1:
                        curr = succs[0]
                    else:
                        # 1-in-2-out 结构处理
                        next_node = None
                        leaf_node = None
                        
                        candidates_for_next = [s for s in succs if covert_subgraph.out_degree(s) > 0]
                        
                        if candidates_for_next:
                            next_node = candidates_for_next[0]
                            remaining = [s for s in succs if s != next_node]
                            if remaining: leaf_node = remaining[0]
                        else:
                            next_node = succs[0]
                            if len(succs) > 1: leaf_node = succs[1]
                        
                        if leaf_node:
                            leaf_map[curr] = leaf_node 
                            visited_global.add(leaf_node)
                            
                        curr = next_node

                if len(spine) > 1:
                    spines.append(spine)

            # C. 几何整形
            avg_span = (width + height) / 2.0
            BASE_SPACING = avg_span / 35.0 
            LEAF_SPACING = BASE_SPACING * 0.8 
            WAVE_AMPLITUDE = BASE_SPACING * 0.4

            for spine in spines:
                orig_points = np.array([pos[n] for n in spine])
                centroid = np.mean(orig_points, axis=0)
                
                random_angle = random.uniform(0, 2 * math.pi)
                u_vec = np.array([math.cos(random_angle), math.sin(random_angle)]) 
                v_vec = np.array([-u_vec[1], u_vec[0]]) 
                
                spine_coords = []
                current_dist = 0.0
                node_local_pos_map = {} 

                for i, node in enumerate(spine):
                    if i > 0:
                        current_dist += BASE_SPACING * random.uniform(0.7, 1.3)
                    
                    lateral_offset = random.uniform(-WAVE_AMPLITUDE, WAVE_AMPLITUDE)
                    local_pos = (u_vec * current_dist) + (v_vec * lateral_offset)
                    spine_coords.append(local_pos)
                    node_local_pos_map[node] = local_pos
                
                spine_coords = np.array(spine_coords)
                local_center = np.mean(spine_coords, axis=0)
                final_spine_coords = (spine_coords - local_center) + centroid

                # 边界修正
                c_x_min, c_x_max = np.min(final_spine_coords[:, 0]), np.max(final_spine_coords[:, 0])
                c_y_min, c_y_max = np.min(final_spine_coords[:, 1]), np.max(final_spine_coords[:, 1])
                
                shift_x, shift_y = 0, 0
                if c_x_min < safe_x_min: shift_x = safe_x_min - c_x_min
                elif c_x_max > safe_x_max: shift_x = safe_x_max - c_x_max
                if c_y_min < safe_y_min: shift_y = safe_y_min - c_y_min
                elif c_y_max > safe_y_max: shift_y = safe_y_max - c_y_max
                
                final_shift = np.array([shift_x, shift_y])
                final_spine_coords += final_shift
                
                for i, node in enumerate(spine):
                    pos[node] = final_spine_coords[i]
                    
                    if node in leaf_map:
                        leaf = leaf_map[node]
                        side_direction = 1 if i % 2 == 0 else -1 
                        leaf_pos_local = node_local_pos_map[node] + (v_vec * LEAF_SPACING * side_direction * 1.5)
                        leaf_pos_final = (leaf_pos_local - local_center) + centroid + final_shift
                        pos[leaf] = leaf_pos_final

        # ==========================
        # 4. 节点筛选与归类
        # ==========================
        bg_nodes = []
        covert_addrs = [] 
        covert_txs = []   
        
        for n in self.graph.nodes():
            if n in covert_tx_ids:
                covert_txs.append(n)
            elif n in covert_related_addrs:
                covert_addrs.append(n)
            else:
                bg_nodes.append(n)

        bg_edges = []
        covert_edges = [] # 统一的隐蔽边
        
        for u, v in self.graph.edges():
            if u in all_covert_nodes and v in all_covert_nodes:
                covert_edges.append((u, v))
            else:
                bg_edges.append((u, v))

        # ==========================
        # 5. 精细化绘制
        # ==========================
        print("开始绘制...")
        ax = plt.gca()
        
        # --- 尺寸参数 (严格参照您的输入) ---
        
        # --- Layer 1: 背景 (严格参照您的输入样式) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color=VIS_CONFIG['COLOR_BG'], 
                            node_size=VIS_CONFIG['SIZE_BG'], 
                            alpha=VIS_CONFIG['ALPHA_BG_NODE'],           
                            linewidths=0, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color=VIS_CONFIG['COLOR_BG'],
                            arrows=False, 
                            width=VIS_CONFIG['WIDTH_BG_EDGE'],           
                            alpha=VIS_CONFIG['ALPHA_BG_EDGE'],          
                            ax=ax)

        # --- Layer 2: 隐蔽链路 (Link Layer) ---
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color=VIS_CONFIG['COLOR_EDGE'], 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            arrowstyle='-|>', arrowsize=VIS_CONFIG['ARROW_SIZE'], 
                            width=VIS_CONFIG['WIDTH_COVERT_EDGE'], alpha=VIS_CONFIG['ALPHA_COVERT'], # 隐蔽连线设为完全不透明
                            connectionstyle="arc3,rad=0.05",
                            ax=ax)

        # --- Layer 3: 隐蔽地址 (Blue Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color=VIS_CONFIG['COLOR_ADDR'], node_shape='o', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # --- Layer 4: 隐蔽交易 (Red Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color=VIS_CONFIG['COLOR_TX'], node_shape='^', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 6. 图例与保存
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        plt.title(f"DDSAC交易拓扑图", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DDSAC_mixed_graph_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()       
    
    def visualize_BlockWhisper_covert(self, covert_tx_ids=None):
        """
        优化版可视化：将隐蔽交易“适当分散”在背景图的中心区域，
        既保留清晰的内部结构，又不占满整幅画面。
        """
        import numpy as np

        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(12, 12), dpi=200)
        
        print(f"正在计算布局 (节点数: {self.graph.number_of_nodes()})...")

        # 2. 初始布局计算 (全图计算)
        try:
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                args="-Grepulsiveforce=1.2 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception as e:
            print(f"PyGraphviz layout failed, using spring_layout... ({e})")
            pos = nx.spring_layout(self.graph, k=0.04, iterations=100, seed=42)

        # ==========================
        # 3. 节点识别与分类
        # ==========================
        covert_related_addrs = set()
        for tx_id in covert_tx_ids:
            if tx_id in self.graph:
                neighbors = list(self.graph.successors(tx_id)) + list(self.graph.predecessors(tx_id))
                covert_related_addrs.update(neighbors)
        
        all_covert_nodes = covert_tx_ids.union(covert_related_addrs)
        bg_nodes = [n for n in self.graph.nodes() if n not in all_covert_nodes]

        # ==========================
        # 4. 隐蔽交易局部优化 (核心修改：局部区域映射)
        # ==========================
        if len(all_covert_nodes) > 1 and len(bg_nodes) > 1:
            print("正在优化隐蔽交易分布 (局部适当分散)...")
            
            # A. 计算背景图的物理参数
            bg_coords = np.array([pos[n] for n in bg_nodes])
            bg_min = bg_coords.min(axis=0)
            bg_max = bg_coords.max(axis=0)
            bg_width = bg_max[0] - bg_min[0]
            bg_height = bg_max[1] - bg_min[1]
            
            # 目标中心点：放在背景图的几何中心
            target_center = (bg_min + bg_max) / 2
            
            # B. 计算隐蔽交易当前的物理参数
            cov_coords = np.array([pos[n] for n in all_covert_nodes])
            cov_min = cov_coords.min(axis=0)
            cov_max = cov_coords.max(axis=0)
            cov_center = cov_coords.mean(axis=0) # 使用重心
            cov_width = cov_max[0] - cov_min[0]
            cov_height = cov_max[1] - cov_min[1]
            
            # 防止除零
            cov_width = max(cov_width, 1e-5)
            cov_height = max(cov_height, 1e-5)

            # C. 计算缩放比例
            # 【关键修改】这里控制“适当分散”的程度。
            # 0.35 表示隐蔽交易群将占据背景图宽度的 35% 左右。
            # 这个比例既能让节点散开看清结构，又不会占满全图。
            TARGET_SCALE_RATIO = 0.75 
            
            scale_x = (bg_width * TARGET_SCALE_RATIO) / cov_width
            scale_y = (bg_height * TARGET_SCALE_RATIO) / cov_height
            final_scale = min(scale_x, scale_y) # 保持原始长宽比
            
            # D. 应用变换
            for n in all_covert_nodes:
                original_pos = np.array(pos[n])
                
                # 1. 归一化 (相对于隐蔽群重心)
                relative = original_pos - cov_center
                
                # 2. 缩放并移动到目标中心
                new_pos = target_center + (relative * final_scale)
                
                pos[n] = tuple(new_pos)

        # ==========================
        # 5. 绘制图层 (保持不变)
        # ==========================
        covert_txs = [n for n in all_covert_nodes if n in covert_tx_ids]
        covert_addrs = [n for n in all_covert_nodes if n not in covert_tx_ids]

        bg_edges = []
        covert_edges = []
        for u, v in self.graph.edges():
            if u in all_covert_nodes and v in all_covert_nodes:
                covert_edges.append((u, v))
            else:
                bg_edges.append((u, v))

        ax = plt.gca()
        
        # Layer 1: 背景
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color=VIS_CONFIG['COLOR_BG'], 
                            node_size=VIS_CONFIG['SIZE_BG'], 
                            alpha=VIS_CONFIG['ALPHA_BG_NODE'],           
                            linewidths=0, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color=VIS_CONFIG['COLOR_BG'],
                            arrows=False, 
                            width=VIS_CONFIG['WIDTH_BG_EDGE'],           
                            alpha=VIS_CONFIG['ALPHA_BG_EDGE'],          
                            ax=ax)

        # Layer 2: 隐蔽链路
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color=VIS_CONFIG['COLOR_EDGE'], 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            arrowstyle='-|>', arrowsize=VIS_CONFIG['ARROW_SIZE'], 
                            width=VIS_CONFIG['WIDTH_COVERT_EDGE'], alpha=VIS_CONFIG['ALPHA_COVERT'], # 隐蔽连线设为完全不透明
                            ax=ax)

        # Layer 3: 隐蔽地址
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color=VIS_CONFIG['COLOR_ADDR'], node_shape='o', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # Layer 4: 隐蔽交易
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color=VIS_CONFIG['COLOR_TX'], node_shape='^', 
                            node_size=VIS_CONFIG['SIZE_COVERT'], 
                            alpha=VIS_CONFIG['ALPHA_COVERT'], linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 6. 图例与保存
        # ==========================
        import matplotlib.patches as mpatches # 确保导入
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title(f"BlockWhisper交易拓扑图", fontsize=14)
        plt.axis('off')

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BlockWhisper_mixed_graph_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()
    
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
    all_transactions = []
    # 统计10个区块
    for i in range(928050, 928051):
        filename = f"dataset/transactions_block_{i}.json"
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
    btg.calculate_diameter()
    btg.get_graph_info()
    btg.visualize()
    # btg.analyze_degree_distribution()
