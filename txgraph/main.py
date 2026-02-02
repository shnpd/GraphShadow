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


    def visualize_covert(self, covert_tx_ids=None):
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
        
        # --- 尺寸参数定义 (修改点) ---
        SIZE_BG = 10        # 【修改】从 2 增大到 10，让正常节点清晰可见
        SIZE_COVERT = 25    # 隐蔽节点保持稍大，形成对比
        
        # --- Layer 1: 背景 (增强可见度) ---
        # 节点：颜色稍微加深，透明度提高
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color='#C0C0C0', # 【修改】颜色加深一点 (原 #E0E0E0)
                            node_size=SIZE_BG, 
                            alpha=0.6,            # 【修改】透明度提高 (原 0.3)
                            linewidths=0, ax=ax)
        
        # 边：线条加粗，透明度提高
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color='#C0C0C0', # 【修改】颜色加深
                            arrows=False, 
                            width=0.5,            # 【修改】线条加粗 (原 0.2)
                            alpha=0.4,            # 【修改】透明度提高 (原 0.2)
                            ax=ax)

        # --- Layer 2: 隐蔽链路 (Link Layer) ---
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color='#FF4500', 
                            node_size=SIZE_COVERT, 
                            arrowstyle='-|>', arrowsize=8, 
                            width=1.0, alpha=1.0, # 隐蔽连线设为完全不透明
                            ax=ax)

        # --- Layer 3: 隐蔽地址 (Blue Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color='#1E90FF', node_shape='o', 
                            node_size=SIZE_COVERT, 
                            alpha=1.0, linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # --- Layer 4: 隐蔽交易 (Red Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color='#D62728', node_shape='^', 
                            node_size=SIZE_COVERT, 
                            alpha=1.0, linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 5. 图例与保存
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易 (Tx)'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址 (Addr)'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        plt.title(f"比特币混合交易图谱", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bitcoin_vis_clear_bg_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()
        


    def visualize_chain_covert(self, covert_tx_ids=None):
        """
        基于 Neato 布局的优化可视化 (边界约束 + 随机边长版)。
        
        改进点：
        1. 缩短边长：让链条更紧凑。
        2. 随机边长：每一步的距离随机化，不再是死板的等长。
        3. 边界约束：计算生成的链条坐标，如果超出背景图范围，强制将其平移回图内。
        """
        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(14, 14), dpi=200) 
        
        print(f"正在计算布局 (节点数: {self.graph.number_of_nodes()})...")

        # ==========================
        # 2. 全局布局 (Neato)
        # ==========================
        try:
            # 使用 Neato 保持全局拓扑
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                # 适当减小排斥力，让背景紧凑一些
                args="-Grepulsiveforce=1.0 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception as e:
            print(f"PyGraphviz layout failed ({e}), falling back to spring_layout...")
            pos = nx.spring_layout(self.graph, k=0.03, iterations=100, seed=42)

        # ==========================
        # 3. 链式结构优化 (核心修改)
        # ==========================
        print("正在执行链式结构优化 (随机化 + 边界检查)...")

        # A. 获取背景图的边界范围 (用于限制隐蔽链)
        all_coords = np.array(list(pos.values()))
        x_min, x_max = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
        y_min, y_max = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])
        
        # 留出 5% 的边距 margin，防止贴边
        width = x_max - x_min
        height = y_max - y_min
        margin_x = width * 0.05
        margin_y = height * 0.05
        safe_x_min, safe_x_max = x_min + margin_x, x_max - margin_x
        safe_y_min, safe_y_max = y_min + margin_y, y_max - margin_y

        # B. 识别隐蔽链
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

            chains = []
            visited = set()
            for head in chain_heads:
                if head in visited: continue
                chain = []
                curr = head
                while True:
                    visited.add(curr)
                    chain.append(curr)
                    succs = [n for n in covert_subgraph.successors(curr) if n not in visited]
                    if not succs: break
                    curr = succs[0]
                if len(chain) > 1:
                    chains.append(chain)

            # C. 几何整形
            
            # 【修改1】缩小基础步长 (从 1/20 缩小到 1/35)
            avg_span = (width + height) / 2.0
            BASE_SPACING = avg_span / 35.0 
            
            # 【修改2】波动幅度
            WAVE_AMPLITUDE = BASE_SPACING * 0.4 

            for chain in chains:
                # 获取原位置以计算重心
                orig_points = np.array([pos[n] for n in chain])
                centroid = np.mean(orig_points, axis=0)
                
                # --- 确定方向 (随机旋转) ---
                random_angle = random.uniform(0, 2 * math.pi) 
                u_vec = np.array([math.cos(random_angle), math.sin(random_angle)]) # 前进方向
                v_vec = np.array([-u_vec[1], u_vec[0]]) # 侧向波动方向
                
                # --- 预计算整条链的形状 (以 (0,0) 为原点) ---
                chain_local_coords = []
                current_dist = 0.0
                
                # 链的总长度将动态计算
                for i in range(len(chain)):
                    # 【修改3】每段长度随机化 (0.6 ~ 1.4 倍基础步长)
                    if i > 0:
                        segment_len = BASE_SPACING * random.uniform(0.6, 1.4)
                        current_dist += segment_len
                    
                    # 侧向波动
                    lateral_offset = random.uniform(-WAVE_AMPLITUDE, WAVE_AMPLITUDE)
                    
                    # 局部坐标 = 前进距离 * u + 侧向偏移 * v
                    local_pos = (u_vec * current_dist) + (v_vec * lateral_offset)
                    chain_local_coords.append(local_pos)
                
                chain_local_coords = np.array(chain_local_coords)
                
                # 将局部坐标的中心对齐到 (0,0)
                local_center = np.mean(chain_local_coords, axis=0)
                chain_centered = chain_local_coords - local_center
                
                # --- 将链放置到重心位置 ---
                final_coords = chain_centered + centroid
                
                # --- 【修改4】边界检查与修正 (Shift Strategy) ---
                # 检查生成的链条是否超出了安全边界
                c_x_min, c_x_max = np.min(final_coords[:, 0]), np.max(final_coords[:, 0])
                c_y_min, c_y_max = np.min(final_coords[:, 1]), np.max(final_coords[:, 1])
                
                shift_x = 0
                shift_y = 0
                
                # 如果左边超了，往右移
                if c_x_min < safe_x_min: 
                    shift_x = safe_x_min - c_x_min
                # 如果右边超了，往左移 (注意：如果链条极长，可能需要缩放，这里优先平移)
                elif c_x_max > safe_x_max: 
                    shift_x = safe_x_max - c_x_max
                    
                if c_y_min < safe_y_min: 
                    shift_y = safe_y_min - c_y_min
                elif c_y_max > safe_y_max: 
                    shift_y = safe_y_max - c_y_max
                
                # 应用平移
                final_coords += np.array([shift_x, shift_y])
                
                # --- 更新 pos ---
                for i, node in enumerate(chain):
                    pos[node] = final_coords[i]

        # ==========================
        # 4. 节点筛选
        # ==========================
        bg_nodes = []
        covert_addrs = []
        covert_txs = []
        
        for n, attr in self.graph.nodes(data=True):
            if n in covert_tx_ids:
                covert_txs.append(n)
            elif n in covert_related_addrs:
                covert_addrs.append(n)
            else:
                bg_nodes.append(n)

        bg_edges = []
        covert_edges = []
        for u, v in self.graph.edges():
            if u in all_covert_nodes and v in all_covert_nodes:
                covert_edges.append((u, v))
            else:
                bg_edges.append((u, v))

        # ==========================
        # 5. 绘制
        # ==========================
        print("开始绘制...")
        ax = plt.gca()
        
        SIZE_BG = 10        
        SIZE_COVERT = 40    
        
        # Layer 1: 背景
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color='#C0C0C0', node_size=SIZE_BG, 
                            alpha=0.5, linewidths=0, ax=ax)
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color='#D0D0D0', arrows=False, 
                            width=0.4, alpha=0.3, ax=ax)

        # Layer 2: 隐蔽链连线 (橙色)
        # 使用 arc3,rad=0.05 增加一点点自然的弯曲感
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color='#FF4500', 
                            node_size=SIZE_COVERT,  
                            arrowstyle='-|>', arrowsize=10, 
                            width=1.2, alpha=1.0, 
                            connectionstyle="arc3,rad=0.05",
                            ax=ax)

        # Layer 3: 隐蔽地址 (蓝色圆点)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color='#1E90FF', node_shape='o', 
                            node_size=SIZE_COVERT, 
                            alpha=1.0, linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # Layer 4: 隐蔽交易 (红色三角)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color='#D62728', node_shape='^', 
                            node_size=SIZE_COVERT + 10, 
                            alpha=1.0, linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # 6. 图例与保存
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title(f"比特币混合交易图谱", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bitcoin_vis_final_optimized_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()


    def visualize_VGAE_covert(self, covert_tx_ids=None):
        """
        [最终定稿版] 
        方法名：visualize_VGAE_covert
        逻辑：边界约束 + 原地分层重排 (Bounded In-place Layered Layout)
        
        特性：
        1. 背景：保留全局力导向布局，展示宏观结构。
        2. 前景：解析隐蔽交易的轮次 (Input->R1->R2...)，将其在原地重排为分层结构。
        3. 约束：严格限制隐蔽节点坐标不超过背景图边界。
        4. 样式：严格保持 Size=10(背景)/25(隐蔽)，无额外文字标注。
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
        import numpy as np
        import re
        from collections import defaultdict
        from datetime import datetime

        if covert_tx_ids is None:
            covert_tx_ids = set()
        else:
            covert_tx_ids = set(covert_tx_ids)

        # 1. 基础设置
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(14, 14), dpi=200) 
        
        print(f"1. 正在计算全局布局 (节点数: {self.graph.number_of_nodes()})...")

        # ==========================
        # 2. 全局布局 (计算初始位置)
        # ==========================
        try:
            # 优先使用 Neato 获得较好的全局观
            pos = nx.nx_agraph.pygraphviz_layout(
                self.graph,
                prog="neato",
                # 稍微增大排斥力，为原地展开留一点缝隙
                args="-Grepulsiveforce=1.2 -Goverlap=False -Gmaxiter=1000"
            )
        except Exception:
            print("PyGraphviz layout failed, using spring_layout...")
            pos = nx.spring_layout(self.graph, k=0.03, iterations=50, seed=42)

        # ==========================
        # 3. 筛选与分类
        # ==========================
        # A. 找出所有隐蔽相关节点
        covert_related_addrs = set()
        for tx_id in covert_tx_ids:
            if tx_id in self.graph:
                neighbors = list(self.graph.successors(tx_id)) + list(self.graph.predecessors(tx_id))
                covert_related_addrs.update(neighbors)
        
        all_covert_nodes = list(covert_tx_ids.union(covert_related_addrs))
        covert_node_set = set(all_covert_nodes)
        
        # B. 节点分类 (用于绘图)
        bg_nodes = []
        covert_addrs_list = []
        covert_txs_list = []
        
        for n, attr in self.graph.nodes(data=True):
            if n in covert_tx_ids:
                covert_txs_list.append(n)
            elif n in covert_related_addrs:
                covert_addrs_list.append(n)
            else:
                bg_nodes.append(n)

        # ==========================
        # 4. 原地分层重排与边界约束
        # ==========================
        if len(all_covert_nodes) > 0 and len(bg_nodes) > 0:
            print("2. 执行原地分层重排与边界约束...")
            
            # A. 获取背景节点的物理边界 (Bounding Box)
            bg_coords = np.array([pos[n] for n in bg_nodes])
            bg_x_min, bg_x_max = np.min(bg_coords[:, 0]), np.max(bg_coords[:, 0])
            bg_y_min, bg_y_max = np.min(bg_coords[:, 1]), np.max(bg_coords[:, 1])
            
            bg_width = bg_x_max - bg_x_min
            bg_height = bg_y_max - bg_y_min
            
            # B. 计算当前隐蔽团簇的重心 (作为原地展开的锚点)
            curr_x = [pos[n][0] for n in all_covert_nodes]
            curr_y = [pos[n][1] for n in all_covert_nodes]
            center_x = sum(curr_x) / len(curr_x)
            center_y = sum(curr_y) / len(curr_y)
            
            # 确保锚点本身在边界内
            center_x = max(bg_x_min, min(bg_x_max, center_x))
            center_y = max(bg_y_min, min(bg_y_max, center_y))

            # --- 分层逻辑 ---
            layers = defaultdict(list)
            for node in all_covert_nodes:
                node_str = str(node)
                layer_idx = 0
                
                # 解析 Round 信息 (例如 _R1_ -> 第2层, Tx -> 第3层, _R2_ -> 第4层)
                match = re.search(r"_R(\d+)_", node_str)
                if match:
                    layer_idx = int(match.group(1)) * 2
                elif len(node_str) == 64: 
                    # 交易节点层级 = 前驱地址层级 + 1
                    preds = list(self.graph.predecessors(node))
                    max_pred_layer = 0
                    for p in preds:
                        m = re.search(r"_R(\d+)_", str(p))
                        if m: max_pred_layer = max(max_pred_layer, int(m.group(1)) * 2)
                    layer_idx = max_pred_layer + 1
                else:
                    layer_idx = 0 # 初始输入
                
                layers[layer_idx].append(node)
            
            # C. 设定间距 (相对于背景尺寸)
            # 这里的系数决定了隐蔽子图的"疏密程度"
            X_STEP = bg_width * 0.035  # 层间距 (左右)
            Y_STEP = bg_height * 0.025 # 节点间距 (上下)
            
            sorted_layers = sorted(layers.keys())
            num_layers = len(sorted_layers)
            
            # D. 计算新坐标并应用约束
            for i, l_idx in enumerate(sorted_layers):
                nodes = layers[l_idx]
                nodes.sort() # 保证同一层内顺序固定
                
                # X轴: 居中展开
                layer_offset_x = i - (num_layers / 2.0)
                this_x = center_x + (layer_offset_x * X_STEP)
                
                # Y轴: 居中展开
                layer_height = (len(nodes) - 1) * Y_STEP
                start_y = center_y - (layer_height / 2.0)
                
                for j, node in enumerate(nodes):
                    this_y = start_y + (j * Y_STEP)
                    
                    # 【核心约束】Clamp: 强制截断坐标至背景边界内
                    final_x = max(bg_x_min, min(bg_x_max, this_x))
                    final_y = max(bg_y_min, min(bg_y_max, this_y))
                    
                    pos[node] = (final_x, final_y)

        # ==========================
        # 5. 绘制 (样式严格保持不变)
        # ==========================
        print("3. 开始绘制...")
        ax = plt.gca()
        
        SIZE_BG = 10        # 背景节点大小
        SIZE_COVERT = 25    # 隐蔽节点大小
        
        # 背景节点
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color='#C0C0C0', node_size=SIZE_BG, 
                            alpha=0.6, linewidths=0, ax=ax)
        
        # 背景边
        bg_edges = [(u, v) for u, v in self.graph.edges() if u not in covert_node_set]
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color='#C0C0C0', arrows=False, 
                            width=0.5, alpha=0.4, ax=ax)

        # 隐蔽连线 (直线)
        covert_edges = [(u, v) for u, v in self.graph.edges() if u in covert_node_set and v in covert_node_set]
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color='#FF4500', 
                            node_size=SIZE_COVERT, 
                            arrowstyle='-|>', arrowsize=8, 
                            width=1.0, alpha=1.0, ax=ax)

        # 隐蔽地址 (蓝色圆点)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs_list, 
                            node_color='#1E90FF', node_shape='o', 
                            node_size=SIZE_COVERT, 
                            alpha=1.0, linewidths=0.5, edgecolors='white', 
                            ax=ax)

        # 隐蔽交易 (红色三角)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs_list, 
                            node_color='#D62728', node_shape='^', 
                            node_size=SIZE_COVERT, 
                            alpha=1.0, linewidths=0.5, edgecolors='black', 
                            ax=ax)

        # ==========================
        # 6. 保存为 PDF
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易 (Tx)'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址 (Addr)'),
            mpatches.Patch(color='#C0C0C0', label='背景交易'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.title(f"比特币混合交易图谱 (VGAE原地分层)", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"visualize_VGAE_covert_{timestamp}.pdf"
        plt.savefig(filename, format="pdf", bbox_inches='tight', dpi=300)
        print(f"绘图完成，已保存至: {filename}")
        plt.show()
        

    def visualize_chain_covert_1in2out(self, covert_tx_ids=None):
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
        SIZE_BG = 10        # 正常节点大小
        SIZE_COVERT = 25    # 隐蔽节点大小 (保持稍大以突出结构)
        
        # --- Layer 1: 背景 (严格参照您的输入样式) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=bg_nodes, 
                            node_color='#C0C0C0', 
                            node_size=SIZE_BG, 
                            alpha=0.6,            # 透明度提高
                            linewidths=0, ax=ax)
        
        nx.draw_networkx_edges(self.graph, pos, edgelist=bg_edges, 
                            edge_color='#C0C0C0', 
                            arrows=False, 
                            width=0.5,            # 线条加粗
                            alpha=0.4,            # 透明度提高
                            ax=ax)

        # --- Layer 2: 隐蔽链路 (Unified Solid Line) ---
        nx.draw_networkx_edges(self.graph, pos, edgelist=covert_edges, 
                            edge_color='#FF4500', 
                            node_size=SIZE_COVERT, 
                            arrowstyle='-|>', arrowsize=8, 
                            width=1.0, alpha=1.0, # 保持不透明，统一实线
                            connectionstyle="arc3,rad=0.05",
                            ax=ax)

        # --- Layer 3: 隐蔽地址 (Blue Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_addrs, 
                            node_color='#1E90FF', node_shape='o', node_size=SIZE_COVERT, 
                            edgecolors='white', linewidths=0.5, ax=ax)

        # --- Layer 4: 隐蔽交易 (Red Nodes) ---
        nx.draw_networkx_nodes(self.graph, pos, nodelist=covert_txs, 
                            node_color='#D62728', node_shape='^', node_size=SIZE_COVERT + 10, 
                            edgecolors='black', linewidths=0.5, ax=ax)

        # ==========================
        # 6. 图例与保存
        # ==========================
        legend_elements = [
            mpatches.Patch(color='#D62728', label='隐蔽交易 (Tx)'),
            mpatches.Patch(color='#1E90FF', label='隐蔽地址 (Addr)'),
            mpatches.Patch(color='#C0C0C0', label='正常交易图'),
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
        plt.title(f"比特币混合交易图谱 (Peeling Chain)", fontsize=14)
        plt.axis('off')

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bitcoin_vis_final_structure_{timestamp}.pdf"
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
