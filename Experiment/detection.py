import networkx as nx
import numpy as np
from txgraph.main import BitcoinTransactionGraph
from graphanalysis.sample_transaction import (
    load_transactions_from_file,
    load_graph_cache,
)
from addressandtransaction import constuct_graph, merge_graphs
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from collections import deque
from tqdm import tqdm  # 用于显示进度条，如果没有请 pip install tqdm
import json
import glob
import itertools


# ==========================================
# 1. 注入特征指纹 (基于你提供的数据)
# ==========================================
METHOD_FINGERPRINTS = {
    "GraphShadow": {
        "out_degree_variance": 0.4683,
        "in_degree_variance": 0.2562,
        "path_ratio": 0.6402
    },
    "BlockWhisper": {
        "out_degree_variance": 0.3104,
        "in_degree_variance": 0.3026,
        "path_ratio": 0.9362
    },
    "DDSAC": {
        "out_degree_variance": 0.9956,
        "in_degree_variance": 0.0622,
        "path_ratio": 0.5333
    },
    "GBCTD": {
        "out_degree_variance": 27.8462,
        "in_degree_variance": 28.4068 ,
        "path_ratio": 0.1112
    }
}


class TargetedTransactionDetector:
    def __init__(self, raw_graph_obj, thresholds, epsilon=0.1, min_edges=0):
        """
        :param thresholds: 针对特定方法定制的阈值字典
        :param epsilon: 检测容忍度
        """
        self.raw_graph = raw_graph_obj.graph
        self.min_edges = min_edges
        
        # 使用传入的针对性阈值
        self.THRESHOLDS = thresholds

        # 动态设置 Epsilon
        self.EPSILON = {
            "out_degree": epsilon,
            "in_degree": epsilon,
            "path_ratio": epsilon,
        }

    def _build_address_interaction_graph(self):
        addr_graph = nx.DiGraph()
        tx_nodes = [n for n, d in self.raw_graph.nodes(data=True) if d.get("node_type") == "transaction"]
        for tx in tx_nodes:
            inputs = [u for u, v, d in self.raw_graph.in_edges(tx, data=True)]
            outputs = [v for u, v, d in self.raw_graph.out_edges(tx, data=True)]
            for i_addr in inputs:
                for o_addr in outputs:
                    if i_addr != o_addr:
                        addr_graph.add_edge(i_addr, o_addr)
        return addr_graph

    def _calculate_metrics(self, subgraph):
        num_nodes = subgraph.number_of_nodes()
        if num_nodes == 0: return None
        
        out_degrees = [d for n, d in subgraph.out_degree()]
        var_out_degree = np.var(out_degrees) if out_degrees else 0
        
        in_degrees = [d for n, d in subgraph.in_degree()]
        var_in_degree = np.var(in_degrees) if in_degrees else 0
        
        longest_path = 0
        try:
            if num_nodes > 0:
                if nx.is_directed_acyclic_graph(subgraph):
                    longest_path = len(nx.dag_longest_path(subgraph))
                else:
                    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
                    for source, targets in all_shortest_paths.items():
                        if targets:
                            longest_path = max(longest_path, max(targets.values()))
        except: pass

        path_ratio = longest_path / num_nodes if num_nodes > 0 else 0
        return {
            "var_out_degree": var_out_degree,
            "var_in_degree": var_in_degree,
            "path_ratio": path_ratio,
        }

    def detect_detailed(self):
        G = self._build_address_interaction_graph()
        subgraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
        detected_subgraphs_nodes = []

        for sub_g in subgraphs:
            if sub_g.number_of_edges() < self.min_edges: continue

            metrics = self._calculate_metrics(sub_g)
            if not metrics: continue

            # 判定逻辑：差异 <= Epsilon (因为阈值就是该方法的特征中心)
            dist_out = abs(self.THRESHOLDS["out_degree_variance"] - metrics["var_out_degree"])
            is_covert_out = dist_out <= self.EPSILON["out_degree"]

            dist_in = abs(self.THRESHOLDS["in_degree_variance"] - metrics["var_in_degree"])
            is_covert_in = dist_in <= self.EPSILON["in_degree"]

            dist_path = abs(self.THRESHOLDS["path_ratio"] - metrics["path_ratio"])
            is_covert_path = dist_path <= self.EPSILON["path_ratio"]

            # 判定条件：满足任意两个特征 或 满足出度特征
            reasons = [x for x in [is_covert_out, is_covert_in, is_covert_path] if x]
            if len(reasons) >= 2 or is_covert_out:
                detected_subgraphs_nodes.append(set(sub_g.nodes()))
        
        return detected_subgraphs_nodes

        
def generate_sliding_window_dataset(
    source_dir='dataset',
    output_dir='experiment/detectiondataset/background_dataset',
    start_height=923800,
    end_height=924300,
    window_size=9
):
    """
    使用滑动窗口生成聚合的区块交易数据集（扁平化合并）。
    窗口策略: [H, H + window_size - 1]
    输出格式: 一个包含该窗口内所有交易的长列表 [tx1, tx2, ..., txN]
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 用于缓存当前窗口的区块数据
    window_buffer = deque()

    print(f"Starting processing from {start_height} to {end_height} (Window: {window_size})...")

    # 1. 预填充前 (window_size - 1) 个区块
    for i in range(window_size - 1):
        current_h = start_height + i
        file_path = os.path.join(source_dir, f"transactions_block_{current_h}.json")
        
        entry = {'height': current_h, 'data': []} # 默认为空列表，防止 None 报错
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        entry['data'] = data
                    else:
                        print(f"Warning: Block {current_h} format incorrect (expected list).")
            except Exception as e:
                print(f"Error reading block {current_h}: {e}")
        else:
            print(f"Warning: Block {current_h} not found.")
            
        window_buffer.append(entry)

    # 2. 主循环：滑动并保存
    loop_start = start_height + window_size - 1
    
    for current_h in tqdm(range(loop_start, end_height + 1), desc="Processing Windows"):
        
        # --- A. 读取新的一块 (右边滑入) ---
        file_path = os.path.join(source_dir, f"transactions_block_{current_h}.json")
        new_block_entry = {'height': current_h, 'data': []}
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        new_block_entry['data'] = data
            except Exception as e:
                print(f"Error reading block {current_h}: {e}")
        
        window_buffer.append(new_block_entry)

        # --- B. 扁平化合并并保存 (核心修改) ---
        
        window_start_h = window_buffer[0]['height']
        window_end_h = current_h
        
        # 检查数据完整性：如果所有块的数据都不是空的（根据你的需求，也可以允许部分为空）
        # 这里假设只要文件存在就要合并
        
        # *** 关键修改开始 ***
        # 将 deque 中存储的 9 个列表合并成 1 个大列表
        flat_transactions = []
        for block_entry in window_buffer:
            # block_entry['data'] 本身是一个列表 [tx, tx, ...]
            # 使用 extend 将其元素追加到 flat_transactions，而不是 append 列表本身
            flat_transactions.extend(block_entry['data'])
        # *** 关键修改结束 ***

        # 如果合并后的列表不为空，则保存
        if flat_transactions:
            output_filename = f"transactions_block_{window_start_h}_{window_end_h}.json"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(flat_transactions, f_out) # 不使用 indent 以节省空间，如需可加 indent=2
        
        # --- C. 移除最旧的一块 (左边滑出) ---
        window_buffer.popleft()

    print("Processing complete.")


# ==========================================
# 4. 核心实验逻辑：针对性检测 (详细输出版)
# ==========================================
def run_targeted_detection_detailed():
    # 1. 配置
    bg_file = "experiment/detectiondataset/background_dataset/transactions_block_923800_923808.json"
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    
    methods_files = {
        "GraphShadow": "CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json",
        "BlockWhisper": "CompareMethod/BlockWhisper/dataset/BlockWhisper_transactions_*.json",
        "DDSAC": "CompareMethod/DDSAC/dataset/DDSAC_transactions_*.json",
        "GBCTD": "CompareMethod/GBCTD/dataset/GBCTD_transactions_*.json"
    }

    print(f"[*] Loading Background: {os.path.basename(bg_file)}")
    full_bg_txs = load_transactions_from_file(bg_file)
    bg_txs = full_bg_txs[:2500]
    
    # --- 预处理背景 ---
    print("[*] Pre-processing background graph...")
    bg_graph_obj = constuct_graph(bg_txs)
    temp_detector = TargetedTransactionDetector(bg_graph_obj, thresholds=METHOD_FINGERPRINTS["DDSAC"]) 
    bg_G = temp_detector._build_address_interaction_graph()
    bg_subgraphs = [bg_G.subgraph(c).copy() for c in nx.weakly_connected_components(bg_G) 
                    if bg_G.subgraph(c).number_of_edges() >= 1]
    
    bg_metrics_list = []
    for sub_g in bg_subgraphs:
        m = temp_detector._calculate_metrics(sub_g)
        if m: bg_metrics_list.append(m)
    
    total_bg_components = len(bg_metrics_list)
    print(f"[*] Background contains {total_bg_components} valid components (Bg.Comp).")

    all_results = []

    # 2. 循环方法
    for method_name, file_pattern in methods_files.items():
        target_fingerprint = METHOD_FINGERPRINTS.get(method_name)
        if not target_fingerprint: continue

        covert_files = glob.glob(file_pattern)
        try: covert_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except: pass
        target_files = covert_files[:100]
        if not target_files: continue

        print(f"\n>>> Targeting Method: {method_name} <<<")

        # 3. 循环 Epsilon
        for eps in epsilon_values:
            
            # --- A. 计算 FP (在背景中误报的数量) ---
            fp_count_single_pass = 0
            for m in bg_metrics_list:
                dist_out = abs(target_fingerprint["out_degree_variance"] - m["var_out_degree"])
                is_covert_out = dist_out <= eps
                dist_in = abs(target_fingerprint["in_degree_variance"] - m["var_in_degree"])
                is_covert_in = dist_in <= eps
                dist_path = abs(target_fingerprint["path_ratio"] - m["path_ratio"])
                is_covert_path = dist_path <= eps
                
                reasons = [x for x in [is_covert_out, is_covert_in, is_covert_path] if x]
                if len(reasons) >= 2 or is_covert_out:
                    fp_count_single_pass += 1
            
            # 缩放 FP: 因为我们即将跑 len(target_files) 次实验，
            # 累计的 FP 应该是 单次背景FP * 实验次数
            total_fp_scaled = fp_count_single_pass * len(target_files)
            # 累计的背景分量总数
            total_bg_scaled = total_bg_components * len(target_files)

            fpr = fp_count_single_pass / total_bg_components if total_bg_components > 0 else 0

            # --- B. 计算 TP (在隐蔽样本中抓到的数量) ---
            sum_tp = 0
            sum_total_real = 0

            for cov_file in target_files:
                cov_txs = load_transactions_from_file(cov_file)
                cov_graph_obj = constuct_graph(cov_txs)
                det = TargetedTransactionDetector(cov_graph_obj, thresholds=target_fingerprint, epsilon=eps, min_edges=0)
                cov_G = det._build_address_interaction_graph()
                real_comps = [c for c in nx.weakly_connected_components(cov_G) if cov_G.subgraph(c).number_of_edges() >= 1]
                
                sum_total_real += len(real_comps)

                detected_nodes_list = det.detect_detailed()

                for real_comp in real_comps:
                    real_set = set(real_comp)
                    is_hit = False
                    for det_set in detected_nodes_list:
                        if not real_set.isdisjoint(det_set):
                            is_hit = True
                            break
                    if is_hit:
                        sum_tp += 1
            
            # --- C. 汇总指标 ---
            sum_fn = sum_total_real - sum_tp

            recall = sum_tp / sum_total_real if sum_total_real > 0 else 0
            precision = sum_tp / (sum_tp + total_fp_scaled) if (sum_tp + total_fp_scaled) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            all_results.append({
                "Method": method_name,
                "Epsilon": eps,
                "Bg_Comp": total_bg_scaled,  # 累计背景分量
                "Cov_Comp": sum_total_real,  # 累计隐蔽分量
                "TP": sum_tp,
                "FP": total_fp_scaled,       # 累计误报
                "FN": sum_fn,
                "Recall": recall,
                "Precision": precision,
                "F1_Score": f1,
                "FP_Rate_on_Bg": fpr
            })

    # ==========================================
    # 5. 输出详细表格
    # ==========================================
    print("\n" + "="*145)
    print(f"{'TARGETED ATTACK SIMULATION RESULTS (Detailed)':^145}")
    print("="*145)
    
    header = "{:<12} | {:<6} | {:<9} | {:<9} | {:<6} | {:<8} | {:<6} | {:<8} | {:<8} | {:<8} | {:<10}"
    print(header.format("Method", "Eps", "Bg.Comp", "Cov.Comp", "TP", "FP", "FN", "Recall", "Precis.", "F1", "FP Rate"))
    print("-" * 145)
    
    df = pd.DataFrame(all_results)
    
    for method in methods_files.keys():
        subset = df[df["Method"] == method]
        for _, row in subset.iterrows():
            print(header.format(
                row['Method'],
                str(row['Epsilon']),
                str(row['Bg_Comp']),
                str(row['Cov_Comp']),
                str(row['TP']),
                str(row['FP']),
                str(row['FN']),
                f"{row['Recall']:.4f}",
                f"{row['Precision']:.4f}",
                f"{row['F1_Score']:.4f}",
                f"{row['FP_Rate_on_Bg']:.2%}"
            ))
        print("-" * 145)
    
    # 保存结果
    df.to_csv("targeted_attack_results_detailed.csv", index=False)
    print("[*] Results saved to targeted_attack_results_detailed.csv")

if __name__ == "__main__":
    run_targeted_detection_detailed()

# if __name__ == "__main__":
#     # generate_sliding_window_dataset(
#     #     source_dir='dataset',               # 你的源文件目录
#     #     output_dir='experiment/detectiondataset/background_dataset',     # 结果保存目录
#     #     start_height=923900,
#     #     end_height=924000,
#     #     window_size=9
#     # )
#     # run_comprehensive_stealthiness_test("DDSAC")
#     run_batch_block_stress_test()