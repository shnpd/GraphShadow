import numpy as np
import networkx as nx
import glob
import os
import json
from tqdm import tqdm
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
class CovertFeatureAnalyzer:
    def __init__(self):
        # 用于存储各方法的特征数据
        self.stats = {
            "GraphShadow": {"out_var": [], "in_var": [], "path_ratio": []},
            "BlockWhisper": {"out_var": [], "in_var": [], "path_ratio": []},
            "DDSAC": {"out_var": [], "in_var": [], "path_ratio": []},
            "GBCTD": {"out_var": [], "in_var": [], "path_ratio": []}
        }

    def _build_graph(self, tx_list):
        """构建地址交互图 (修复版: 兼容 input_addrs/inputs)"""
        G = nx.DiGraph()
        for tx in tx_list:
            # --- 关键修改：兼容多种键名 ---
            # 优先尝试你数据中的 'input_addrs'，如果没有则尝试 'inputs'
            raw_inputs = tx.get('input_addrs') or tx.get('inputs') or []
            raw_outputs = tx.get('output_addrs') or tx.get('outputs') or []

            # 提取输入地址
            inputs = []
            for inp in raw_inputs:
                if isinstance(inp, str):
                    inputs.append(inp)
                elif isinstance(inp, dict):
                    # 处理嵌套结构 {'addresses': ['...']}
                    addrs = inp.get('addresses', [])
                    if addrs: inputs.extend(addrs)
            
            # 提取输出地址
            outputs = []
            for out in raw_outputs:
                if isinstance(out, str):
                    outputs.append(out)
                elif isinstance(out, dict):
                    addrs = out.get('addresses', [])
                    if addrs: outputs.extend(addrs)
            
            # 构建边
            for i_addr in inputs:
                for o_addr in outputs:
                    if i_addr != o_addr:
                        G.add_edge(i_addr, o_addr)
        return G

    def _calculate_metrics(self, subgraph):
        """计算原文定义的三个核心指标"""
        num_nodes = subgraph.number_of_nodes()
        if num_nodes == 0: return None

        # 1. 出度方差
        out_degrees = [d for n, d in subgraph.out_degree()]
        var_out = np.var(out_degrees) if out_degrees else 0
        
        # 2. 入度方差
        in_degrees = [d for n, d in subgraph.in_degree()]
        var_in = np.var(in_degrees) if in_degrees else 0

        # 3. 路径比 (Longest Path / Num Nodes)
        longest_path = 0
        try:
            if num_nodes > 0:
                # 针对有向无环图(DAG)优化计算
                if nx.is_directed_acyclic_graph(subgraph):
                    longest_path = len(nx.dag_longest_path(subgraph))
                else:
                    # 简化计算：全对最短路径的最大值
                    # 注意：对于大图这可能较慢，但对隐蔽子图通常可行
                    all_shortest_paths = dict(nx.all_pairs_shortest_path_length(subgraph))
                    for source, targets in all_shortest_paths.items():
                        if targets:
                            longest_path = max(longest_path, max(targets.values()))
        except:
            pass
        
        path_ratio = longest_path / num_nodes if num_nodes > 0 else 0
        
        return var_out, var_in, path_ratio

    def analyze_method(self, method_name, file_pattern, sample_limit):
        files = glob.glob(file_pattern)
        try:
            files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except: pass
        
        target_files = files[:sample_limit]
        print(f"[*] Analyzing {method_name} ({len(target_files)} samples)...")

        valid_samples = 0
        for f_path in tqdm(target_files, desc=f"Processing {method_name}", leave=False):
            try:
                with open(f_path, 'r', encoding='utf-8') as f:
                    tx_list = json.load(f)
            except: continue

            G = self._build_graph(tx_list)
            
            # 这里的 min_edges 必须设为 0，否则 DDSAC 会被过滤导致 NO DATA
            # 只要有边 (>=1) 我们就统计它的特征
            subgraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
            
            for sub_g in subgraphs:
                if sub_g.number_of_edges() < 1: continue
                
                var_out, var_in, ratio = self._calculate_metrics(sub_g)
                
                self.stats[method_name]["out_var"].append(var_out)
                self.stats[method_name]["in_var"].append(var_in)
                self.stats[method_name]["path_ratio"].append(ratio)
                valid_samples += 1
        
        if valid_samples == 0:
            print(f"[!] Warning: No valid subgraphs found for {method_name}. Check JSON keys!")

    def print_recommendations(self):
        print("\n" + "="*100)
        print(f"{'METHOD':<15} | {'METRIC':<12} | {'MEAN (New Th)':<15} | {'STD':<10} | {'Original Th':<12}")
        print("-" * 100)
        
        ref_th = {"out_var": 1.0, "in_var": 0.01, "path_ratio": 0.5}

        for method, metrics in self.stats.items():
            for m_key in ["out_var", "in_var", "path_ratio"]:
                data = metrics[m_key]
                if not data:
                    print(f"{method:<15} | {m_key:<12} | {'NO DATA':<15} | {'-':<10} | {ref_th[m_key]:<12}")
                    continue
                
                mean_val = np.mean(data)
                std_val = np.std(data)
                
                print(f"{method:<15} | {m_key:<12} | {mean_val:<15.4f} | {std_val:<10.4f} | {ref_th[m_key]:<12}")
            print("-" * 100)

if __name__ == "__main__":
    analyzer = CovertFeatureAnalyzer()
    
    # 路径配置 (请确认路径是否正确)
    base_dir = "CompareMethod"
    configs = [
        ("GraphShadow", f"{base_dir}/GraphShadow/dataset/GraphShadow_transactions_*.json"),
        ("BlockWhisper", f"{base_dir}/BlockWhisper/dataset/BlockWhisper_transactions_*.json"),
        ("DDSAC", f"{base_dir}/DDSAC/dataset/DDSAC_transactions_*.json"),
        ("GBCTD", f"{base_dir}/GBCTD/dataset/GBCTD_transactions_*.json")
    ]
    
    for name, pattern in configs:
        if glob.glob(pattern):
            analyzer.analyze_method(name, pattern, sample_limit=200)
        else:
            print(f"[!] Warning: File pattern not found: {pattern}")

    analyzer.print_recommendations()



