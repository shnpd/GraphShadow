
import os
import json
import glob
import numpy as np
import networkx as nx
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from experiment.addressandtransaction import constuct_graph
# ==========================================
# 1. 特征指纹 (Targeted Fingerprints)
# ==========================================
METHOD_FINGERPRINTS = {
    "GraphShadow": {
        "out_degree_variance": 0.5730,
        "in_degree_variance": 0.3724,
        "path_ratio": 0.6197 
    },
    "BlockWhisper": {
        "out_degree_variance": 0.4214,
        "in_degree_variance":  0.3963  ,
        "path_ratio":0.8387 
    },
    "DDSAC": {
        "out_degree_variance": 0.9965,
        "in_degree_variance": 0.0554,
        "path_ratio": 0.5294
    },
    "GBCTD": {
        "out_degree_variance":  1.3755  ,
        "in_degree_variance": 1.3755   ,
        "path_ratio":0.4809 
    }
}

# ==========================================
# 2. 检测器与辅助类 (保持精简)
# ==========================================
class TargetedTransactionDetector:
    def __init__(self, raw_graph_obj, thresholds, epsilon=0.1, min_edges=0):
        self.raw_graph = raw_graph_obj.graph
        self.min_edges = min_edges
        self.THRESHOLDS = thresholds
        self.EPSILON = {k: epsilon for k in ["out_degree", "in_degree", "path_ratio"]}

    def _build_address_interaction_graph(self):
        # ... (与之前相同，构建 DiGraph) ...
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
        # ... (与之前相同，计算3个指标) ...
        num_nodes = subgraph.number_of_nodes()
        if num_nodes == 0: return None
        out_degrees = [d for n, d in subgraph.out_degree()]
        var_out = np.var(out_degrees) if out_degrees else 0
        in_degrees = [d for n, d in subgraph.in_degree()]
        var_in = np.var(in_degrees) if in_degrees else 0
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
        return {"var_out_degree": var_out, "var_in_degree": var_in, "path_ratio": path_ratio}


def load_transactions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==========================================
# 3. 核心实验逻辑：定额重采样测试
# ==========================================
def run_fixed_quota_experiment():
    # --- 实验参数配置 ---
    
    epsilon_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    
    # [关键配置] 统一尺度
    FIXED_COVERT_QUOTA = 5000  
    FIXED_BG_QUOTA = 5000     
    
    methods_files = {
        "GraphShadow": "CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json",
        "BlockWhisper": "CompareMethod/BlockWhisper/dataset/BlockWhisper_transactions_*.json",
        "DDSAC": "CompareMethod/DDSAC/dataset/DDSAC_transactions_*.json",
        "GBCTD": "CompareMethod/GBCTD/dataset/GBCTD_transactions_*.json"
    }

    # 1. 预处理背景 (只做一次)
    bg_txs = []
    for i in tqdm(range(20)):
        bg_file = f"dataset/transactions_block_{923800 + i}.json"
        full_bg_txs = load_transactions_from_file(bg_file)
        bg_txs.extend(full_bg_txs)
    
    # 提取所有背景子图及其指标
    print(f"bg tx count:{len(bg_txs)}")
    bg_graph_obj = constuct_graph(bg_txs)
    # 使用临时阈值初始化以便计算 metrics
    temp_detector = TargetedTransactionDetector(bg_graph_obj, thresholds=METHOD_FINGERPRINTS["GBCTD"]) 
    bg_G = temp_detector._build_address_interaction_graph()
    bg_subgraphs = [bg_G.subgraph(c).copy() for c in nx.weakly_connected_components(bg_G) 
                    if bg_G.subgraph(c).number_of_edges() >= 3]
    
    bg_metrics_list = []
    for sub_g in bg_subgraphs:
        m = temp_detector._calculate_metrics(sub_g)
        if m: bg_metrics_list.append(m)
    
    print(f"[*] Background Pool Size: {len(bg_metrics_list)} components (will be scaled to {FIXED_BG_QUOTA})")

    all_results = []

    # 2. 循环每个方法
    for method_name, file_pattern in methods_files.items():
        target_fingerprint = METHOD_FINGERPRINTS.get(method_name)
        if not target_fingerprint: continue

        covert_files = glob.glob(file_pattern)
        try: covert_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
        except: pass
        target_files = covert_files
        if not target_files: continue

        print(f"\n>>> Processing Method: {method_name} <<<")
        
        # --- 步骤 A: 收集该方法所有的真实隐蔽子图指标 ---
        # 我们先把所有文件里的子图指标都算出来，存到一个大池子里
        covert_metrics_pool = []
        
        for cov_file in tqdm(target_files, desc="Collecting Subgraphs", leave=False):
            cov_txs = load_transactions_from_file(cov_file)
            cov_graph_obj = constuct_graph(cov_txs)
            # 阈值无所谓，这里只为了调calculate_metrics
            det = TargetedTransactionDetector(cov_graph_obj, thresholds=target_fingerprint, min_edges=0)
            cov_G = det._build_address_interaction_graph()
            real_comps = [c for c in nx.weakly_connected_components(cov_G) if cov_G.subgraph(c).number_of_edges() >= 1]
            
            for c in real_comps:
                sub_g = cov_G.subgraph(c)
                m = det._calculate_metrics(sub_g)
                if m: covert_metrics_pool.append(m)
        
        print(f"    -> Collected {len(covert_metrics_pool)} unique components.")
        
        # --- 步骤 B: 定额重采样 (Resampling) ---
        # 如果样本不够 1000，则随机重复抽取；如果超过，则随机截取
        if len(covert_metrics_pool) == 0:
            print("    [!] No components found for this method!")
            continue
            
        # 使用 random.choices 进行有放回采样 (Upsampling) 或 random.sample (Downsampling)
        # 这里统一用 choices (有放回) 以支持任意数量的扩展
        resampled_covert_metrics = random.choices(covert_metrics_pool, k=FIXED_COVERT_QUOTA)
        
        # 3. 循环 Epsilon 进行检测
        for eps in epsilon_values:
            
            # --- C. 计算 FP (背景误报) ---
            # 基于当前的 target_fingerprint 和 eps，计算背景池的误报率
            fp_raw_count = 0
            for m in bg_metrics_list:
                dist_out = abs(target_fingerprint["out_degree_variance"] - m["var_out_degree"])
                is_covert_out = dist_out <= eps
                dist_in = abs(target_fingerprint["in_degree_variance"] - m["var_in_degree"])
                is_covert_in = dist_in <= eps
                dist_path = abs(target_fingerprint["path_ratio"] - m["path_ratio"])
                is_covert_path = dist_path <= eps
                
                reasons = [x for x in [is_covert_out, is_covert_in, is_covert_path] if x]
                if len(reasons) >= 2 or is_covert_out:
                    fp_raw_count += 1
            
            # 计算背景误报率
            bg_fp_rate = fp_raw_count / len(bg_metrics_list) if len(bg_metrics_list) > 0 else 0
            
            # 扩展到统一的背景规模 (FIXED_BG_QUOTA)
            # 例如：如果背景误报率是 30%，我们在 50000 个背景中就会有 15000 个 FP
            final_fp = bg_fp_rate * FIXED_BG_QUOTA

            # --- D. 计算 TP (隐蔽检出) ---
            # 在重采样后的 1000 个隐蔽子图中检测
            tp_count = 0
            for m in resampled_covert_metrics:
                dist_out = abs(target_fingerprint["out_degree_variance"] - m["var_out_degree"])
                is_covert_out = dist_out <= eps
                dist_in = abs(target_fingerprint["in_degree_variance"] - m["var_in_degree"])
                is_covert_in = dist_in <= eps
                dist_path = abs(target_fingerprint["path_ratio"] - m["path_ratio"])
                is_covert_path = dist_path <= eps
                
                reasons = [x for x in [is_covert_out, is_covert_in, is_covert_path] if x]
                if len(reasons) >= 2 or is_covert_out:
                    tp_count += 1
            
            final_tp = tp_count
            final_fn = FIXED_COVERT_QUOTA - final_tp

            # --- E. 计算最终指标 ---
            recall = final_tp / FIXED_COVERT_QUOTA
            precision = final_tp / (final_tp + final_fp) if (final_tp + final_fp) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            all_results.append({
                "Method": method_name,
                "Epsilon": eps,
                "Total_Real": FIXED_COVERT_QUOTA, # 统一为 1000
                "TP": final_tp,
                "FP (Scaled)": int(final_fp),     # 统一背景下的 FP
                "FN": final_fn,
                "Recall": recall,
                "Precision": precision,
                "F1_Score": f1,
                "FP_Rate": bg_fp_rate
            })

    # 打印到控制台 (保持不变，用于快速查看)
    print("\n" + "="*125)
    print(f"{'FIXED-QUOTA NORMALIZED RESULTS (Cov={FIXED_COVERT_QUOTA}, Bg={FIXED_BG_QUOTA})':^125}")
    print("="*125)
    header = "{:<12} | {:<6} | {:<8} | {:<8} | {:<10} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8}"
    print(header.format("Method", "Eps", "Total", "TP", "FP(Std)", "FN", "Recall", "Precis.", "F1", "FP Rate"))
    print("-" * 125)
    
    # 简单打印
    for res in all_results:
        print(header.format(
            res['Method'], str(res['Epsilon']), str(res['Total_Real']),
            str(res['TP']), str(res['FP (Scaled)']), str(res['FN']),
            f"{res['Recall']:.4f}", f"{res['Precision']:.4f}",
            f"{res['F1_Score']:.4f}", f"{res['FP_Rate']:.2%}"
        ))
    
    # --- 调用新方法保存 Excel 和 PDF ---
    save_experiment_visualization(all_results)


def save_experiment_visualization(all_results, output_dir='.'):
    """
    将实验结果保存为Excel表格和PDF可视化图表。
    
    参数:
        all_results (list): 包含实验结果字典的列表。
        output_dir (str): 输出文件的保存目录。
    """
    # 1. 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ---------------------------
    # 2. 保存为 Excel 表格
    # ---------------------------
    excel_path = os.path.join(output_dir, 'fixed_quota_results.xlsx')
    try:
        # 使用 ExcelWriter 可以更精细地控制格式（如果需要）
        # 这里直接保存，Pandas 默认格式通常已经很清晰
        df.to_excel(excel_path, index=False, sheet_name='Experiment Results')
        print(f"[*] Excel table saved to: {excel_path}")
    except ImportError:
        # 如果没有安装 openpyxl，回退到 csv
        csv_path = os.path.join(output_dir, 'fixed_quota_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"[!] openpyxl not found. Saved as CSV to: {csv_path}")

    # ---------------------------
    # 3. 绘制并保存 PDF 图表
    # ---------------------------
    pdf_path = os.path.join(output_dir, 'fixed_quota_plots.pdf')
    
    # 设置绘图风格
    plt.style.use('seaborn-v0_8-whitegrid') # 或者使用 'ggplot'
    
    # 定义要绘制的指标
    metrics = [
        ('Recall', 'Recall'),
        ('Precision', 'Precision'),
        ('F1_Score', 'F1-Score'),
        ('FP_Rate', 'Background FP Rate')
    ]
    
    methods = df['Method'].unique()
    
    # 颜色映射，确保不同方法颜色一致
    colors = plt.cm.tab10(range(len(methods)))
    method_color_map = dict(zip(methods, colors))
    
    with PdfPages(pdf_path) as pdf:
        for metric_col, metric_name in metrics:
            plt.figure(figsize=(10, 6))
            
            for method in methods:
                # 提取特定方法的数据并按 Epsilon 排序
                subset = df[df['Method'] == method].sort_values('Epsilon')
                
                plt.plot(subset['Epsilon'], subset[metric_col], 
                         marker='o', markersize=5, linewidth=2,
                         label=method, color=method_color_map[method])
            
            # 图表装饰
            plt.title(f'{metric_name} vs Epsilon (Fixed Quota)', fontsize=14)
            plt.xlabel('Epsilon (Log Scale)', fontsize=12)
            plt.ylabel(metric_name, fontsize=12)
            plt.xscale('log') # 使用对数坐标处理宽范围的 Epsilon
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.legend(title='Method', loc='best')
            plt.tight_layout()
            
            # 保存当前页面到 PDF
            pdf.savefig()
            plt.close()
            
    print(f"[*] PDF plots saved to: {pdf_path}")
    
if __name__ == "__main__":
    run_fixed_quota_experiment()