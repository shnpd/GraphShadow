import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.ticker as ticker
import seaborn as sns # 可选：如果安装了seaborn，配色会更好看，没有也没关系

def analyze_and_plot_refined():
    # 1. 定义文件列表
    files = [
        "graphanalysis/diameter150/dist_stats_start_923800.jsonl",
        "graphanalysis/diameter150/dist_stats_start_923950.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924100.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924250.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924400.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924550.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924700.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924850.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925000.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925150.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925300.jsonl",
    ]

    data = []

    # 2. 读取文件并提取数据
    for file in files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        dist_list = record.get('dist_list', [])
                        if dist_list:
                            dist_array = np.array(dist_list)
                            data.append({
                                'window_size': record.get('window_size'),
                                'Max': np.max(dist_array),
                                'P99': np.percentile(dist_array, 99),
                                'P95': np.percentile(dist_array, 95),
                                'P90': np.percentile(dist_array, 90)
                            })
                    except:
                        continue
        except FileNotFoundError:
            continue

    df = pd.DataFrame(data)

   # 3. 聚合分析
    grouped = df.groupby('window_size').agg(['mean', 'std'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped['window_size'] = grouped['window_size'].astype(int)
    grouped = grouped.sort_values('window_size')

    # --- 绘图通用设置 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # [核心修改] 定义自定义刻度：1, 5, 10, 15...
    max_window = grouped['window_size'].max()
    # 这里的逻辑是：先取1，然后取从5开始步长为5的所有整数
    custom_ticks = [1] + list(range(5, int(max_window) + 1, 5))
    
    # 筛选出这些特定窗口的数据
    subset = grouped[grouped['window_size'].isin(custom_ticks)].copy()
    
    # ==========================================
    # 图 1: 趋势分析 (Trend Analysis)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(14, 7))

    x = grouped['window_size']
    
    # 绘制完整曲线以保持平滑度
    ax1.plot(x, grouped['Max_mean'], label='Max', color='#e74c3c', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.plot(x, grouped['P99_mean'], label='P99', color='#f39c12', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.plot(x, grouped['P95_mean'], label='P95', color='#2980b9', linestyle='-', linewidth=1.5, alpha=0.9)
    ax1.plot(x, grouped['P90_mean'], label='P90', color='#27ae60', linestyle='-', linewidth=1.5, alpha=0.9)

    # [优化] 只在自定义刻度处 (1, 5, 10...) 添加标记
    ax1.scatter(subset['window_size'], subset['Max_mean'], color='#e74c3c', marker='o', s=40, zorder=5)
    ax1.scatter(subset['window_size'], subset['P99_mean'], color='#f39c12', marker='s', s=40, zorder=5)
    ax1.scatter(subset['window_size'], subset['P95_mean'], color='#2980b9', marker='^', s=50, zorder=5)
    ax1.scatter(subset['window_size'], subset['P90_mean'], color='#27ae60', marker='D', s=40, zorder=5)

    ax1.set_title('Mean Diameter per Window Size', fontsize=16, pad=15)
    ax1.set_xlabel('Window Size', fontsize=14)
    ax1.set_ylabel('Diameter Value (Mean)', fontsize=14)
    ax1.legend(loc='upper left', frameon=True, fontsize=12, shadow=True, facecolor='white', framealpha=1)
    
    # [核心修改] 设置 X 轴刻度
    ax1.set_xticks(custom_ticks)
    ax1.set_xlim(left=0, right=max_window + 2)
    ax1.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

  # ==========================================
    # 图 2: 波动幅度 (Stability Analysis) - 紧凑排列
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(15, 7))
    custom_ticks_values = [1] + list(range(5, int(max_window) + 1, 5))
    x_indices = np.arange(len(custom_ticks_values))
    avg_max = grouped['Max_std'].mean()
    avg_p99 = grouped['P99_std'].mean()
    avg_p95 = grouped['P95_std'].mean()
    avg_p90 = grouped['P90_std'].mean()

    # 3. 设置柱宽
    # 索引间距为 1。如果一组有 4 个柱子，bar_width = 0.2 意味着总宽 0.8，留白 0.2。
    # 想要间距更小？增大 bar_width (例如 0.22)
    # 想要间距更大？减小 bar_width (例如 0.15)
    bar_width = 0.2

    # 绘制柱状图 (使用 x_indices 而不是真实的 window_size)
    ax2.bar(x_indices - 1.5*bar_width, subset['Max_std'], width=bar_width, 
            label=f'Max Deviation (Avg: {avg_max:.2f})', color='#e74c3c', alpha=0.6, edgecolor='white')
    ax2.bar(x_indices - 0.5*bar_width, subset['P99_std'], width=bar_width, 
            label=f'P99 Deviation (Avg: {avg_p99:.2f})', color='#f39c12', alpha=0.7, edgecolor='white')
    ax2.bar(x_indices + 0.5*bar_width, subset['P95_std'], width=bar_width, 
            label=f'P95 Deviation (Avg: {avg_p95:.2f})', color='#2980b9', alpha=0.9, edgecolor='white')
    ax2.bar(x_indices + 1.5*bar_width, subset['P90_std'], width=bar_width, 
            label=f'P90 Deviation (Avg: {avg_p90:.2f})', color='#27ae60', alpha=1.0, edgecolor='white')

    ax2.set_title('Stability Analysis: Standard Deviation', fontsize=16, pad=15)
    ax2.set_xlabel('Window Size', fontsize=14)
    ax2.set_ylabel('Standard Deviation', fontsize=14)
    
    ax2.legend(loc='upper left', fontsize=12, frameon=True, facecolor='white', framealpha=1, edgecolor='gray', fancybox=True, shadow=True)
    
    # [关键步骤] 替换 X 轴刻度标签
    # 在索引位置显示真实的数值
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(custom_ticks_values)
    
    # 调整视野范围，留一点边距
    ax2.set_xlim(left=-0.6, right=len(x_indices) - 0.4)
    
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def generate_mapping_table():
    """
    基于 P95 平均值生成直径与窗口的映射表
    """
    # 1. 定义文件列表
    files = [
        "graphanalysis/diameter150/dist_stats_start_923800.jsonl",
        "graphanalysis/diameter150/dist_stats_start_923950.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924100.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924250.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924400.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924550.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924700.jsonl",
        "graphanalysis/diameter150/dist_stats_start_924850.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925000.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925150.jsonl",
        "graphanalysis/diameter150/dist_stats_start_925300.jsonl",
    ]

    data = []

    # 2. 读取文件并提取数据
    for file in files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        dist_list = record.get('dist_list', [])
                        if dist_list:
                            dist_array = np.array(dist_list)
                            data.append({
                                'window_size': record.get('window_size'),
                                'Max': np.max(dist_array),
                                'P99': np.percentile(dist_array, 99),
                                'P95': np.percentile(dist_array, 95),
                                'P90': np.percentile(dist_array, 90)
                            })
                    except:
                        continue
        except FileNotFoundError:
            continue

    df = pd.DataFrame(data)

   # 3. 聚合分析
    grouped = df.groupby('window_size').agg(['mean', 'std'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped['window_size'] = grouped['window_size'].astype(int)
    grouped = grouped.sort_values('window_size')
    
    
    mapping_data = []
    
    # 1. 确保按窗口大小升序排列
    sorted_df = grouped.sort_values('window_size').copy()

    mapping_data = []
    prev_threshold_int = 0  # 记录上一个写入表格的区间右端点（整数）
    last_max_p95 = 0.0      # 用于维护单调递增的物理阈值

    for index, row in sorted_df.iterrows():
        current_p95 = row['P95_mean']
        window = int(row['window_size'])
        
        # 2. 核心逻辑：确保单调递增
        # 只有当前阈值 > 之前出现过的最大阈值时，才生成新区间
        if current_p95 > last_max_p95:
            current_threshold_int = int(current_p95) # 取整，如 5.29 -> 5
            
            # 避免取整后出现 [2, 2] 这种无效区间（当两个阈值取整后相等时跳过）
            if current_threshold_int > prev_threshold_int:
                mapping_data.append({
                    "直径区间 (D)": f"({prev_threshold_int}, {current_threshold_int}]",
                    "原始P95均值": round(current_p95, 2),
                    "推荐窗口大小 (W)": window,
                    "判定逻辑": f"{prev_threshold_int} < D <= {current_threshold_int}"
                })
                # 更新已记录的整数边界和物理最大值
                prev_threshold_int = current_threshold_int
                last_max_p95 = current_p95
        else:
            # 如果窗口17的阈值(47)小于窗口16(48)，代码会进入这里直接跳过窗口17
            print(f"⚠️ 窗口 {window} 的阈值 {current_p95:.2f} 未超过前序最大值 {last_max_p95:.2f}，已跳过以保持单调性。")

    # 3. 生成最终 DataFrame
    df_final = pd.DataFrame(mapping_data)

    # 4. 添加溢出区间
    max_val = prev_threshold_int
    df_final = pd.concat([df_final, pd.DataFrame([{
        "直径区间 (D)": f"> {max_val}",
        "原始P95均值": "Overflow",
        "推荐窗口大小 (W)": "需人工介入",
        "判定逻辑": f"D > {max_val}"
    }])], ignore_index=True)

    # 5. 保存至 Excel
    filename = "直径窗口映射表_精简版.xlsx"
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df_final.to_excel(writer, index=False, sheet_name='Threshold_Mapping')
            worksheet = writer.sheets['Threshold_Mapping']
            for i, col in enumerate(df_final.columns):
                column_len = max(df_final[col].astype(str).map(len).max(), len(col)) + 5
                worksheet.column_dimensions[chr(65 + i)].width = column_len
        print(f"✅ 映射表已成功保存至: {filename}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")



if __name__ == "__main__":
    # 调用函数执行
    # w, p99 = analyze_and_plot_refined()
    generate_mapping_table()
    # fit_blockchain_diameter_model(w, p99)