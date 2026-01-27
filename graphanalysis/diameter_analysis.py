import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def analyze_original_diameter():
    # 1. 定义文件列表
    files = [
        "diameter/dist_stats_start_923800.jsonl",
        "diameter/dist_stats_start_923820.jsonl",
        "diameter/dist_stats_start_923840.jsonl",
        "diameter/dist_stats_start_923860.jsonl",
        "diameter/dist_stats_start_923880.jsonl",
        "diameter/dist_stats_start_923900.jsonl",
        "diameter/dist_stats_start_923920.jsonl",
        "diameter/dist_stats_start_923940.jsonl",
        "diameter/dist_stats_start_923960.jsonl",
        "diameter/dist_stats_start_923980.jsonl"
    ]

    data = []

    print("开始读取文件并处理数据...")
    # 2. 读取文件并提取数据
    for file in files:
        with open(file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    dist_list = record.get('dist_list', [])

                    if dist_list:
                        dist_array = np.array(dist_list)

                        # 3. 计算统计量
                        # 使用分位数 (P99, P95) 来作为更稳健的直径指标，过滤极端值
                        max_d = np.max(dist_array)
                        p99 = np.percentile(dist_array, 99)
                        p95 = np.percentile(dist_array, 95)

                        data.append({
                            'window_size': record.get('window_size'),
                            'max_dist': max_d,
                            'p99_dist': p99,
                            'p95_dist': p95
                        })
                except json.JSONDecodeError:
                    continue

    # 转换为 DataFrame 方便分析
    df = pd.DataFrame(data)

    # 4. 数据聚合分析
    # 按窗口大小分组，计算均值(mean)和标准差(std)
    # 标准差用于绘制误差棒，展示不同文件间数据的波动情况
    grouped = df.groupby('window_size').agg({
        'max_dist': ['mean', 'std'],
        'p99_dist': ['mean', 'std'],
        'p95_dist': ['mean', 'std']
    }).reset_index()

    # 展平多层列名
    grouped.columns = ['window_size', 'max_mean', 'max_std', 'p99_mean', 'p99_std', 'p95_mean', 'p95_std']

    print("分析完成，统计结果如下：")
    print(grouped)

    # 5. 结果绘图
    plt.figure(figsize=(12, 6))

    # 绘制 Max 直径 (虚线)
    plt.errorbar(grouped['window_size'], grouped['max_mean'], yerr=grouped['max_std'],
                 label='Max Diameter (Mean)', capsize=5, marker='o', linestyle='--', alpha=0.7)

    # 绘制 P99 直径 (实线，推荐指标)
    plt.errorbar(grouped['window_size'], grouped['p99_mean'], yerr=grouped['p99_std'],
                 label='99th Percentile Diameter (Mean)', capsize=5, marker='s', linestyle='-', linewidth=2)

    # 绘制 P95 直径 (点线)
    plt.errorbar(grouped['window_size'], grouped['p95_mean'], yerr=grouped['p95_std'],
                 label='95th Percentile Diameter (Mean)', capsize=5, marker='^', linestyle=':', alpha=0.7)

    plt.title('Blockchain Transaction Graph Diameter vs. Window Size')
    plt.xlabel('Window Size (Number of Blocks)')
    plt.ylabel('Diameter (Distance)')
    plt.xticks(grouped['window_size'])  # 确保x轴显示所有整数窗口大小
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    return grouped['window_size'].values, grouped['p99_mean'].values


def fit_blockchain_diameter_model(x, y):
    # 定义幂律函数模型: y = a * x^b
    def power_func(x, a, b):
        return a * np.power(x, b)
    
    # 4. 执行曲线拟合
    try:
        # curve_fit 返回拟合参数 popt 和协方差矩阵 pcov
        popt, pcov = curve_fit(power_func, x, y)
        a_fit, b_fit = popt
    except RuntimeError:
        print("错误: 曲线拟合失败。")
        return None

    # 5. 计算 R-squared (决定系数) 来评估拟合好坏
    y_pred = power_func(x, a_fit, b_fit)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 6. 绘图展示 (可选)
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='black', label='实际数据 (平均 P99)')
    
    # 生成平滑曲线
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = power_func(x_smooth, a_fit, b_fit)
    
    plt.plot(x_smooth, y_smooth, color='red', linestyle='-', linewidth=2, 
             label=f'拟合曲线: $D = {a_fit:.4f} \cdot x^{{{b_fit:.4f}}}$\n($R^2 = {r_squared:.4f}$)')
    
    plt.title('Blockchain Transaction Graph Diameter Fitting (Power Law)')
    plt.xlabel('Window Size (Blocks)')
    plt.ylabel('Diameter Threshold (P99)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show() # 或者 plt.savefig('fitting_result.png')
    
    result = {
        'a': a_fit,
        'b': b_fit,
        'r_squared': r_squared,
        'formula': f"Diameter = {a_fit:.4f} * (Window_Size)^{b_fit:.4f}"
    }
    
    return result


if __name__ == "__main__":
    # 调用函数执行
    w, p99 = analyze_original_diameter()
    fit_blockchain_diameter_model(w, p99)