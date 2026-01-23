import random
from collections import defaultdict

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import wasserstein_distance
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns

# from graphanalysis.sample_transaction import construct_graph_from_block, load_transactions_from_file
from txgraph.main import BitcoinTransactionGraph
import sample_transaction

import os
import pickle


def load_or_collect_diameter_data(
        all_transactions,
        scales,
        samples_per_scale,
        cache_path="diameter_cache.pkl"
):
    if os.path.exists(cache_path):
        print(f"从缓存文件加载数据: {cache_path}")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
    else:
        print("缓存不存在，开始重新采样...")
        data = collect_diameter_data(
            all_transactions,
            scales,
            samples_per_scale
        )
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"采样结果已保存至: {cache_path}")

    return data


def get_max_diameter(btg):
    """
    计算图中任意两个可达节点之间经过的最小边数的最大值。
    """
    G = btg.graph
    if len(G) == 0:
        return 0
    max_distance = 0
    for source, dist_map in nx.shortest_path_length(G):
        if dist_map:
            current_max = max(dist_map.values())
            if current_max > max_distance:
                max_distance = current_max
    return max_distance


def collect_diameter_data(all_transactions, scale_list, samples_per_scale=30):
    """
    模块 1: 进行多尺度随机采样并计算直径。

    返回:
        dict: key = 实际节点数 n
              value = 该 n 下观测到的所有路径长度 l 的列表
    """
    results = defaultdict(list)
    print(f"开始采样分析，总交易池大小: {len(all_transactions)}")
    for target_size in scale_list:
        print(f"正在分析规模 N ≈ {target_size}...")
        for _ in range(samples_per_scale):
            # 防止采样规模超过交易池
            sample_size = min(target_size, len(all_transactions))
            sampled_txs = random.sample(all_transactions, sample_size)
            # 构建临时子图
            btg_temp = BitcoinTransactionGraph()
            for tx in sampled_txs:
                btg_temp.add_transaction(
                    tx['hash'],
                    tx['input_addrs'],
                    tx['output_addrs']
                )
            # 记录节点数与直径
            num_nodes = btg_temp.graph.number_of_nodes()
            if num_nodes > 1:
                diameter = get_max_diameter(btg_temp)
                results[num_nodes].append(diameter)
    return dict(results)


def fit_function(node_cnt, path_len):
    # 模块 2: 使用对数函数进行非线性拟合。
    def fit_func(x, a, b):
        return a * np.power(x, b)

    # 拟合
    params, _ = curve_fit(fit_func, node_cnt, path_len)
    a, b = params
    # 计算拟合优度 R^2
    r2 = r2_score(path_len, fit_func(node_cnt, a, b))

    print(f"\n拟合完成！")
    print(f"拟合函数: Threshold(N) = {a:.6f} * N^{b:.4f}")
    print(f"拟合优度 R^2: {r2:.4f}")

    # 模块 3: 可视化拟合结果。
    plt.figure(figsize=(10, 6))
    plt.scatter(node_cnt, path_len, color='blue', alpha=0.3, label='Sampled Data')
    # 绘制拟合曲线
    x_fit = np.linspace(min(node_cnt), max(node_cnt), 100)
    y_fit = fit_func(x_fit, a, b)
    label_str = f'Power Law Fit (R2={r2:.3f})'
    plt.plot(x_fit, y_fit, color='red', linewidth=2,
             label=label_str)
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Network Diameter (L)')
    plt.title('Blockchain Graph Diameter vs. Node Count')
    plt.legend()
    plt.grid(True)
    plt.show()
    return a, b, r2


def fit_quantile_threshold(node_cnt, path_len, quantile=0.95):
    print(f"\n--- 开始分位数回归分析 (Tau={quantile}) ---")

    # 1. 整理数据为 DataFrame
    df = pd.DataFrame({'N': node_cnt, 'L': path_len})
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df, x='N', y='L',
        fill=True,
        thresh=0.05,  # 忽略密度最低的 5% 区域，让图更干净
        levels=20,  # 画 20 层等高线，看起来非常连续
        bw_adjust=1.0,  # 平滑系数，可尝试调整为 1.5 或 0.5
        cmap="Spectral_r"
    )
    plt.title("Continuous Density Estimation (2D KDE)")
    plt.show()


# 1. 绘制散点图
def plot_scatter(data_dict, jitter=0.1):
    """
    绘制节点数 n 与路径长度 l 的散点图，用于展示非函数关系
    data_dict: Dict[int, List[int]]
    jitter: x 轴抖动幅度，避免点重叠
    """

    xs = []
    ys = []

    for n, l_list in data_dict.items():
        for l in l_list:
            xs.append(n + np.random.uniform(-jitter, jitter))
            ys.append(l)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.5, s=15)

    plt.xlabel("Number of nodes")
    plt.ylabel("Diameter")
    plt.title("Relationship between graph size and diameter")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# 2. 统计分箱宽度
def analyze_max_wasserstein_vs_binwidth(data_dict,
                                        width_range=range(10, 301, 10),
                                        min_samples=50,
                                        output_file='max_wasserstein_analysis.png'):
    """
    计算并绘制：分箱宽度 vs. 该宽度下所有分箱中Wasserstein距离的最大值

    参数:
    - data_dict: {节点数: [路径长度列表]}
    - width_range: 想要测试的分箱宽度序列 (例如 10, 20, ... 200)
    - min_samples: 只有当某个节点数的样本量超过此值才参与计算 (过滤噪声)
    """

    # 1. 数据预处理：获取所有有效的节点数
    # 过滤掉样本极少的节点，保证统计显著性
    valid_nodes = sorted([k for k, v in data_dict.items() if len(v) >= min_samples])

    if not valid_nodes:
        print("错误：有效数据不足。")
        return

    min_node = valid_nodes[0]
    max_node = valid_nodes[-1]

    plot_data = []

    # 2. 遍历每一个测试的分箱宽度 (Bin Width)
    for width in width_range:
        bin_wassersteins = []

        # 按照当前宽度 width 进行切分
        # Bin 1: [start, start + width)
        # Bin 2: [start + width, start + 2*width) ...
        current_start = min_node

        while current_start < max_node:
            current_end_bound = current_start + width

            # 在当前分箱区间 [current_start, current_end_bound) 内寻找数据
            # 找到该区间内实际存在的 最小节点数 (First) 和 最大节点数 (Last)
            nodes_in_bin = [n for n in valid_nodes if current_start <= n < current_end_bound]

            if len(nodes_in_bin) >= 2:
                # 只有当分箱内至少有两个不同的有效节点数时，才能计算跨度
                first_node = nodes_in_bin[0]
                last_node = nodes_in_bin[-1]

                # 如果 first 和 last 是同一个点，距离为0，无意义，跳过
                if first_node != last_node:
                    dist = wasserstein_distance(data_dict[first_node], data_dict[last_node])
                    bin_wassersteins.append(dist)

            # 移动到下一个分箱
            current_start += width

        # 3. 记录当前宽度下的 最大 Wasserstein 距离
        if bin_wassersteins:
            percentile = 90  # 取 90 百分位数作为“最大”距离的代表
            max_w = np.percentile(bin_wassersteins, percentile)
            avg_w = np.mean(bin_wassersteins)  # 顺便记录平均值供参考
            plot_data.append({
                'Bin Width': width,
                'Max Wasserstein Distance': max_w,
                'Avg Wasserstein Distance': avg_w
            })

    df_res = pd.DataFrame(plot_data)

    if df_res.empty:
        print("未能计算出结果，请检查数据密度或分箱范围。")
        return

    # 4. 绘图
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # 绘制核心曲线：最大距离
    sns.lineplot(data=df_res, x='Bin Width', y='Max Wasserstein Distance',
                 marker='o', linewidth=2.5, color='#d62728', label=f'Max Divergence ({percentile}%)')

    # (可选) 绘制平均距离作为对比背景
    sns.lineplot(data=df_res, x='Bin Width', y='Avg Wasserstein Distance',
                 marker='x', linestyle='--', color='gray', alpha=0.6, label='Avg Divergence (Reference)')

    # 添加辅助阈值线 (例如 0.2 或 0.5，视具体业务容忍度而定)
    # 这里的阈值意味着：在这个宽度下，最坏情况下的分布漂移量
    threshold = 0.3
    plt.axhline(y=threshold, color='green', linestyle=':', label=f'Tolerance Threshold ({threshold})')

    plt.title("Maximum Wasserstein distance under different bin width", fontsize=14, pad=15)
    plt.xlabel("Bin Width ($x$)", fontsize=12)
    plt.ylabel("Max Wasserstein Distance in Bins", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    x_max = df_res['Bin Width'].max()
    # 创建从x_min到x_max，步长为10的刻度
    xticks = np.arange(0, x_max + 1, 20)  # 加1确保包含最大值
    plt.xticks(xticks)
    plt.tight_layout()
    plt.show()

    return df_res


# 3. 统计概率分布
def generate_prob_matrix(data_dict, max_node=300, bin_width=30):
    """
    输入：原始数据 dict
    输出：
    1. prob_matrix: 供程序调用的概率矩阵 DataFrame
    2. 生成可视化的表格图片
    """

    # 1. 数据展平
    flattened_data = []
    for node_count, lengths in data_dict.items():
        if not lengths: continue
        for length in lengths:
            flattened_data.append({'Node Count': node_count, 'Path Length': length})

    df = pd.DataFrame(flattened_data)

    # 2. 严格设定分箱范围：0 到 300 (左闭右开)
    # range(0, 331, 30) 是为了确保能覆盖到 300 这个点（分箱将是 [270, 300), [300, 330)）
    # 如果您的数据严格小于300，可以设为 range(0, 301, 30)
    bins = list(range(0, max_node + bin_width + 1, bin_width))

    # 3. 生成符合数学规范的区间标签 "[0,30)"
    labels = [f"[{bins[i]},{bins[i + 1]})" for i in range(len(bins) - 1)]

    # 4. 执行分箱 (right=False 表示左闭右开)
    df['Bin_Label'] = pd.cut(df['Node Count'], bins=bins, labels=labels, right=False)

    # 5. 计算概率矩阵 (列归一化)
    # 这一步生成的表格，每一列的和都是 1.0
    prob_matrix = pd.crosstab(index=df['Path Length'],
                              columns=df['Bin_Label'],
                              normalize='columns')

    # 缺失值补0（表示该长度在该节点范围内从未出现）
    prob_matrix = prob_matrix.fillna(0)

    # --- 绘制并保存表格图片 (用于论文展示) ---
    plt.figure(figsize=(14, len(prob_matrix) * 0.6 + 2))
    ax = plt.gca()
    ax.axis('off')

    # 格式化显示数据 (保留4位小数)
    display_matrix = prob_matrix.applymap(lambda x: f"{x:.4f}")

    table = ax.table(cellText=display_matrix.values,
                     rowLabels=display_matrix.index,
                     colLabels=display_matrix.columns,
                     cellLoc='center',
                     loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    # 设置表头颜色或加粗
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')  # 浅灰色表头

    plt.title(f"Path Length Probability Distribution\n(Range [0, {max_node}], Bin Width {bin_width})",
              fontsize=14, pad=15, weight='bold')
    plt.tight_layout()
    plt.show()

    return prob_matrix


def save_matrix_to_excel(prob_matrix, filename='path_length_probability_matrix.xlsx'):
    """
    将概率矩阵保存为 Excel 文件，并进行格式化美化
    """
    # 创建 ExcelWriter 对象 (使用 xlsxwriter 引擎以支持格式设置)
    # 如果没有安装 xlsxwriter，pip install xlsxwriter，或者去掉 engine 参数用默认的
    try:
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        # 写入数据
        # float_format="%.4f": 关键参数，确保 Excel 里显示的数字保留4位小数
        prob_matrix.to_excel(writer, sheet_name='Probability Distribution', float_format="%.4f")

        # --- 以下是美化部分 (可选) ---
        workbook = writer.book
        worksheet = writer.sheets['Probability Distribution']

        # 定义格式：居中对齐
        cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter'})

        # 设置列宽，让表格看起更清楚
        # 设置第一列 (索引列: 路径长度) 宽度
        worksheet.set_column(0, 0, 15, cell_format)
        # 设置数据列 (分箱列) 宽度
        worksheet.set_column(1, len(prob_matrix.columns), 12, cell_format)

        # 保存文件
        writer.close()
        print(f"✅ 文件已成功保存为: {filename}")

    except Exception as e:
        # 如果出错（比如没装 xlsxwriter），回退到最简单的保存方式
        print(f"高级格式保存失败 ({e})，正在尝试普通保存...")
        prob_matrix.to_excel(filename, float_format="%.4f")
        print(f"✅ 文件已以普通格式保存为: {filename}")

if __name__ == "__main__":
    btg = BitcoinTransactionGraph()
    all_transactions = []
    for i in range(928050, 928060):
        filename = f"../dataset/transactions_block_{i}.json"
        file_transactions = sample_transaction.load_transactions_from_file(filename)
        all_transactions.extend(file_transactions)
        # all_transactions.extend(random.sample(file_transactions, 10))
    # for tx in all_transactions:
    #     btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])

    scales = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
    # 固定采样结果
    random.seed(42)
    dict = load_or_collect_diameter_data(all_transactions, scales, samples_per_scale=10000)
    # plot_scatter(dict)
    # get_safe_path_length = fit_quantile_threshold(n_data, l_data, quantile=0.95)
    # verify_bin_stability(n_data, l_data, 20)
    # analyze_max_wasserstein_vs_binwidth(dict)
    save_matrix_to_excel(generate_prob_matrix(dict,270))
    # 拟合函数
    # a, b, r2 = fit_function(n_data, l_data)
