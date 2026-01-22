import os
import pickle
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from graphanalysis.sample_transaction import construct_graph_from_block, load_graph_cache


def prepare_address_node_analysis(self):
    results = defaultdict(int)
    for node, attr in self.graph.nodes(data=True):
        if attr.get("node_type") != "address":
            continue
        n = self.graph.in_degree(node)
        m = self.graph.out_degree(node)
        if n + m == 1:
            continue
        results[n + m] += 1
    return dict(sorted(results.items()))


def plot_bar_broken_axis_aggregated(data_dict):
    # -------------------------------------------------------
    # 1. 数据处理：聚合 > 30 的数据
    # -------------------------------------------------------
    # 分离正常部分 (x <= 30)
    x_normal = sorted([k for k in data_dict.keys() if k <= 30])
    y_normal = [data_dict[k] for k in x_normal]

    # 计算聚合部分 (> 30 的总和)
    y_tail_sum = sum(data_dict[k] for k in data_dict.keys() if k > 30)

    # 合并数据用于绘图
    # 我们为聚合项定义一个虚拟的 x 坐标，比如紧挨着最后一个正常点 + 2 的位置
    # 这样在视觉上它就是“下一根”柱子
    if x_normal:
        tail_x_pos = x_normal[-1] + 1
    else:
        tail_x_pos = 1

    x_plot = x_normal + [tail_x_pos] if y_tail_sum > 0 else x_normal
    y_plot = y_normal + [y_tail_sum] if y_tail_sum > 0 else y_normal

    # -------------------------------------------------------
    # 2. 创建截断坐标轴 (双子图)
    # -------------------------------------------------------
    # height_ratios=[1, 3] 让下半部分稍微高一点，因为那里数据多
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6), dpi=120,
                                   gridspec_kw={'height_ratios': [1, 3]})

    # 调整子图间距
    plt.subplots_adjust(hspace=0.05)

    # 统一柱子样式
    bar_width = 1.0
    bar_color = 'steelblue'  # 统一颜色，不特殊区分

    # -------------------------------------------------------
    # 3. 在两个坐标轴上绘制【同一份聚合后的数据】
    # -------------------------------------------------------
    # 绘制上半部分
    ax1.bar(x_plot, y_plot, width=bar_width, color=bar_color, align='center')
    # 绘制下半部分
    ax2.bar(x_plot, y_plot, width=bar_width, color=bar_color, align='center')

    # -------------------------------------------------------
    # 4. 设置截断范围 (Cut 2000-6000)
    # -------------------------------------------------------
    # ax1 (上图): 显示 6000 以上
    # 动态获取最大值来设置上界
    y_max = max(y_plot) if y_plot else 6000
    # ax1.set_ylim(6000, y_max * 1.1)

    # ax2 (下图): 显示 0 到 2000
    # ax2.set_ylim(0, 2000)

    # -------------------------------------------------------
    # 5. 隐藏边框与绘制截断线
    # -------------------------------------------------------
    # 隐藏 ax1 底边框 和 ax2 顶边框
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # 移除 ax1 的 X 轴刻度，只保留 ax2 的
    ax1.tick_params(axis='x', which='both', length=0)

    # 绘制斜线 //
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # -------------------------------------------------------
    # 6. X 轴刻度与标签处理 (关键)
    # -------------------------------------------------------
    # 生成 0 到 30 的刻度
    ticks = list(range(0, 31, 1))
    labels = [str(t) for t in ticks]

    # 如果有聚合项，追加 ">30" 的标签
    if y_tail_sum > 0:
        ticks.append(tail_x_pos)
        labels.append('>30')

    # 设置刻度 (只需要设置 ax2，因为 sharex=True)
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)

    # 限制 X 轴显示范围，让 ">30" 靠右但不贴边
    # ax2.set_xlim(min(x_plot) - 1 if x_plot else 0, max(x_plot) + 5)

    # -------------------------------------------------------
    # 7. 其他装饰
    # -------------------------------------------------------
    ax1.set_title('Bitcoin Address Node Centrality Distribution')
    # Y轴标签放在中间位置的一种近似做法，只给下面加
    ax2.set_ylabel('Frequency')
    ax2.set_xlabel('Centrality (Degree)')

    # 给两个图都加网格，看起来更连贯
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.show()


def fit_power_y_only(data_dict):
    # --- 1. 数据准备 (保持不变) ---
    x_data = np.array(sorted(data_dict.keys()))
    y_data = np.array([data_dict[x] for x in x_data])

    # --- 2. 定义模型 (保持不变) ---
    def power_law_func(x, A, alpha):
        return A * np.power(x, alpha)

    # --- 3. 拟合 (保持不变: 在原始空间拟合) ---
    # 依然是对原始 x_data 和 y_data 进行拟合，不取对数
    p0 = [max(y_data), -1.0]

    try:
        popt, pcov = curve_fit(power_law_func, x_data, y_data, p0=p0, maxfev=10000)
    except RuntimeError:
        print("拟合失败，未能找到最优参数")
        return

    A_fit, alpha_fit = popt

    # 计算 R^2
    y_pred = power_law_func(x_data, A_fit, alpha_fit)
    r2 = r2_score(y_data, y_pred)

    print(f"拟合参数: A = {A_fit:.2f}, alpha = {alpha_fit:.2f}")
    print(f"拟合优度 (R^2): {r2:.4f}")

    # --- 4. 绘图 (修改部分) ---
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 12

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # A. 生成平滑的曲线数据
    # 因为X轴依然是线性的，所以我们用 linspace
    x_smooth = np.linspace(min(x_data), max(x_data), 3000)
    y_smooth = power_law_func(x_smooth, A_fit, alpha_fit)

    # B. 绘制实际数据 (散点)
    ax.scatter(x_data, y_data, color='#003366', s=15, alpha=0.6,
               edgecolors='none', label='Empirical Data')

    # C. 绘制拟合曲线
    ax.plot(x_smooth, y_smooth, color='#c44e52', linewidth=2, linestyle='--',
            label='Fitted function')

    # E. 图表细节
    ax.set_title('Address Node Centrality Fitting', fontsize=14, pad=15)
    ax.set_xlabel('Centrality', fontsize=12)
    ax.set_ylabel('Node Count', fontsize=12)  # 标签也相应修改提示

    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=10, loc='upper right')

    # 网格线设置 (对数坐标下，开启次级刻度网格会让读数更清晰)
    ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.2)

    plt.tight_layout()
    plt.show()


def plot_bar(data_dict):
    # -------------------------------------------------------
    # 1. 数据准备
    # -------------------------------------------------------
    # 提取 <= 30 的部分
    x_normal = sorted([k for k in data_dict.keys() if k <= 30])
    y_normal = [data_dict[k] for k in x_normal]

    # 计算 > 30 的总和
    y_tail_sum = sum(data_dict[k] for k in data_dict.keys() if k > 30)

    # 确定聚合柱子的 X 轴坐标 (紧跟在最后一个数据后面)
    if x_normal:
        tail_x_pos = x_normal[-1] + 1
    else:
        tail_x_pos = 1

    # 如果有大于30的数据，将其合并到绘图列表中
    x_plot = list(x_normal)
    y_plot = list(y_normal)

    if y_tail_sum > 0:
        x_plot.append(tail_x_pos)
        y_plot.append(y_tail_sum)

    # -------------------------------------------------------
    # 2. 绘图
    # -------------------------------------------------------
    plt.figure(figsize=(12, 6), dpi=120)

    # 【关键修改】 width=0.8 (小于1.0) 即可产生间隙
    # 统一使用 color='steelblue'，不再区分颜色
    bar_width = 0.9

    plt.bar(x_plot, y_plot, width=bar_width, color='steelblue', align='center', label='Frequency')

    # -------------------------------------------------------
    # 3. 标注与美化
    # -------------------------------------------------------
    plt.xlabel('Centrality')
    plt.ylabel('Node Count')
    plt.title('Address Node Centrality Distribution')

    # --- 自定义 X 轴刻度 ---
    # 生成 0 到 30 的刻度
    current_ticks = list(range(0, 31, 5))
    current_labels = [str(t) for t in current_ticks]

    # 追加 ">30" 的标签
    if y_tail_sum > 0:
        current_ticks.append(tail_x_pos)
        current_labels.append('>30')

    plt.xticks(current_ticks, current_labels)

    # 调整 X 轴显示范围
    plt.xlim(min(x_plot) - 1 if x_plot else 0, tail_x_pos + 2)

    plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_bar_broken_axis_aggregated(prepare_address_node_analysis(construct_graph_from_block(928060,928070)))
    # plot_bar(load_data(928060,928070))
    btg = load_graph_cache(928060, 928070, "address_cache.pkl")
    data = prepare_address_node_analysis(btg)
    print(data)
    plot_bar(data)
    fit_power_y_only(data)
