import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sample_transaction import construct_graph_from_block
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


def plot_heat_graph(X, Y, Z):
    data_list = []
    for x, y, z in zip(X, Y, Z):
        data_list.append({'In': x, 'Out': y, 'Count': z})
    df = pd.DataFrame(data_list)

    display_limit = 10
    cap_value = display_limit + 1  # 用 16 代表 15+

    # 将大于 15 的值替换为 16 (即 15+ 类别)
    df['In_Capped'] = df['In'].apply(lambda x: x if x <= display_limit else cap_value)
    df['Out_Capped'] = df['Out'].apply(lambda x: x if x <= display_limit else cap_value)

    # 3. 聚合数据
    # 因为多个原始坐标可能都被映射到了 (16, 16) 或其他位置，所以需要对 Count 求和
    df_grouped = df.groupby(['Out_Capped', 'In_Capped'])['Count'].sum().reset_index()

    # 4. 转换为矩阵 (Pivot Table)
    pivot_table = df_grouped.pivot(index="Out_Capped", columns="In_Capped", values="Count")

    # 关键步骤：重索引 (Reindex)
    # 强制补全 1 到 16 的所有行列，确保矩阵是 16x16 的正方形，缺失值填 0
    # 这样可以防止数据中缺少某一度数导致坐标轴错位
    target_range = range(1, cap_value + 1)
    pivot_table = pivot_table.reindex(index=target_range, columns=target_range, fill_value=0)

    # 5. 绘图配置
    plt.figure(figsize=(12, 10))

    # 生成坐标轴标签: ['1', '2', ..., '15', '15+']
    labels = [str(i) for i in range(1, display_limit + 1)] + [f'{display_limit}+']

    # 绘制热力图
    ax = sns.heatmap(pivot_table,
                     annot=True,
                     fmt='g',
                     cmap='YlGnBu',
                     norm=LogNorm(),
                     linewidths=.5,
                     cbar_kws={'label': 'Transaction Count (Log Scale)'},
                     xticklabels=labels,  # 设置自定义 X 轴标签
                     yticklabels=labels)  # 设置自定义 Y 轴标签

    # 6. 坐标轴调整
    ax.invert_yaxis()  # Y轴从下往上
    plt.xlabel('In-Degree (Number of Inputs)', fontsize=12)
    plt.ylabel('Out-Degree (Number of Outputs)', fontsize=12)
    plt.title(f'Heatmap of Transaction Types (Aggregated {display_limit}+)', fontsize=14)

    # 显示
    plt.tight_layout()
    plt.show()

# 文件缓存
def load_data(
        startId, endId,
        cache_path="transaction_node_cache.pkl"
):
    if os.path.exists(cache_path):
        print(f"从缓存文件加载数据: {cache_path}")
        with open(cache_path, "rb") as f:
            X, Y, Z = pickle.load(f)
    else:
        print("缓存不存在，开始重新采样...")
        X, Y, Z = prepare_data(
            construct_graph_from_block(startId, endId)
        )
        with open(cache_path, "wb") as f:
            pickle.dump((X, Y, Z), f)
        print(f"采样结果已保存至: {cache_path}")

    return X, Y, Z




def fit_cutoff_power_law_2d_2(X, Y, Z):
    # 定义拟合函数
    def model_func(data, A, alpha, beta, lam1, lam2, lam3):
        n, m = data
        # 指数部分引入各向异性和交互项: - (λ1*n + λ2*m + λ3*sqrt(nm))
        exponent = -(lam1 * n + lam2 * m + lam3 * np.sqrt(n * m))
        return A * np.power(n, alpha) * np.power(m, beta) * np.exp(exponent)
    # 执行拟合
    initial_guess = [1000, 2.0, 2.0, 0.01, 0.01, 0.001]
    lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    try:
        # 执行拟合
        popt, pcov = curve_fit(
            model_func, (X, Y), Z,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20000  # 增加迭代次数，因为参数变多了
        )
    except RuntimeError as e:
        print(f"拟合失败: {e}")
        return
    A_fit, alpha_fit, beta_fit, lam1_fit, lam2_fit, lam3_fit = popt
    print("-" * 30)
    print("【模型四：二维指数截断幂律分布】")
    func_str = (f"y(n,m) = {A_fit:.2f} * n^({alpha_fit:.2f}) * m^({beta_fit:.2f}) * "
                f"e^(-({lam1_fit:.4f}n + {lam2_fit:.4f}m + {lam3_fit:.4f}√nm))")
    print(f"拟合公式: {func_str}")
    # 计算 R^2
    model_value = model_func((X, Y), *popt)
    # X2, Y2, Z2 = prepare_data_from_block(928055, 928060)
    # r2 = r2_score(Z2[:100], model_value[:100])
    r2 = r2_score(Z, model_value)
    print(f"R^2: {r2:.4f}")
    # 绘制图像
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    # ===============================
    # 1. 真实数据：灰色散点（论文标准）
    # ===============================
    ax.scatter(
        X, Y, Z,
        color='black',
        s=12,
        alpha=0.35,
        linewidth=0,
        label='Empirical data'
    )
    # 生成网格用于绘制平滑曲面
    # 使用 logspace 确保在低数值区间（如 1-10）有足够的采样密度
    x_range = np.logspace(np.log10(min(X)), np.log10(max(X)), 60)
    y_range = np.logspace(np.log10(min(Y)), np.log10(max(Y)), 60)
    X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
    # 计算网格上的高度
    exponent_mesh = -(lam1_fit * X_mesh + lam2_fit * Y_mesh + lam3_fit * np.sqrt(X_mesh * Y_mesh))
    Z_mesh = A_fit * np.power(X_mesh, alpha_fit) * np.power(Y_mesh, beta_fit) * np.exp(exponent_mesh)

    # ===============================
    # 2. 拟合曲面：半透明连续面
    # ===============================
    surf = ax.plot_surface(
        X_mesh, Y_mesh, Z_mesh,
        color='tab:red',
        alpha=0.45,
        linewidth=0,
        antialiased=True
    )

    # ===============================
    # 3. 坐标轴设置（数学符号）
    # ===============================
    ax.set_xlabel(r'In-degree $n$', labelpad=8)
    ax.set_ylabel(r'Out-degree $m$', labelpad=8)
    ax.set_zlabel(r'Frequency $P(n,m)$', labelpad=8)

    # 对于度分布，**非常建议使用对数坐标**
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_zscale('log')

    # ===============================
    # 4. 视角（论文常用）
    # ===============================
    ax.view_init(elev=25, azim=45)

    # ===============================
    # 5. 网格与背景（弱化）
    # ===============================
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    # ===============================
    # 6. 标题（简洁、学术）
    # ===============================
    ax.set_title(
        'Joint in-degree and out-degree of transaction node distribution',
        y=1.03
    )

    # ===============================
    # 7. 图例（小而克制）
    # ===============================
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='black',
            linestyle='None',
            markersize=6,
            alpha=0.6,
            label='Empirical data'
        ),
        Patch(
            facecolor='tab:red',
            edgecolor='tab:red',
            alpha=0.45,
            label='Fitted function'
        )
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(1.0, 0.95),  # 第二个参数 < 1 → 向下
        frameon=False
    )
    plt.tight_layout()
    plt.show()

def prepare_data(self):
    joint_dist = defaultdict(int)
    for node, attr in self.graph.nodes(data=True):
        if attr.get("node_type") != "transaction":
            continue
        x = self.graph.in_degree(node)
        y = self.graph.out_degree(node)
        joint_dist[(x, y)] += 1
    x_data = []
    y_data = []
    z_data = []
    for (x, y), count in joint_dist.items():
        if x == 0 or y == 0:
            continue
        if x == 1 and (y==1 or y==2):
            continue
        # if 1 < count and 0 < n < 500 and 0 < m < 500:
        x_data.append(x)
        y_data.append(y)
        z_data.append(count)

    # 转换为 numpy 数组以便计算
    X = np.array(x_data)
    Y = np.array(y_data)
    Z = np.array(z_data)
    print(X)
    print(Y)
    print(Z)
    return X, Y, Z

if __name__ == "__main__":
    # fit_log_power_law_2d()
    # fit_cutoff_power_law_2d()
    # X, Y, Z = load_data(928060, 928070)
    X, Y, Z = prepare_data(
        construct_graph_from_block(928060, 928070)
    )
    # fit_cutoff_power_law_2d(X, Y, Z)
    fit_cutoff_power_law_2d_2(X, Y, Z)





#
# def fit_power_law_2d(X, Y, Z):
#     # 定义拟合函数：二维幂律分布 (Zipf)
#     # 公式: z = A * x^(-alpha) * y^(-beta)
#     def model_func(data, A, alpha, beta):
#         n, m = data
#         return A * np.power(n, -alpha) * np.power(m, -beta)
#
#     # 执行拟合
#     popt, pcov = curve_fit(model_func, (X, Y), Z, p0=[1000, 2.0, 2.0], maxfev=5000)
#     A_fit, alpha_fit, beta_fit = popt
#     print("-" * 30)
#     print("【模型一：标准幂律分布】")
#     print(f"拟合公式: z = {A_fit:.2f} * n^(-{alpha_fit:.2f}) * m^(-{beta_fit:.2f})")
#     model_value = model_func((X, Y), *popt)
#     r2 = r2_score(Z, model_value)
#     print(f"R^2: {r2:.4f}")
#
#     # 绘制图像
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制真实数据点 (散点)
#     ax.scatter(X, Y, Z, c=np.log1p(Z), cmap='viridis', s=20, label='Filtered Data', alpha=0.7)
#     # 绘制拟合曲面 (网格)
#     x_range = np.linspace(min(X), max(X), 50)
#     y_range = np.linspace(min(Y), max(Y), 50)
#     X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
#     Z_mesh = model_func((X_mesh, Y_mesh), *popt)
#     ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='orangered', alpha=0.5, rstride=2, cstride=2,
#                       label='Fitted Function')
#     ax.set_xlabel('Degree n')
#     ax.set_ylabel('Degree m')
#     ax.set_zlabel('Count (Frequency)')
#     ax.set_title('2D Power Law Fitting (n>0, m>0, count<10000)')
#     # 关键：设置 Z 轴为对数坐标，因为幂律分布在普通坐标下看不清尾部拟合情况
#     plt.legend()
#     plt.show()
#
#
# def fit_log_power_law_2d(X, Y, Z):
#     log_Z = np.log(Z)  # 关键：对频次取对数
#
#     # 2. 定义对数模型函数
#     # log(z) = log(A) - alpha*log(n) - beta*log(m)
#     def model_func(data, log_A, alpha, beta):
#         n, m = data
#         return log_A - alpha * np.log(n) - beta * np.log(m)
#
#     # 3. 执行拟合 (针对 log_Z)
#     popt, pcov = curve_fit(model_func, (X, Y), log_Z, p0=[np.log(1000), 2.0, 2.0], maxfev=5000)
#     log_A_fit, alpha_fit, beta_fit = popt
#     print("-" * 30)
#     print("【模型二：对数空间标准幂律】")
#     print(f"拟合公式: ln(z) = {log_A_fit:.2f} - {alpha_fit:.2f}*ln(n) - {beta_fit:.2f}*ln(m)")
#     model_value = model_func((X, Y), *popt)
#     r2 = r2_score(log_Z, model_value)
#     print(f"R^2: {r2:.4f}")
#
#     # 5. 绘制图像
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # 绘制真实数据点 (Z轴显示为 Log Count)
#     ax.scatter(X, Y, log_Z, c=log_Z, cmap='viridis', s=20, label='Real Data (Log)', alpha=0.7)
#     # 绘制拟合曲面
#     x_range = np.linspace(min(X), max(X), 50)
#     y_range = np.linspace(min(Y), max(Y), 50)
#     X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
#     Z_mesh_log = model_func((X_mesh, Y_mesh), *popt)  # 计算出来的直接就是 log 值
#     ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh_log, color='orangered', alpha=0.5, rstride=2, cstride=2,
#                       label='Fitted Log-Plane')
#     ax.set_xlabel('Degree n')
#     ax.set_ylabel('Degree m')
#     ax.set_zlabel('Log(Count)')  # 注意 Z 轴含义变了
#     ax.set_title(f'Log-Space Power Law Fitting (R^2={r2:.4f})')
#
#     plt.legend()
#     plt.show()
#
#
# def fit_log_cutoff_power_law_2d(X, Y, Z):
#     # 1. 准备数据
#     log_Z = np.log(Z)
#
#     # 2. 定义带截断的对数模型函数
#     # log(z) = log(A) - alpha*log(n) - beta*log(m) - lambda*(n + m)
#     def model_func(data, log_A, alpha, beta, lam):
#         n, m = data
#         return log_A - alpha * np.log(n) - beta * np.log(m) - lam * (n + m)
#
#     # 3. 执行拟合
#     # lam 初始值给一个小正数，比如 0.01
#     popt, pcov = curve_fit(model_func, (X, Y), log_Z, p0=[np.log(1000), 2.0, 2.0, 0.01], maxfev=10000)
#     log_A_fit, alpha_fit, beta_fit, lam_fit = popt
#     print("-" * 30)
#     print("【模型三：对数空间截断幂律】")
#     print(f"拟合公式: ln(z) = {log_A_fit:.2f} - {alpha_fit:.2f}ln(n) - {beta_fit:.2f}ln(m) - {lam_fit:.4f} * (n+m)")
#     model_value = model_func((X, Y), *popt)
#     # X2, Y2, Z2 = prepare_data_from_block(928055, 928060)
#     # r2 = r2_score( np.log(Z2)[:100], model_value[:100])
#     r2 = r2_score(log_Z, model_value)
#
#     print(f"对数空间 R^2: {r2:.4f}")
#
#     # 5. 绘制图像
#     fig = plt.figure(figsize=(16, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     # 绘制真实数据点 (散点)
#     ax.scatter(X, Y, log_Z, c='orangered', s=15, label='Real Data', depthshade=False)
#     # 绘制拟合曲面 (Surface)
#     x_range = np.linspace(min(X), max(X), 50)
#     y_range = np.linspace(min(Y), max(Y), 50)
#     X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
#     Z_mesh = model_func((X_mesh, Y_mesh), *popt)
#     # plot_surface: 绘制半透明实体面，alpha=0.3 让我们可以透过曲面看到后面的点
#     ax.plot_surface(X_mesh, Y_mesh, Z_mesh, cmap='Blues',
#                     alpha=0.7, edgecolor='none', label='Fitted Model')
#     # 绘制残差垂线 (Residual Lines)
#     # # 遍历每个点，画一条从“真实值”到“预测曲面值”的直线
#     # for i in range(len(X)):
#     #     x_pt, y_pt = X[i], Y[i]
#     #     z_real = log_Z[i]
#     #     z_pred = model_value[i]
#     #     # 线的颜色根据误差方向变化：上方为红色线，下方为绿色线 (或者统一灰色)
#     #     line_color = 'gray'
#     #     # 仅绘制误差较大的线的连接，避免图太乱 (可选)
#     #     ax.plot([x_pt, x_pt], [y_pt, y_pt], [z_real, z_pred],
#     #             color=line_color, alpha=0.3, linewidth=1)
#     # D. 设置坐标轴和视角
#     ax.set_xlabel('Degree n', fontsize=12)
#     ax.set_ylabel('Degree m', fontsize=12)
#     ax.set_zlabel('Log(Count)', fontsize=12)
#
#     # 调整视角，以便更好地观察长尾部分
#     # # elev: 仰角, azim: 方位角
#     # ax.view_init(elev=25, azim=-60)
#     # # 手动添加图例 (Surface对象有时难以直接添加图例)
#     # import matplotlib.lines as mlines
#     # red_dot = mlines.Line2D([], [], color='orangered', marker='o', linestyle='None', markersize=8, label='Real Data')
#     # blue_patch = mlines.Line2D([], [], color='lightblue', marker='s', linestyle='None', markersize=10, alpha=0.5,
#     #                            label='Fitted Surface')
#     # gray_line = mlines.Line2D([], [], color='gray', linestyle='-', linewidth=1, label='Error (Residual)')
#     # ax.legend(handles=[red_dot, blue_patch, gray_line], loc='upper right')
#     plt.tight_layout()
#     plt.show()
#
#
# def fit_cutoff_power_law_2d(X, Y, Z):
#     # 定义拟合函数
#     def model_func(data, A, alpha, beta, lam1, lam2, lam3):
#         n, m = data
#         # 指数部分引入各向异性和交互项: - (λ1*n + λ2*m + λ3*sqrt(nm))
#         exponent = -(lam1 * n + lam2 * m + lam3 * np.sqrt(n * m))
#         return A * np.power(n, alpha) * np.power(m, beta) * np.exp(exponent)
#     # 执行拟合
#     initial_guess = [1000, 2.0, 2.0, 0.01, 0.01, 0.001]
#     lower_bounds = [0, -np.inf, -np.inf, 0, 0, -np.inf]
#     upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
#     try:
#         # 执行拟合
#         popt, pcov = curve_fit(
#             model_func, (X, Y), Z,
#             p0=initial_guess,
#             bounds=(lower_bounds, upper_bounds),
#             maxfev=20000  # 增加迭代次数，因为参数变多了
#         )
#     except RuntimeError as e:
#         print(f"拟合失败: {e}")
#         return
#     A_fit, alpha_fit, beta_fit, lam1_fit, lam2_fit, lam3_fit = popt
#     print("-" * 30)
#     print("【模型四：二维指数截断幂律分布】")
#     func_str = (f"y(n,m) = {A_fit:.2f} * n^({alpha_fit:.2f}) * m^({beta_fit:.2f}) * "
#                 f"e^(-({lam1_fit:.4f}n + {lam2_fit:.4f}m + {lam3_fit:.4f}√nm))")
#     print(f"拟合公式: {func_str}")
#     # 计算 R^2
#     model_value = model_func((X, Y), *popt)
#     # X2, Y2, Z2 = prepare_data_from_block(928055, 928060)
#     # r2 = r2_score(Z2[:100], model_value[:100])
#     r2 = r2_score(Z, model_value)
#     print(f"R^2: {r2:.4f}")
#     # 绘制图像
#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     # 绘制真实数据点 (散点)
#     ax.scatter(X, Y, Z, c=np.log1p(Z), cmap='viridis', s=20, label='Reality Data', alpha=0.7)
#     # 生成网格用于绘制平滑曲面
#     # 使用 logspace 确保在低数值区间（如 1-10）有足够的采样密度
#     x_range = np.logspace(np.log10(min(X)), np.log10(max(X)), 60)
#     y_range = np.logspace(np.log10(min(Y)), np.log10(max(Y)), 60)
#     X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
#     # 计算网格上的高度
#     exponent_mesh = -(lam1_fit * X_mesh + lam2_fit * Y_mesh + lam3_fit * np.sqrt(X_mesh * Y_mesh))
#     Z_mesh = A_fit * np.power(X_mesh, alpha_fit) * np.power(Y_mesh, beta_fit) * np.exp(exponent_mesh)
#     # 绘制曲面
#     ax.plot_wireframe(X_mesh, Y_mesh, Z_mesh, color='orangered', alpha=0.4, rstride=2, cstride=2,
#                       label='Fitted function')
#
#     ax.set_xlabel('In-Degree')
#     ax.set_ylabel('Out-Degree')
#     ax.set_zlabel('Count')
#     ax.set_title("Transaction node in-degree and out-degree distribution", fontsize=10)
#     # 调整视角以便观察
#     ax.view_init(elev=25, azim=45)
#     plt.legend()
#     plt.show()
