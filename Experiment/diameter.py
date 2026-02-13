import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import copy
import os
import random
import pandas as pd
import re
import matplotlib.patches as patches
import platform
# 引入项目模块 (确保这些路径在你的项目中存在)
from txgraph.main import BitcoinTransactionGraph
from graphanalysis.sample_transaction import load_transactions_from_file

# ==========================================
# 1. 基础构建函数
# ==========================================
def constuct_graph(tx_list):
    """根据交易列表构建图"""
    btg = BitcoinTransactionGraph()
    for tx in tx_list:
        btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
    return btg

def merge_graphs(G_base_nx, G_inject_nx):
    """合并正常图与隐蔽图"""
    G_mixed_nx = copy.deepcopy(G_base_nx)
    G_mixed_nx = nx.compose(G_mixed_nx, G_inject_nx)
    return G_mixed_nx

# ==========================================
# 2. 核心算法：近似有向直径计算
# ==========================================
def calculate_approx_directed_diameter(G):
    """
    计算有向图的近似直径（忽略不可达节点）
    
    原理：
    1. 随机选取 sample_size 个起始节点（探测器）。
    2. 对每个起点，计算它能到达的所有节点的最短路径。
    3. 找到所有可达路径中的最大值。
    
    :param G: NetworkX DiGraph
    :param sample_size: 采样次数，建议 50-100，越大越准但越慢
    :return: 估计的最大跳数 (Integer)
    """
    nodes = list(G.nodes())
    target_sources = nodes
    max_dist = 0
    
    # 开始探测
    for start_node in target_sources:
        # nx.single_source_shortest_path_length 会自动忽略不可达的节点
        # 它返回一个字典: {target_node: distance} 这里的 distance 是跳数
        lengths = nx.single_source_shortest_path_length(G, start_node)
        
        # 如果这是一个孤立点（只能到自己，距离为0），跳过
        if not lengths:
            continue
            
        # 获取从当前 start_node 出发能走到的最远距离
        current_max = max(lengths.values())
        
        # 更新全局最大值
        if current_max > max_dist:
            max_dist = current_max
                

    return max_dist

# ==========================================
# 1. 字体配置 (必须在 plt.style.use 之后调用)
# ==========================================
def configure_chinese_font():
    """
    根据操作系统自动配置 Matplotlib 中文字体
    """
    
    
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']

        
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    

# ==========================================
# 2. 映射表读取逻辑
# ==========================================
def load_diameter_window_mapping(excel_path):
    """
    使用 pd.read_excel 读取并解析直径区间
    返回: [{'range': (lower, upper), 'w': window_size}, ...]
    """
    # 自动路径修正
    if not os.path.exists(excel_path):
        alternatives = [
            os.path.basename(excel_path), 
            os.path.join('constructtx', os.path.basename(excel_path)), # 你的代码中提到的目录
            os.path.join('data', os.path.basename(excel_path)),
            os.path.join('..', os.path.basename(excel_path))
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                print(f"[提示] 自动重定向文件路径至: {alt}")
                excel_path = alt
                break
    
    if not os.path.exists(excel_path):
        print(f"[严重错误] 找不到 Excel 文件: {excel_path}")
        return []

    mapping = []
    try:
        # === 核心修改：使用 read_excel ===
        df = pd.read_excel(excel_path)
        
        # 清洗列名，防止有空格
        df.columns = [str(c).strip() for c in df.columns]
        
        for _, row in df.iterrows():
            try:
                # 获取窗口大小 W
                w_val = int(row['推荐窗口大小 (W)'])
                
                # 获取直径区间字符串，例如 "(0, 2]"
                interval_str = str(row['直径区间 (D)'])
                
                # === 核心修改：参考你的正则解析 ===
                # 提取字符串中的所有数字（包括小数）
                # 注意：这里改用 regex 提取浮点数，以防区间是小数
                match = re.findall(r"(\d+(?:\.\d+)?)", interval_str)
                
                lower, upper = 0, 0
                
                if len(match) >= 2:
                    # 如果是 (0, 2]，则取两个数
                    lower = float(match[0])
                    upper = float(match[1])
                elif len(match) == 1:
                    # 如果只有一种边界，需判断是上限还是下限
                    # 简单判断：如果字符串里有 '>'，则为下限，上限无穷大
                    val = float(match[0])
                    if '>' in interval_str:
                        lower = val
                        upper = float('inf')
                    else:
                        # 默认作为上限
                        upper = val
                
                if upper > lower:
                    mapping.append({'range': (lower, upper), 'w': w_val})
                    
            except Exception:
                continue
                
        print(f"[成功] 已加载映射表，包含 {len(mapping)} 个区间规则")
        return mapping

    except Exception as e:
        print(f"[读取失败] Excel 读取出错: {e}")
        return []


def plot_diameter_trend_optimized(all_transactions_dict, 
                                  mapping_file='constructtx/直径窗口映射表_精简版.xlsx', 
                                  output_dir='experiment/'):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

    configure_chinese_font()

    # 调整画布比例，右侧留出更多空间给标签
    fig, ax = plt.subplots(figsize=(14, 8)) 
    
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', 'P']
    
    method_names = list(all_transactions_dict.keys())
    print(">>> 开始处理数据并绘图...")

    # --- 1. 绘制主数据折线 ---
    all_x = set()
    max_d_val = 0 
    
    for idx, method in enumerate(method_names):
        size_data_map = all_transactions_dict[method]
        sorted_sizes = sorted(size_data_map.keys())
        x_sizes = []
        y_diameters = []
        
        for size in sorted_sizes:
            tx_list = size_data_map[size]
            if not tx_list: continue
            try:
                # 请确保这些函数在您的上下文中可用
                btg = constuct_graph(tx_list)
                G_nx = btg.graph
                d = calculate_approx_directed_diameter(G_nx) if G_nx.number_of_nodes() > 0 else 0
            except:
                d = 0
            
            x_sizes.append(size)
            y_diameters.append(d)
            all_x.add(size)
            max_d_val = max(max_d_val, d)

        ax.plot(x_sizes, y_diameters, 
                 label=method,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 markersize=8, linewidth=2.5, alpha=0.9, zorder=10)

    # --- 2. 绘制右侧窗口映射 (优化版) ---
    mapping = load_diameter_window_mapping(mapping_file)
    
    if mapping:
        # 设置 Y 轴上限，留出一点余量
        y_limit_top = max_d_val * 1.15 if max_d_val > 0 else 10
        ax.set_ylim(0, y_limit_top)
        
        # 创建右侧双轴
        ax2 = ax.twinx()
        ax2.set_ylim(0, y_limit_top)
        ax2.set_ylabel('区块窗口大小', fontsize=13, color='#444', labelpad=55)
        
        # 过滤出当前视图范围内的区间
        visible_intervals = [m for m in mapping if m['range'][0] < y_limit_top]
        
        # === 关键优化：智能稀疏标注 ===
        # 计算最小标签间距（例如：总高度的 4%）
        min_label_gap = y_limit_top * 0.04 
        last_label_pos = -100 # 上一个绘制标签的中心位置
        
        for i, item in enumerate(visible_intervals):
            lower, upper = item['range']
            w = item['w']
            
            # 截断超出视野的部分
            draw_lower = max(0, lower)
            draw_upper = min(upper, y_limit_top)
            
            if draw_lower >= draw_upper: continue
            
            height = draw_upper - draw_lower
            mid_y = (draw_lower + draw_upper) / 2
            
            # A. 绘制背景分割线 (Grid Lines) - 始终绘制，表示精确界限
            ax.axhline(y=draw_upper, color='gray', linestyle=':', linewidth=0.5, alpha=0.5, zorder=0)
            
            # B. 绘制右侧色带 (Visual Bar)
            # 使用两种灰色交替，或者根据 W 大小渐变
            # 这里使用交替色让分界更明显
            bar_color = '#e0e0e0' if i % 2 == 0 else '#f5f5f5'
            
            # 在图表最右侧绘制一个矩形条
            rect = patches.Rectangle((1.0, draw_lower), 0.02, height, 
                                     transform=ax.get_yaxis_transform(), 
                                     facecolor=bar_color, edgecolor='none', clip_on=False)
            ax.add_patch(rect)
            
            # C. 智能绘制文字标签
            # 规则：只有当当前中心点与上一个标签距离足够远，且当前区间高度不是微乎其微时才绘制
            should_label = False
            
            # 情况1: 区间本身很高，必须标
            if height > min_label_gap * 0.8:
                should_label = True
            # 情况2: 距离上一个标签够远
            elif (mid_y - last_label_pos) > min_label_gap:
                should_label = True
                
            # 最后一个区间通常要标一下
            if i == len(visible_intervals) - 1 and (mid_y - last_label_pos) > min_label_gap/2:
                should_label = True

            if should_label:
                # 绘制标签
                ax2.text(1.03, mid_y, f"W={w}", 
                         transform=ax.get_yaxis_transform(),
                         va='center', ha='left', fontsize=9, color='#333', fontweight='bold')
                
                # 如果区间很宽，在色带中间加个小横杠指向文字
                if height > min_label_gap:
                    ax2.plot([1.0, 1.02], [mid_y, mid_y], transform=ax.get_yaxis_transform(), 
                             color='#666', linewidth=1)
                
                last_label_pos = mid_y # 更新位置记录

        ax2.set_yticks([]) # 隐藏右侧坐标轴原本的刻度
        
    else:
        print("[提示] 映射表为空，跳过背景绘制")

    # --- 3. 图表修饰 ---
    ax.set_title("交易图直径变化趋势与窗口阈值对照", fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("隐蔽消息大小", fontsize=13, fontweight='bold')
    ax.set_ylabel("交易图直径", fontsize=13, fontweight='bold')
    
    sorted_all_x = sorted(list(all_x))
    if sorted_all_x:
        ax.set_xticks(sorted_all_x)
        ax.set_xticklabels([str(s) for s in sorted_all_x], rotation=45, fontsize=10)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.3) # 仅保留X轴网格，Y轴已有分割线
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    # 调整布局，防止右侧标签被切掉
    plt.subplots_adjust(right=0.85)
    
    save_path = os.path.join(output_dir, 'diameter_trends_optimized.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[完成] 优化图表已保存至: {save_path}")
    plt.show()
    
    
def init_all_transactions(method_names, base_path='./data'):
    """
    初始化所有方法的交易数据。
    
    参数:
    method_names (list): 包含4个隐蔽方法名的列表，例如 ['GBCTD', 'Method2', 'Method3', 'Method4']
    base_path (str): 存放交易文件的文件夹路径
    
    返回:
    dict: 结构为 dict[方法名][消息大小] = 交易数据
    """
    
    # 初始化总字典
    all_data = {}

    print(f"开始初始化交易数据...")

    # 第一层循环：遍历 4 个方法
    for method in method_names:
        all_data[method] = {} # 为每个方法创建一个子字典
        print(f"正在读取方法组: {method}")
        
        # 第二层循环：遍历 10 个文件大小 (1024 * 1 到 1024 * 10)
        for i in range(1, 11):
            size = 1024 * i
            
            # 按照指定规则构建文件名：隐蔽方法名 + transactions + 隐蔽消息大小
            # 注意：如果你的文件有后缀（如 .csv），请在这里加上，例如：f"{...}.csv"
            file_name = f"{method}/{method}_transactions_{size}.json"
            
            # 拼接完整路径
            file_path = os.path.join(base_path, file_name)
            
            try:
                # 调用你提供的读取函数
                transactions = load_transactions_from_file(file_path)
                
                # 将数据存入字典，key 为消息大小 (int)
                all_data[method][size] = transactions
                
            except Exception as e:
                print(f"  [警告] 读取文件 {file_name} 失败: {e}")
                all_data[method][size] = None

    print("初始化完成。")
    return all_data

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    # print(">>> 正在加载正常交易数据...")
    # normal_tx = []
    # # 加载10个区块的数据
    # for i in range(923800, 923810):
    #     filename = f"dataset/transactions_block_{i}.json"
    #     try:
    #         file_transactions = load_transactions_from_file(filename)
    #         # 随机采样以控制图规模，确保实验速度
    #         if len(file_transactions) > 25:
    #             normal_tx.extend(random.sample(file_transactions, 25))
    #         else:
    #             normal_tx.extend(file_transactions)
    #     except FileNotFoundError:
    #         print(f"Warning: 文件 {filename} 未找到，跳过。")



    my_methods = ['DDSAC','BlockWhisper','GBCTD','GraphShadow']
    transaction_dict = init_all_transactions(my_methods, base_path='CompareMethod')
    # 运行对比
    plot_diameter_trend_optimized(transaction_dict, output_dir='experiment')
    