import concurrent
import json
import os
import random
from collections import defaultdict

from crawler.crawler import save_block_transaction
# from crawler.crawler import save_block_transaction
from graphanalysis.path_length import get_node_distance_list
from graphanalysis.sample_transaction import load_transactions_from_file, load_graph_cache
from txgraph.main import BitcoinTransactionGraph


def append_distance_result(file_path, start_height, window_size, dist_list):
    """
    将统计结果追加保存到 JSONL 文件中。

    :param file_path: 保存的文件路径 (建议以 start_height 命名，或者汇总到一个大文件)
    :param start_height: 起始区块高度
    :param window_size: 当前窗口大小 (w)
    :param dist_list: 计算出的距离列表
    """
    # 构造一条记录
    record = {
        "key": f"{start_height}:{window_size}",  # 你的需求：key 为当前结束的区块高度
        "start_height": start_height,  # 记录起始高度，方便分组
        "window_size": window_size,  # 记录窗口大小
        "sample_count": len(dist_list),  # 记录样本数量(列表长度)，方便快速概览
        "dist_list": dist_list  # 核心数据
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 使用 'a' (append) 模式追加写入
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"追加文件：{file_path}")

def process_single_sample(start_height, max_window=10):
    """
    处理单个起始区块的样本：增量计算 1~10 个区块的直径
    """
    print(f"开始分析起始区块: {start_height}")
    sample_results = {}
    btg = BitcoinTransactionGraph()
    try:
        # 逐步增加窗口大小 (1 -> 10) w为窗口大小
        for w in range(1, max_window + 1):
            current_block_height = start_height + w - 1
            # 1. 爬取对应区块交易
            save_block_transaction(current_block_height, current_block_height + 1)
            # 2. 加载当前这一个新区块的数据
            file_path = f"../dataset/transactions_block_{current_block_height}.json"
            file_transactions = load_transactions_from_file(file_path)
            # 3. 增量构图
            print("-------------------开始增量构图-------------------")
            for tx in file_transactions:
                btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
            print("-------------------结束增量构图-------------------")

            # 3. 计算当前图的距离分布列表
            dist_list = get_node_distance_list(btg)
            # 取列表的 90 分位
            dist_list.sort()
            idx = int(len(dist_list) * 0.9)
            effective_diameter = dist_list[idx]
            sample_results[w] = effective_diameter
            # 保存完整列表结果
            save_file = f"diameter/dist_stats_start_{start_height}.jsonl"
            append_distance_result(save_file, start_height, w, dist_list)
            print(f"起始区块: {start_height} ，窗口：{w}，分析完成：{sample_results}。")
    except Exception as e:
        print(f"起始区块: {start_height} 分析发生错误: {e}")
        return None
    print(f"起始区块: {start_height} 分析完成。")  # 返回窗口与直径的映射
    print(sample_results)
    return sample_results


def main_statistics():
    # 参数配置
    MIN_HEIGHT = 923821
    MAX_HEIGHT = 933821
    SAMPLE_SIZE = 15
    MAX_WINDOW = 20
    # 1. 随机采样起始区块
    # valid_range = range(MIN_HEIGHT, MAX_HEIGHT - MAX_WINDOW)
    # start_heights = sorted(random.sample(valid_range, SAMPLE_SIZE))
    # print(f"选定的 {SAMPLE_SIZE} 个起始区块: {start_heights}")

    # mock start_heights
    start_heights = list(range(923800, 933800, 20))

    # 2. 遍历起始区块进行统计
    results_collection = defaultdict(list)  # { window_size: [d1, d2, ... d30] }
    for i, h in enumerate(start_heights):
        try:
            # 直接调用处理函数
            res = process_single_sample(h, MAX_WINDOW)
            # 收集结果
            if res:
                for w, diameter in res.items():
                    results_collection[w].append(diameter)
        except Exception as exc:
            print(f"起始区块 {h} 处理异常: {exc}")
            continue
    # 返回的是窗口大小对应图直径列表的字典
    return results_collection


if __name__ == "__main__":
    random.seed(42)
    res = main_statistics()
    print(res)
    # cnt = get_diameter_from_block(927000, 10)
    # cnt = sorted(cnt)
    # idx = int(round(len(cnt) * 0.9, 10))
    # print(cnt[idx])
