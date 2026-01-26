import concurrent
import json
import random
from collections import defaultdict

from crawler.crawler import save_block_transaction
from graphanalysis.path_length import get_max_diameter
from graphanalysis.sample_transaction import load_transactions_from_file, load_graph_cache
from txgraph.main import BitcoinTransactionGraph


# def get_diameter_from_block(height, count):
#     # 爬取对应区块交易
#     save_block_transaction(height, height + count)
#     # 区块交易构图
#     btg = load_graph_cache(height, height + count, f"cache/graph_cache_{height}to{height + count}.pkl")
#     # 计算直径
#     return get_max_diameter(btg)


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
            for tx in file_transactions:
                btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
            # 3. 计算当前图的距离分布列表
            dist_list = get_max_diameter(btg)
            # 取列表的 90 分位
            dist_list.sort()
            idx = int(len(dist_list) * 0.9)
            effective_diameter = dist_list[idx]
            sample_results[w] = effective_diameter
    except Exception as e:
        print(f"起始区块: {start_height} 分析发生错误: {e}")
        return None
    print(f"起始区块: {start_height} 分析完成。")  # 返回窗口与直径的映射
    return sample_results


def main_statistics():
    # 参数配置
    MIN_HEIGHT = 923821
    MAX_HEIGHT = 933821
    SAMPLE_SIZE = 15
    MAX_WINDOW = 10
    # 1. 随机采样起始区块
    valid_range = range(MIN_HEIGHT, MAX_HEIGHT - MAX_WINDOW)
    start_heights = sorted(random.sample(valid_range, SAMPLE_SIZE))
    print(f"选定的 {SAMPLE_SIZE} 个起始区块: {start_heights}")

    # mock start_heights
    start_heights = [923800, 923810, 923820]

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
