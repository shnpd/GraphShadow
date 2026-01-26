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


def process_single_sample(sample_id, start_height, max_window=10):
    """
    处理单个起始区块的样本：增量计算 1~10 个区块的直径
    """
    print(f"[线程 {sample_id}] 开始分析起始区块: {start_height}")
    sample_results = {}
    btg = BitcoinTransactionGraph()
    try:
        # 逐步增加窗口大小 (1 -> 10) w为窗口大小
        for w in range(1, max_window + 1):
            current_block_height = start_height + w - 1
            # 爬取对应区块交易
            save_block_transaction(current_block_height, current_block_height + 1)
            # 2. 加载当前这一个新区块的数据
            file_path = f"../dataset/transactions_block_{current_block_height}.json"
            file_transactions = load_transactions_from_file(file_path)
            for tx in file_transactions:
                btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
            pass
            # 3. 计算当前图的距离分布列表
            dist_list = get_max_diameter(btg)
            # 取列表的 90 分位
            dist_list.sort()
            idx = int(len(dist_list) * 0.9)
            effective_diameter = dist_list[idx]
            sample_results[w] = effective_diameter
    except Exception as e:
        print(f"  [线程 {sample_id}] 发生错误: {e}")
        return None
    print(f"[线程 {sample_id}] 完成。")
    # 返回窗口与直径的映射
    return sample_results


def main_statistics():
    # 参数配置
    MIN_HEIGHT = 923821
    MAX_HEIGHT = 933821
    SAMPLE_SIZE = 5
    MAX_WINDOW = 2
    MAX_WORKERS = 2  # 线程数，建议设为 CPU 核心数或稍大 (如果是 IO 密集型)

    # 1. 随机采样起始区块
    valid_range = range(MIN_HEIGHT, MAX_HEIGHT - MAX_WINDOW)
    start_heights = sorted(random.sample(valid_range, SAMPLE_SIZE))
    print(f"选定的 {SAMPLE_SIZE} 个起始区块: {start_heights}")
    # 2. 多线程执行分析
    results_collection = defaultdict(list)  # { window_size: [d1, d2, ... d30] }
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        # future_to_height = { Future : h }，建立 Future → 起始区块高度 的映射
        future_to_height = {
            # executor.submit()立即返回一个 Future 对象，该对象代表“未来会完成的计算结果”
            executor.submit(process_single_sample, i, h, MAX_WINDOW): h
            for i, h in enumerate(start_heights)
        }
        # 获取结果
        for future in concurrent.futures.as_completed(future_to_height):
            # 取回任务的起始区块高度
            h = future_to_height[future]
            try:
                # 获取任务执行结果
                res = future.result()
                if res:
                    for w, diameter in res.items():
                        results_collection[w].append(diameter)
            except Exception as exc:
                print(f"起始区块 {h} 生成异常: {exc}")
    # 返回的是窗口大小对应图直径
    return results_collection

if __name__ == "__main__":
    random.seed(42)
    res = main_statistics()
    print(res)
    # cnt = get_diameter_from_block(927000, 10)
    # cnt = sorted(cnt)
    # idx = int(round(len(cnt) * 0.9, 10))
    # print(cnt[idx])
