import json
import os
import pickle

from plotgraph.main import BitcoinTransactionGraph


# 从数据文件或API加载数据的函数
def load_transactions_from_file(file_path):
    """
    从文件加载交易
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        transactions = json.load(f)
    print(f"✓ 从 {file_path} 成功加载 {len(transactions)} 笔交易")
    return transactions


def construct_graph_from_block(startId, endId):
    # 获取原始交易数据
    all_transactions = []
    for i in range(startId, endId):
        filename = f"../dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        all_transactions.extend(file_transactions)
    # 将交易数据构图
    btg = BitcoinTransactionGraph()
    for tx in all_transactions:
        btg.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])
    return btg


# 文件缓存
def load_graph_cache(startId, endId, cache_path):
    if os.path.exists(cache_path):
        print(f"从缓存文件加载数据: {cache_path}")
        with open(cache_path, "rb") as f:
            btg = pickle.load(f)
    else:
        print("缓存不存在，开始重新采样...")
        btg = construct_graph_from_block(startId, endId)
        with open(cache_path, "wb") as f:
            pickle.dump(btg, f)
        print(f"采样结果已保存至: {cache_path}")
    return btg


if __name__ == "__main__":
    all_transactions = []
    # 统计10个区块
    for i in range(928050, 928060):
        filename = f"../dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        all_transactions.extend(file_transactions)
        # all_transactions.extend(random.sample(file_transactions, 10))
    with open("sampled_elements.json", "w") as f:
        json.dump(all_transactions, f)
