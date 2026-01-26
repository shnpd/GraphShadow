import json

from crawler.crawler import save_block_transaction
from graphanalysis.path_length import get_max_diameter
from graphanalysis.sample_transaction import load_transactions_from_file, load_graph_cache
from txgraph.main import BitcoinTransactionGraph


def get_diameter_from_block(height, count):
    # 爬取对应区块交易
    save_block_transaction(height, height + count)
    # 区块交易构图
    btg = load_graph_cache(height,height+count,f"cache/graph_cache_{height}to{height+count}.pkl")
    # 计算直径
    return get_max_diameter(btg)

if __name__ == "__main__":
    cnt = get_diameter_from_block(927000,1)
    cnt = sorted(cnt)
    print(cnt)