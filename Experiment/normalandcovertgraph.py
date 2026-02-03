import sys
import os
import random
from graphanalysis.sample_transaction import load_transactions_from_file, load_graph_cache
from constructtx.constructTxSplitG_v3 import generate_covert_transactions
from txgraph.main import BitcoinTransactionGraph
import constructtx.utils as utils

def build_mixed_graph_and_visualize(normal_transaction, covert_transaction):
    """
    构造混合交易图并可视化。
    """
    # 1. 初始化图对象
    # ------------------------------------------------
    # 注意：这里使用的是你的 BitcoinTransactionGraph 类
    # 确保该类已经更新了支持 covert_tx_ids 参数的 visualize 方法
    btg_mixed = BitcoinTransactionGraph()
    
    print(f"\n{'='*10} Step 1: 加载正常交易 {'='*10}")
    
    # 2. 加载并添加正常交易
    # ------------------------------------------------
    for tx in normal_transaction:
        btg_mixed.add_transaction(tx['hash'], tx['input_addrs'], tx['output_addrs'])

    covert_tx_ids = set()
    for ctx in covert_transaction:
        btg_mixed.add_transaction(ctx['hash'], ctx['input_addrs'], ctx['output_addrs'])
        # 记录 ID 用于后续高亮
        covert_tx_ids.add(ctx['hash'])
        
    print(f"注入完成。混合图最终规模: {btg_mixed.graph.number_of_nodes()} 节点")
    print(f"隐蔽交易占比: {len(covert_transaction)} / {len(normal_transaction) + len(covert_transaction)}")

    print(f"\n{'='*10} Step 3: 可视化 {'='*10}")
    
    # 5. 可视化绘制
    # ------------------------------------------------
    # 调用改进后的 visualize_covert 方法，传入隐蔽交易 ID 集合
    try:
        btg_mixed.visualize_blockwhisper_covert(covert_tx_ids=covert_tx_ids)
    except AttributeError:
        print("❌ 错误: 你的 BitcoinTransactionGraph 类似乎没有更新 visualize_covert 方法。")
        print("请确保 visualize_covert 方法支持 `covert_tx_ids` 参数 (参考上一个问题的回答)。")
    except Exception as e:
        print(f"❌ 绘图时发生未知错误: {e}")




if __name__ == "__main__":
    
    # 统计10个区块的正常交易
    normal_tx = []
    for i in range(928050, 928060):
        filename = f"dataset/transactions_block_{i}.json"
        file_transactions = load_transactions_from_file(filename)
        normal_tx.extend(random.sample(file_transactions, 100))


    # 生成隐蔽交易
    # covert_tx, _ = generate_covert_transactions(
    #     message_size_B=1024, 
    #     num_groups=5 # 使用之前的分组策略
    # )

    # covert_tx = generate_chain_transactions(n=6, rounds=8)
    # covert_tx = load_transactions_from_file("experiment/covert_transactions.json")
    covert_tx = load_transactions_from_file("CompareMethod/BlockWhisper/BlockWhisper_transactions.json")

    build_mixed_graph_and_visualize(
        normal_transaction = normal_tx,
        covert_transaction = covert_tx,
    )

