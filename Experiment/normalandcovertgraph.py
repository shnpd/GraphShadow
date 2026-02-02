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
        btg_mixed.add_transaction(ctx['hash'], ctx['inputs'], ctx['outputs'])
        # 记录 ID 用于后续高亮
        covert_tx_ids.add(ctx['hash'])
        
    print(f"注入完成。混合图最终规模: {btg_mixed.graph.number_of_nodes()} 节点")
    print(f"隐蔽交易占比: {len(covert_transaction)} / {len(normal_transaction) + len(covert_transaction)}")

    print(f"\n{'='*10} Step 3: 可视化 {'='*10}")
    
    # 5. 可视化绘制
    # ------------------------------------------------
    # 调用改进后的 visualize_covert 方法，传入隐蔽交易 ID 集合
    try:
        btg_mixed.visualize_covert(covert_tx_ids=covert_tx_ids)
    except AttributeError:
        print("❌ 错误: 你的 BitcoinTransactionGraph 类似乎没有更新 visualize_covert 方法。")
        print("请确保 visualize_covert 方法支持 `covert_tx_ids` 参数 (参考上一个问题的回答)。")
    except Exception as e:
        print(f"❌ 绘图时发生未知错误: {e}")


def generate_chain_transactions(n, rounds):
    """
    生成 n 条平行的链式交易结构。
    
    :param n: 初始地址个数 (即链的条数)
    :param rounds: 交易轮数 (链的长度)
    :return: 包含所有生成交易的列表
    """
    all_transactions = []
    
    # 1. 初始化 n 个起始地址 (假设这些地址初始就有 UTXO)
    # 这些是第 0 轮交易的输入
    current_addresses = [utils.generate_random_address() for _ in range(n)]
    
    print(f"初始化完成: {n} 条链，准备生成 {rounds} 轮交易...")

    # 2. 开始循环生成每一轮交易
    for r in range(rounds):
        next_round_addresses = [] # 用于存储本轮生成的输出，作为下一轮的输入
        
        # 每一轮生成 n 笔交易 (每条链 1 笔)
        for i in range(n):
            # 获取当前链的输入地址
            input_addr = current_addresses[i]
            
            # 生成新的接收地址 (输出地址)
            output_addr = utils.generate_random_address()
            
            # 生成交易 ID
            tx_id = utils.generate_tx_id()
            
            # 构造交易对象
            # 结构：1个输入 -> 1个输出
            tx = {
                'hash': tx_id,          # 或 'hash'，根据你图代码的习惯
                'inputs': [input_addr],
                'outputs': [output_addr]
            }
            
            all_transactions.append(tx)
            
            # 记录输出地址，供下一轮使用，保持链式结构不乱
            next_round_addresses.append(output_addr)
        
        # 更新 current_addresses，将光标移动到最新生成的地址上
        current_addresses = next_round_addresses

    print(f"生成结束。共生成 {len(all_transactions)} 笔交易。")
    return all_transactions

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

    covert_tx = generate_chain_transactions(n=5, rounds=8)


    build_mixed_graph_and_visualize(
        normal_transaction=normal_tx,
        covert_transaction = covert_tx,
    )

