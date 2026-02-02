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
        btg_mixed.visualize_chain_covert_1in2out(covert_tx_ids=covert_tx_ids)
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
                'input_addrs': [input_addr],
                'output_addrs': [output_addr]
            }
            
            all_transactions.append(tx)
            
            # 记录输出地址，供下一轮使用，保持链式结构不乱
            next_round_addresses.append(output_addr)
        
        # 更新 current_addresses，将光标移动到最新生成的地址上
        current_addresses = next_round_addresses

    print(f"生成结束。共生成 {len(all_transactions)} 笔交易。")
    return all_transactions

def generate_1in2out_chain_transactions(n, rounds):
    """
    生成 n 条平行的 1输入-2输出 链式交易结构 (Peeling Chain)。
    
    结构逻辑：
    Tx1: [Addr_A] -> [Addr_Label_1, Addr_Change_1]
    Tx2: [Addr_Change_1] -> [Addr_Label_2, Addr_Change_2]
    ...
    
    :param n: 链的条数 (并行交易的数量)
    :param rounds: 链的深度 (交易轮数)
    :return: 包含所有生成交易的列表
    """
    all_transactions = []
    
    # 1. 初始化 n 个起始地址 (第 0 轮的输入)
    current_addresses = [utils.generate_random_address() for _ in range(n)]
    
    print(f"初始化完成: 准备生成 {n} 条平行链，每条链长度为 {rounds}...")

    # 2. 循环生成每一轮
    for r in range(rounds):
        next_round_addresses = [] # 仅存储将用于下一轮输入的地址（找零地址）
        
        for i in range(n):
            # --- 步骤 A: 确定输入 ---
            # 获取当前链的“头部”地址
            input_addr = current_addresses[i]
            
            # --- 步骤 B: 生成两个输出 ---
            # 输出 1: Label 地址 (或 Payload)。
            # 在 DDSAC 中，这个地址承载隐蔽信息，通常不再用于构建下一笔交易（或者被暂时搁置）。
            output_addr_label = utils.generate_random_address()
            
            # 输出 2: Change (找零) 地址。
            # 这是链条延续的关键，下一轮交易将花费这个地址里的钱。
            output_addr_change = utils.generate_random_address()
            
            # --- 步骤 C: 构造 1-in-2-out 交易 ---
            tx_id = utils.generate_tx_id()
            
            tx = {
                'hash': tx_id,
                'input_addrs': [input_addr],
                # 这里的顺序不影响链结构，但在DDSAC中通常混淆两者
                'output_addrs': [output_addr_label, output_addr_change] 
            }
            
            all_transactions.append(tx)
            
            # --- 步骤 D: 维护链式结构 (关键修改点) ---
            # 我们只把 "找零地址" 加入 next_round_addresses
            # 这样下一次循环时，这条链依然只有 1 个输入，保持单链形态，而不是分裂成树
            next_round_addresses.append(output_addr_change)
        
        # 更新 current_addresses，将光标移动到所有的“找零地址”上，准备下一轮
        current_addresses = next_round_addresses

    print(f"生成结束。共生成 {len(all_transactions)} 笔交易。")
    print(f"拓扑形态: {n} 条互相独立的平行链，每条链由 {rounds} 个 1-in-2-out 交易首尾相连组成。")
    
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

    # covert_tx = generate_chain_transactions(n=6, rounds=8)
    # covert_tx = load_transactions_from_file("experiment/covert_transactions.json")
    covert_tx = generate_1in2out_chain_transactions(n=6, rounds=11)
    build_mixed_graph_and_visualize(
        normal_transaction = normal_tx,
        covert_transaction = covert_tx,
    )

