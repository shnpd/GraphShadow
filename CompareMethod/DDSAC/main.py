
import json
from constructtx import utils


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

def save_transactions_to_json(transaction_list, filename="my_transactions.json"):
    """
    直接将交易列表保存为 JSON 文件，不做额外处理
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transaction_list, f, indent=4)
        print(f"✓ 成功保存 {len(transaction_list)} 笔交易到文件: {filename}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")

        
if __name__ == "__main__":
    covert_tx = generate_1in2out_chain_transactions(n=6, rounds=11)
    save_transactions_to_json(covert_tx, "CompareMethod/DDSAC/DDSAC_transactions.json")

