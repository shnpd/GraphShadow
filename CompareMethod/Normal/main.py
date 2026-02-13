from graphanalysis.sample_transaction import load_transactions_from_file
import json
import random

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
    
    for i in range(1, 101): 
        output_filename = f"CompareMethod/Normal/dataset/Normal_transactions_{i}.json"
        filename = f"dataset/transactions_block_{928050+i}.json"
        file_transactions = load_transactions_from_file(filename)
        tx_list = random.sample(file_transactions, 100)
        save_transactions_to_json(tx_list, output_filename)