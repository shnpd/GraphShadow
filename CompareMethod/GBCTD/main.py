import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import json
import os
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import secrets  # Used for generating random hashes
import math
# ==========================================
# 1. Data Loading & Graph Construction (Unchanged)
# ==========================================
class BlockDataLoader:
    def __init__(self, start_block, end_block):
        self.start_block = start_block
        self.end_block = end_block
        self.address_map = {} 
        self.edges = set()
        self.node_count = 0

    def load_data(self):
        def load_transactions_from_file(file_path):
            if not os.path.exists(file_path): return []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return []

        all_transactions = []
        print(f"[*] Loading data from block {self.start_block} to {self.end_block}...")
        for i in range(self.start_block, self.end_block + 1):
            filename = f"dataset/transactions_block_{i}.json"
            txs = load_transactions_from_file(filename)
            if txs:
                all_transactions.extend(random.sample(txs, min(len(txs), 200)))
        return all_transactions

    def build_graph(self, transactions):
        print("[*] Building transaction graph (Sparse Mode)...")
        self.address_map = {}
        self.edges = set()
        for tx in transactions:
            inputs = tx.get('input_addrs', [])
            outputs = tx.get('output_addrs', [])
            for in_addr in inputs:
                if in_addr not in self.address_map: self.address_map[in_addr] = len(self.address_map)
                for out_addr in outputs:
                    if out_addr not in self.address_map: self.address_map[out_addr] = len(self.address_map)
                    u, v = self.address_map[in_addr], self.address_map[out_addr]
                    if u != v: self.edges.add((u, v))
        
        self.node_count = len(self.address_map)
        if self.node_count == 0: raise ValueError("Error: No nodes in the graph.")

        pos_edge_index = torch.tensor(list(self.edges), dtype=torch.long).t()
        indices = pos_edge_index
        values = torch.ones(indices.shape[1])
        self_loop_index = torch.arange(0, self.node_count, dtype=torch.long).unsqueeze(0).repeat(2, 1)
        edge_index_hat = torch.cat([indices, self_loop_index], dim=1)
        edge_values_hat = torch.cat([values, torch.ones(self.node_count)])
        
        deg = torch.zeros(self.node_count)
        deg = deg.scatter_add_(0, edge_index_hat[0], edge_values_hat)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        row, col = edge_index_hat
        norm_values = deg_inv_sqrt[row] * edge_values_hat * deg_inv_sqrt[col]
        norm_adj = torch.sparse_coo_tensor(edge_index_hat, norm_values, (self.node_count, self.node_count))
        
        return norm_adj, pos_edge_index

# ==========================================
# 2. VGAE Model (Unchanged)
# ==========================================
class GCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
    def forward(self, x, adj):
        support = self.linear(x)
        return torch.spmm(adj, support)

class VGAE(nn.Module):
    def __init__(self, input_dim, hidden1=32, hidden2=16):
        super(VGAE, self).__init__()
        self.input_dim = input_dim
        self.gc1 = GCNLayer(input_dim, hidden1)
        self.gc_mu = GCNLayer(hidden1, hidden2)
        self.gc_logstd = GCNLayer(hidden1, hidden2)
    def encode(self, x, adj):
        hidden1 = F.relu(self.gc1(x, adj))
        return self.gc_mu(hidden1, adj), self.gc_logstd(hidden1, adj)
    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            return mu + torch.randn_like(std) * std
        return mu
    def forward(self, x, adj):
        mu, logstd = self.encode(x, adj)
        z = self.reparameterize(mu, logstd)
        return z, mu, logstd

# ==========================================
# 3. Loss Function (Unchanged)
# ==========================================
def negative_sampling(pos_edge_index, num_nodes):
    return torch.randint(0, num_nodes, (2, pos_edge_index.shape[1]), dtype=torch.long)

def get_link_prediction_loss(z, pos_edge_index, neg_edge_index, mu, logstd, num_nodes):
    pos_score = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    recon_loss = F.binary_cross_entropy_with_logits(scores, labels)
    kl = -0.5 / num_nodes * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))
    return recon_loss + kl

# ==========================================
# 4. Covert Transaction Construction (Core Logic)
# ==========================================
class CovertTransactionConstructor:
    def __init__(self, model):
        self.model = model

    def run_algorithm_1(self, records, target_tx_count=2, max_outputs_per_tx=3):
        attempts = 0
        max_attempts = 2000
        
        while len(records) > target_tx_count and attempts < max_attempts:
            if len(records) < 2: break
            
            idx_a, idx_b = random.sample(range(len(records)), 2)
            tx_a, tx_b = records[idx_a], records[idx_b]
            
            # Conflict check
            inputs_a_addrs = {item[0] for item in tx_a['inputs']}
            inputs_b_addrs = {item[0] for item in tx_b['inputs']}
            
            cond_conflict = inputs_a_addrs.isdisjoint(tx_b['outputs']) and \
                            inputs_b_addrs.isdisjoint(tx_a['outputs'])
            
            potential_outputs = tx_a['outputs'].union(tx_b['outputs'])
            # STRICT constraint: max outputs per merged tx
            cond_size = len(potential_outputs) <= max_outputs_per_tx
            
            if cond_conflict and cond_size:
                new_tx = {
                    "inputs": tx_a['inputs'] + tx_b['inputs'], 
                    "outputs": potential_outputs
                }
                for idx in sorted([idx_a, idx_b], reverse=True):
                    records.pop(idx)
                records.append(new_tx)
                attempts = 0
            else:
                attempts += 1
        return records

    def construct_single_round(self, current_inputs, round_id, max_degree):
            num_utxos = len(current_inputs)
            
            # [优化1] 限制新地址池的大小
            num_new_outputs = max(2, int(num_utxos * 1.2)) 
            
            total_nodes = num_utxos + num_new_outputs
            new_outputs = [f"Rec_R{round_id}_{i:02d}" for i in range(num_new_outputs)]
            
            # ... (中间的 图生成、智能映射、剪枝 代码完全保持不变) ...
            # ... 从 "1. 生成拓扑" 到 "4. 生成初始记录" 的代码都不用动 ...
            
            # 1. 生成拓扑
            features = torch.randn(total_nodes, self.model.input_dim)
            adj_input = torch.eye(total_nodes)
            self.model.eval()
            with torch.no_grad():
                z, _, _ = self.model(features, adj_input)
                adj_probs = torch.sigmoid(torch.matmul(z, z.t())).numpy()
            
            generated_adj = (adj_probs > 0.55).astype(float)
            np.fill_diagonal(generated_adj, 0)
            
            # 2. 智能映射
            out_degrees = np.sum(generated_adj, axis=1)
            sorted_indices = np.argsort(out_degrees)[::-1]
            sender_node_indices = sorted_indices[:num_utxos]
            
            node_map = {} 
            assigned_nodes = set()
            
            for i, node_idx in enumerate(sender_node_indices):
                addr = current_inputs[i]
                unique_id = (addr, f"R{round_id}_{i}") 
                node_map[node_idx] = {"type": "input", "val": unique_id}
                assigned_nodes.add(node_idx)
                
            rec_idx = 0
            for i in range(total_nodes):
                if i not in assigned_nodes:
                    node_map[i] = {"type": "output", "val": new_outputs[rec_idx]}
                    rec_idx += 1

            # 3. 剪枝与转换
            temp_connections = defaultdict(list)
            rows, cols = np.nonzero(generated_adj)
            
            for r, c in zip(rows, cols):
                sender_info = node_map[r]
                receiver_info = node_map[c]
                if sender_info["type"] == "input" and receiver_info["type"] == "output":
                    temp_connections[r].append(receiver_info["val"])
            
            # 兜底
            for node_id, info in node_map.items():
                if info["type"] == "input":
                    if node_id not in temp_connections or len(temp_connections[node_id]) == 0:
                        temp_connections[node_id].append(random.choice(new_outputs))

            # 4. 生成初始记录
            initial_records = []
            for node_id, receivers in temp_connections.items():
                # [关键修改2] 更加倾向于 1-out
                # 将 1-out 的概率提升到 90%。
                # 只有初始是 1-out，两个合并起来才容易是 2-out (必然满足 <=3)
                limit = 1 if random.random() < 0.82 else 2
                
                if not receivers: receivers = [random.choice(new_outputs)]
                
                # 优先选择已经被选过的 output (热点机制)，增加重叠率
                # 这是一个简单的小技巧：如果列表里有多个，只取第一个（通常是权重最高的）
                selected_receivers = receivers[:limit]
                
                utxo_tuple = node_map[node_id]["val"] 
                initial_records.append({
                    "inputs": [utxo_tuple], 
                    "outputs": set(selected_receivers)
                })

            # 5. 平衡合并策略
            target_tx = max(1, int(num_utxos * 0.5)) # 目标是两两合并
            
            # [关键修改3] 稍微放宽单笔交易输出上限到 4
            # 允许 2-in-4 (最坏情况) 发生，防止合并被完全卡死
            final_txs = self.run_algorithm_1(initial_records, 
                                        target_tx_count=target_tx, 
                                        max_outputs_per_tx=4) 
            
            next_round_inputs = []
            for tx in final_txs:
                next_round_inputs.extend(list(tx['outputs']))
                
            return final_txs, next_round_inputs
    def construct_multi_round_chain(self, initial_inputs, target_total_inputs):
        """ Multi-round construction """
        print("\n" + "="*50)
        print(f">>> Starting multi-round covert chain construction (Target: Consume {target_total_inputs} UTXOs)")
        print("="*50)
        
        all_rounds_history = []
        current_utxos = initial_inputs
        total_inputs_consumed = 0
        round_count = 1
        
        while total_inputs_consumed < target_total_inputs:
            if not current_utxos: break
                
            print(f"[*] Round {round_count}: Processing {len(current_utxos)} inputs...")
            # max_degree=2 encourages 1-input-2-output base transactions
            txs, next_utxos = self.construct_single_round(current_utxos, round_count, max_degree=2)
            
            actual_consumed = sum([len(tx['inputs']) for tx in txs])
            
            all_rounds_history.extend(txs) 
            
            total_inputs_consumed += actual_consumed
            current_utxos = next_utxos
            round_count += 1
            
                
        return all_rounds_history


if __name__ == "__main__":
    TRAIN_MODE = False  
    MODEL_PATH = "CompareMethod/GBCTD/vgae_model_checkpoint.pth"
    
    # ------------------ Model Loading ------------------
    model = None
    if TRAIN_MODE:
        pass 
    else:
        if not os.path.exists(MODEL_PATH):
            print(f"Model not found: {MODEL_PATH}")
            exit(1)
        checkpoint = torch.load(MODEL_PATH)
        model = VGAE(input_dim=checkpoint.get('input_dim', 32))
        model.load_state_dict(checkpoint['state_dict'])

    if model is None: exit(1)

    initial_utxos = [f"1Addr_{i}" for i in range(1, 17)]
    constructor = CovertTransactionConstructor(model)
    # 传输消息大小
    msg_size_B = 4096
    target_total_inputs = math.ceil((msg_size_B * 8) / 128)
    print(f"需要构造{target_total_inputs}个输入")
    # Execute construction
    flat_tx_list = constructor.construct_multi_round_chain(
        initial_inputs=initial_utxos, 
        target_total_inputs=target_total_inputs
    )
    
    # ------------------ JSON Export ------------------
    print("\n" + "="*50)
    print("       Exporting JSON File (covert_transactions.json)       ")
    print("="*50)
    
    formatted_txs = []
    
    for tx in flat_tx_list:
        # 1. Generate random hash
        tx_hash = secrets.token_hex(32)
        
        # 2. Clean Inputs
        raw_inputs = tx['inputs']
        clean_input_addrs = [item[0] for item in raw_inputs]
        
        # 3. Clean Outputs
        clean_output_addrs = list(tx['outputs'])
        
        # 4. Construct dictionary
        tx_dict = {
            "hash": tx_hash,
            "input_addrs": clean_input_addrs,
            "output_addrs": clean_output_addrs
        }
        
        formatted_txs.append(tx_dict)
        
        # Print preview
        print(f"Hash: {tx_hash[:8]}... | In: {clean_input_addrs} -> Out: {clean_output_addrs}")

    # Save file
    output_filename = "CompareMethod/GBCTD/GBCTD_transactions.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_txs, f, indent=4)
        
    print(f"\n[✓] Successfully saved {len(formatted_txs)} transactions to {output_filename}")