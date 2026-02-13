import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import secrets
import math
from collections import defaultdict
import networkx as nx

from constructtx import utils
from txgraph.main import BitcoinTransactionGraph
from graphanalysis.sample_transaction import load_transactions_from_file

# ==========================================
# 1. 模型定义 (保持不变)
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
# 2. 论文核心算法实现: Graph Reconstruction
# ==========================================
class CovertTransactionConstructor:
    def __init__(self, model):
        self.model = model

    def generate_topology(self, num_nodes, target_input_count):
        """
        [逻辑核心]
        根据 target_input_count (需要的输入总数) 生成对应数量的边。
        因为 Algorithm 1 是合并过程，输入数量是守恒的。
        所以：初始生成的边数 == 最终交易的输入总数。
        """
        print(f"[*] Step 1: Generating Topology...")
        print(f"    - Address Pool Size (Nodes): {num_nodes}")
        print(f"    - Target Inputs (Edges): {target_input_count}")
        
        # 1. 初始化特征与邻接矩阵
        features = torch.randn(num_nodes, self.model.input_dim)
        adj_input = torch.eye(num_nodes)

        # 2. 模型推理
        self.model.eval()
        with torch.no_grad():
            z, _, _ = self.model(features, adj_input)
            adj_probs = torch.sigmoid(torch.matmul(z, z.t())).numpy()

        # 3. 移除对角线 (无自环)
        np.fill_diagonal(adj_probs, 0)

        # 4. 精确截取 Top-K 条边
        # K = target_input_count
        flat_probs = adj_probs.flatten()
        
        # 边界检查：全连接图的最大边数
        max_possible_edges = num_nodes * (num_nodes - 1)
        if target_input_count > max_possible_edges:
            print(f"[!] Warning: Target inputs {target_input_count} exceeds graph capacity {max_possible_edges}.")
            print(f"    - Auto-capping edges to {max_possible_edges}.")
            print(f"    - Hint: Increase 'pool_size' (num_nodes) to allow more inputs.")
            target_input_count = max_possible_edges

        # 找到概率阈值
        if target_input_count >= len(flat_probs):
            threshold = 0.0
        else:
            partition_idx = len(flat_probs) - target_input_count
            threshold = np.partition(flat_probs, partition_idx)[partition_idx]

        # 生成二值化邻接矩阵
        generated_adj = (adj_probs >= threshold).astype(int)
        
        # 提取边列表
        rows, cols = np.nonzero(generated_adj)
        edges = list(zip(rows, cols))

        # [关键] 再次精确截断，处理阈值边缘可能有多个相同概率值的情况
        if len(edges) > target_input_count:
            random.shuffle(edges) # 打乱以避免位置偏差
            edges = edges[:target_input_count]
            
        print(f"    -> Generated {len(edges)} edges (Initial Input Units).")
        return edges

    def algorithm_1_reconstruct(self, edge_units, node_labels, target_tx_count):
        """
        [Algorithm 1 - Count Preserved Version]
        修改点：使用 List 代替 Set，防止相同地址合并时被去重导致计数丢失。
        """
        # 1. 初始化: 使用列表存储地址
        S = []
        for u_idx, v_idx in edge_units:
            u_addr = node_labels[u_idx]
            v_addr = node_labels[v_idx]
            # [修改] 使用 list [] 而不是 set {}
            S.append({"inputs": [u_addr], "outputs": [v_addr]})

        print(f"[*] Step 2: Merging Transactions (Algorithm 1)...")
        print(f"    - Initial Transactions: {len(S)}")
        print(f"    - Target Transactions:  {target_tx_count}")
        
        max_failures = 5000
        failures = 0
        
        while len(S) > target_tx_count:
            if failures > max_failures:
                print(f"[!] Max failures reached. Stopped merging at {len(S)} transactions.")
                break
                
            idx_a, idx_b = random.sample(range(len(S)), 2)
            tx_a = S[idx_a]
            tx_b = S[idx_b]
            
            # [修改] 约束检查: 需要先转为 set 才能使用 isdisjoint
            # Check: A[In] ∩ B[Out] == Ø AND B[In] ∩ A[Out] == Ø
            inputs_a_set = set(tx_a['inputs'])
            outputs_a_set = set(tx_a['outputs'])
            inputs_b_set = set(tx_b['inputs'])
            outputs_b_set = set(tx_b['outputs'])
            
            cond1 = inputs_a_set.isdisjoint(outputs_b_set)
            cond2 = inputs_b_set.isdisjoint(outputs_a_set)
            
            if cond1 and cond2:
                # [修改] 使用列表拼接 (+)，保留所有元素，包括重复的地址
                new_inputs = tx_a['inputs'] + tx_b['inputs']
                new_outputs = tx_a['outputs'] + tx_b['outputs']
                
                # (可选) 长度限制检查
                if len(new_inputs) > 8 or len(new_outputs) > 8:
                    failures += 1
                    continue
                
                new_tx = {"inputs": new_inputs, "outputs": new_outputs}
                
                # 更新列表 S
                if idx_a > idx_b:
                    S.pop(idx_a); S.pop(idx_b)
                else:
                    S.pop(idx_b); S.pop(idx_a)
                
                S.append(new_tx)
                failures = 0
            else:
                failures += 1
                
        return S

    def construct(self, pool_size, total_inputs_needed, compression_ratio=0.6):
        """
        参数说明:
        pool_size: 对应 num_nodes，即参与交易的地址池大小。
        total_inputs_needed: 最终所有交易的输入总和 (由消息大小决定)。
        compression_ratio: 压缩率，决定了平均每笔交易包含几个 Input。
                           (例如 0.5 意味着平均每笔交易 2 个 Input)
        """
        # 1. 生成地址池 (Node Labels)
        # 这些地址仅仅是图上的节点标签
        node_labels = [utils.generate_random_address() for _ in range(pool_size)]
        # 2. 生成拓扑 (Edges)
        # 这里的 target_edge_count 直接等于 total_inputs_needed
        # 因为每个 Edge 最终都会成为一个 Input
        edge_units = self.generate_topology(
            num_nodes=pool_size, 
            target_input_count=total_inputs_needed
        )
        
        # 3. 计算目标交易数量 (Transaction Count)
        # Input总数不变，改变的是交易的"密度"
        target_tx_count = max(1, int(len(edge_units) * compression_ratio))
        
        # 4. 重构交易
        final_txs = self.algorithm_1_reconstruct(edge_units, node_labels, target_tx_count)
        
        # 5. [逻辑校验] 验证最终 Input 总数
        actual_total_inputs = sum(len(tx['inputs']) for tx in final_txs)
        print(f"\n[Verification]")
        print(f"    - Requested Inputs: {total_inputs_needed}")
        print(f"    - Actual Inputs:    {actual_total_inputs}")
        
        if actual_total_inputs != len(edge_units):
            print(f"    [!] Mismatch: Inputs lost during merging? (Should not happen)")
        else:
            print(f"    [OK] Input count conserved.")
            
        return final_txs


def save_transactions_to_json(transaction_list, filename="my_transactions.json"):
# ------------------ JSON Export ------------------
    formatted_txs = []
    for tx in transaction_list:
        tx_hash = secrets.token_hex(32)

        # 将 set 转回 list 以便 JSON 序列化
        clean_input_addrs = list(tx["inputs"])
        clean_output_addrs = list(tx["outputs"])

        tx_dict = {
            "hash": tx_hash,
            "input_addrs": clean_input_addrs,
            "output_addrs": clean_output_addrs,
        }

        formatted_txs.append(tx_dict)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(formatted_txs, f, indent=4)


def constuct_graph(tx_list):
    btg = BitcoinTransactionGraph()
    for tx in tx_list:
        btg.add_transaction(tx["hash"], tx["input_addrs"], tx["output_addrs"])
    return btg

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
        model = VGAE(input_dim=checkpoint.get("input_dim", 32))
        model.load_state_dict(checkpoint["state_dict"])

    if model is None:
        exit(1)

    # 构造初始地址池
    initial_utxos = [utils.generate_random_address() for _ in range(16)]  
    constructor = CovertTransactionConstructor(model)

    # 计算目标输入总数 (模拟消息大小)
    # for i in range (1,11):
    #     msg_size_B = i * 1024
    #     target_total_inputs = math.ceil(msg_size_B * 8 / 29) 
    #     print(f"Target Total Inputs needed: {target_total_inputs}")

    #     filename = f"CompareMethod/GBCTD/GBCTD_transactions_{msg_size_B}.json"
    #     tx_list = constructor.construct(
    #         pool_size=200,
    #         total_inputs_needed=target_total_inputs,
    #         compression_ratio=0.6
    #     )
    #     save_transactions_to_json(tx_list, filename)
    msg_size_B = 512
    target_total_inputs = math.ceil(msg_size_B * 8 / 29) 
    for i in range(1, 1001):  
        # 生成文件名
        filename = f"CompareMethod/GBCTD/dataset/GBCTD_transactions_{i}.json"
        # 执行构造
        tx_list = constructor.construct(
            pool_size=200,
            total_inputs_needed=target_total_inputs,
            compression_ratio=1
        )
        save_transactions_to_json(tx_list, filename)
        
        
       