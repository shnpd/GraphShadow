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

    def generate_topology(self, num_nodes, target_edge_count):
        """
        使用 VGAE 生成全局拓扑结构 (Adjacency Matrix)
        对应论文 Section IV.A/B: 使用生成模型学习并生成图结构
        """
        # 构造随机输入特征 (模拟论文中提到的 Random Matrix)
        features = torch.randn(num_nodes, self.model.input_dim)
        # 初始邻接矩阵 (可以是单位阵或全零，这里用单位阵作为基底)
        adj_input = torch.eye(num_nodes)
        
        self.model.eval()
        with torch.no_grad():
            z, _, _ = self.model(features, adj_input)
            # 重构邻接矩阵 A* = sigmoid(Z * Z^T)
            adj_probs = torch.sigmoid(torch.matmul(z, z.t())).numpy()
        
        # 移除对角线 (无自环)
        np.fill_diagonal(adj_probs, 0)
        
        # 为了确保生成足够的边以承载 input，我们需要选取概率最高的 Top-K 条边
        # K = target_edge_count
        flat_probs = adj_probs.flatten()
        # 获取第 K 大的概率值作为阈值
        if target_edge_count >= len(flat_probs):
            threshold = 0.0
        else:
            # 使用 partition 快速找到第 K 大的元素
            partition_idx = len(flat_probs) - target_edge_count
            threshold = np.partition(flat_probs, partition_idx)[partition_idx]
        
        # 生成二值化邻接矩阵
        generated_adj = (adj_probs >= threshold).astype(int)
        
        # 获取边列表 (source, target)
        rows, cols = np.nonzero(generated_adj)
        edges = list(zip(rows, cols))
        
        # 如果边数过多（因为阈值可能有重复值），截断到目标数量
        if len(edges) > target_edge_count:
            edges = edges[:target_edge_count]
            
        print(f"[*] VGAE Topology Generated: {num_nodes} Nodes, {len(edges)} Edges (Units)")
        return edges

    def algorithm_1_reconstruct_records(self, edge_units, node_labels, target_tx_count):
        """
        实现论文 Algorithm 1: Reconstruct Records From a Graph
        
        Require: 
            S: set of transaction record units (edges)
            R: target number of transactions (target_tx_count)
        Ensure:
            S: set of reconstructed transaction records
        """
        # 初始化 S: 将每一条边转换为一个独立的交易记录单元
        # Transaction Record Unit: {inputs: [u], outputs: [v]}
        S = []
        for u_idx, v_idx in edge_units:
            # 映射索引到真实的地址字符串
            u_addr = node_labels[u_idx]
            v_addr = node_labels[v_idx]
            S.append({
                "inputs": {u_addr},   # 使用 set 方便集合运算
                "outputs": {v_addr}
            })
        
        current_count = len(S)
        print(f"[*] Start merging... Initial Units: {current_count}, Target: {target_tx_count}")
        
        max_failures = 1000  # 防止死循环的保护机制
        failures = 0
        
        # while |S| > R do
        while len(S) > target_tx_count:
            if failures > max_failures:
                print(f"[!] Warning: Max failures reached. Stopped at {len(S)} transactions.")
                break
                
            # Select two records A and B from S randomly
            idx_a, idx_b = random.sample(range(len(S)), 2)
            tx_a = S[idx_a]
            tx_b = S[idx_b]
            
            # Constraint Check (论文原文逻辑):
            # if A[In] ∩ B[Out] == Ø AND B[In] ∩ A[Out] == Ø then
            # (确保地址不会在同一笔交易中既是输入又是输出，即资金回流)
            cond1 = tx_a['inputs'].isdisjoint(tx_b['outputs'])
            cond2 = tx_b['inputs'].isdisjoint(tx_a['outputs'])
            
            if cond1 and cond2:
                # Create C: Merge Inputs and Outputs
                new_inputs = tx_a['inputs'].union(tx_b['inputs'])
                new_outputs = tx_a['outputs'].union(tx_b['outputs'])
                
                # (可选：增加一个为了美观的限制，防止生成几百个输入的大交易)
                if len(new_inputs) > 5 or len(new_outputs) > 5:
                    failures += 1
                    continue

                new_tx = {
                    "inputs": new_inputs,
                    "outputs": new_outputs
                }
                
                # S = (S - A - B) U C
                # 由于是 list，我们移除索引较大的，再移除索引较小的，然后添加
                # 这样不会影响索引顺序
                if idx_a > idx_b:
                    S.pop(idx_a)
                    S.pop(idx_b)
                else:
                    S.pop(idx_b)
                    S.pop(idx_a)
                
                S.append(new_tx)
                failures = 0 # Reset failures on success
            else:
                failures += 1
                
        return S

    def construct_transactions(self, initial_utxos, total_input_slots_needed):
        """
        主流程
        1. 准备节点池 (Initial UTXOs + Generated Output Addresses)
        2. 生成图拓扑 (确定谁转给谁)
        3. 运行算法1进行合并
        """
        # 1. 确定节点总数
        # 我们需要足够的边来消耗 total_input_slots_needed (因为初始每个边消耗1个input slot)
        # 假设图比较稀疏，节点数 N 大约需要比边数 E 少一些或者相当。
        # 为了保证有足够的 unique output 地址，我们生成较多的节点。
        num_inputs = len(initial_utxos)
        # 预估需要的 Output 节点数。
        # 如果最终全是 2-in-2-out，则 Input 总数 ≈ Output 总数。
        # 为了安全起见，生成足够的接收地址池。
        num_generated_addrs = max(total_input_slots_needed, 50) 
        
        node_labels = initial_utxos.copy()
        for i in range(num_generated_addrs):
            node_labels.append(f"Rec_Addr_{i}")
            
        total_nodes = len(node_labels)
        
        # 2. 生成拓扑 (Edges)
        # 我们需要的边数 = 需要消耗的 Input 数量 (因为最开始每个 Input 对应一条边)
        # 注意：这里我们强制保留所有 initial_utxos 作为源节点的边。
        # 但 VGAE 是随机生成的，可能 initial_utxos 没有出边。
        # 策略：生成大量边，然后筛选出包含我们需要的。
        # 或者：简单地要求生成的边数为 target_total_inputs
        
        raw_edges = self.generate_topology(total_nodes, target_edge_count=total_input_slots_needed)
        
        # 3. 强制修正 (Ensure Initial UTXOs are used)
        # 论文方法是基于“现有图”重构。这里我们是“构造图”。
        # 我们需要确保 initial_utxos 在这些边中充当了 Source (u) 的角色。
        # 如果 VGAE 生成的边没有覆盖某些 initial_utxos，我们需要手动调整或添加。
        
        # 建立当前 Source 的集合
        current_sources = set([edge[0] for edge in raw_edges])
        
        # 找到前 len(initial_utxos) 个节点的索引
        initial_indices = list(range(len(initial_utxos)))
        
        final_edges = []
        used_edge_slots = 0
        
        # A. 优先保留 VGAE 生成的、且 Source 是初始地址的边 (保留真实分布特征)
        for u, v in raw_edges:
            if u in initial_indices:
                final_edges.append((u, v))
                used_edge_slots += 1
        
        # B. 补全未使用的初始地址 (如果 VGAE 没生成，强行连一条到随机 Output)
        for idx in initial_indices:
            # 检查该地址是否已经在 final_edges 中作为 Source 出现
            is_covered = False
            for u, v in final_edges:
                if u == idx:
                    is_covered = True
                    break
            
            if not is_covered:
                # 随机找一个 Output 节点 (索引 >= len(initial_utxos))
                target = random.randint(len(initial_utxos), total_nodes - 1)
                final_edges.append((idx, target))
                used_edge_slots += 1
                
        # C. 如果还需要更多 Input (达到 target_total_inputs)，从剩余的 VGAE 边中填充
        # (这代表中间节点的流转，即 A->B->C 中的 B->C)
        if used_edge_slots < total_input_slots_needed:
            for u, v in raw_edges:
                if used_edge_slots >= total_input_slots_needed:
                    break
                # 避免重复添加
                if (u, v) not in final_edges:
                    final_edges.append((u, v))
                    used_edge_slots += 1
                    
        print(f"[*] Final Edges Prepared: {len(final_edges)} (Ensured Initial UTXOs coverage)")

        # 4. 执行算法 1 (重构/合并)
        # 设定目标交易数 R。
        # 如果我们希望主要是 2-in-2-out，那么 R ≈ Total_Inputs / 2
        # 如果希望 1-in-2-out (Input=1, Output=2)，那么 R ≈ Total_Inputs
        # 论文中通常通过合并来混淆。这里设定一个混合目标，比如 60% 的输入数量
        target_R = int(len(final_edges) * 0.6) 
        
        reconstructed_txs = self.algorithm_1_reconstruct_records(
            final_edges, 
            node_labels, 
            target_tx_count=target_R
        )
        
        return reconstructed_txs

# ==========================================
# 5. Main Execution
# ==========================================
def main():
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

    # 构造初始地址池 (1Addr_1 ... 1Addr_16)
    initial_utxos = [f"1Addr_{i}" for i in range(1, 17)]
    
    constructor = CovertTransactionConstructor(model)
    
    # 计算目标输入总数 (模拟消息大小)
    msg_size_B = 4096
    target_total_inputs = math.ceil(msg_size_B * 8 / 30) 
    print(f"Target Total Inputs needed: {target_total_inputs}")
    
    # 执行构造
    tx_list = constructor.construct_transactions(
        initial_utxos=initial_utxos, 
        total_input_slots_needed=target_total_inputs
    )
    
    # ------------------ JSON Export ------------------
    print("\n" + "="*50)
    print("       Exporting JSON File (covert_transactions.json)       ")
    print("="*50)
    
    formatted_txs = []
    
    for tx in tx_list:
        tx_hash = secrets.token_hex(32)
        
        # 将 set 转回 list 以便 JSON 序列化
        clean_input_addrs = list(tx['inputs'])
        clean_output_addrs = list(tx['outputs'])
        
        tx_dict = {
            "hash": tx_hash,
            "input_addrs": clean_input_addrs,
            "output_addrs": clean_output_addrs
        }
        
        formatted_txs.append(tx_dict)
        
        # 简单打印预览前几个
        if len(formatted_txs) <= 5:
            print(f"Hash: {tx_hash[:8]}... | In({len(clean_input_addrs)}): {clean_input_addrs} -> Out({len(clean_output_addrs)})")

    output_filename = "CompareMethod/GBCTD/GBCTD_transactions.json"
    # 确保目录存在
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_txs, f, indent=4)
        
    print(f"\n[✓] Successfully saved {len(formatted_txs)} transactions to {output_filename}")
    
    # [修改点] 将 tx['inputs'] 改为 tx['input_addrs']
    print(f"[✓] Total Inputs Used: {sum(len(tx['input_addrs']) for tx in formatted_txs)}")

if __name__ == "__main__":
    main()