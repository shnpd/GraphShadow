import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import os
import json
from tqdm import tqdm
from torch.optim import Adam

# 复用你提供的模型定义
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
def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix: D^{-1/2} * (A + I) * D^{-1/2}
    修复版：真正执行归一化乘法，防止度数大的节点导致数值爆炸。
    """
    # 1. 添加自环: A_tilde = A + I
    num_nodes = adj.size(0)
    adj_eye = torch.eye(num_nodes).to_sparse()
    adj_tilde = (adj + adj_eye).coalesce()
    
    indices = adj_tilde.indices()
    values = adj_tilde.values()
    
    # 2. 计算度: D_tilde_ii = sum_j(A_tilde_ij)
    row_sum = torch.zeros(num_nodes)
    row_sum.index_add_(0, indices[0], values)
    
    # 3. 计算 D^(-1/2)
    # 防止除以0，加上一个极小量或者处理Inf
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    
    # 4. 执行归一化: val_new = val * d_inv_sqrt[row] * d_inv_sqrt[col]
    # indices[0] 是行索引, indices[1] 是列索引
    values_norm = values * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
    
    # 5. 返回归一化后的稀疏矩阵
    return torch.sparse_coo_tensor(indices, values_norm, adj.shape)

def loss_function(preds, labels, mu, logstd, n_nodes, norm, pos_weight):
    # 数值保护：pos_weight
    if torch.isnan(pos_weight) or torch.isinf(pos_weight):
        pos_weight = torch.tensor(1.0)

    # 1. 重构损失
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    
    # 2. KL 散度数值保护 (从 max=10 降为 max=6)
    # exp(2*6) = exp(12) ≈ 162754，足够覆盖合理范围且不易溢出
    logstd = torch.clamp(logstd, max=6)
    
    # KL 公式
    kl = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))
    
    return cost + kl
# ==========================================
# 4. 数据预处理与图构建
# ==========================================
def load_and_build_graph(max_nodes=10000):
    """
    从JSON交易数据中构建地址交互图 (Address Interaction Graph)
    """
    
    tx_list = []
    for i in tqdm(range(10)):
        data_path = f"dataset/transactions_block_{923800 + i}.json"
        with open(data_path, 'r') as f:
            txs = json.load(f)
        tx_list.extend(txs)
        
    # 1. 映射地址到索引
    addr_to_idx = {}
    edges = []
    
    # 限制只读取部分交易以构建图，防止内存爆炸
    for tx in tx_list: 
        inputs = tx.get('inputs', []) or tx.get('input_addrs', [])
        outputs = tx.get('outputs', []) or tx.get('output_addrs', [])
        
        clean_inputs = []
        for inp in inputs:
            if isinstance(inp, str): clean_inputs.append(inp)
            elif isinstance(inp, dict): clean_inputs.extend(inp.get('addresses', []))
            
        clean_outputs = []
        for out in outputs:
            if isinstance(out, str): clean_outputs.append(out)
            elif isinstance(out, dict): clean_outputs.extend(out.get('addresses', []))

        for u in clean_inputs:
            if u not in addr_to_idx:
                if len(addr_to_idx) >= max_nodes: continue
                addr_to_idx[u] = len(addr_to_idx)
            for v in clean_outputs:
                if v not in addr_to_idx:
                    if len(addr_to_idx) >= max_nodes: continue
                    addr_to_idx[v] = len(addr_to_idx)
                if u != v:
                    edges.append((addr_to_idx[u], addr_to_idx[v]))

    num_nodes = len(addr_to_idx)
    print(f"    -> Graph built: {num_nodes} nodes, {len(edges)} edges.")

    if num_nodes == 0:
        raise ValueError("No nodes found in dataset.")

    # Build Sparse Adjacency
    if len(edges) > 0:
        adj_indices = torch.tensor(edges).t()
        adj_values = torch.ones(len(edges))
        adj = torch.sparse_coo_tensor(adj_indices, adj_values, (num_nodes, num_nodes))
    else:
        adj = torch.sparse_coo_tensor(torch.empty(2,0), torch.empty(0), (num_nodes, num_nodes))

    features = torch.randn(num_nodes, 32)
    
    # Preprocessing
    adj_dense = adj.to_dense()
    pos_sum = adj_dense.sum()
    neg_sum = num_nodes * num_nodes - pos_sum
    
    pos_weight = neg_sum / (pos_sum + 1e-9)
    norm = (num_nodes * num_nodes) / ((neg_sum + 1e-9) * 2)
    
    # Normalize Adj (Add Self Loops)
    adj_norm = normalize_adj(adj)

    return adj_norm, features, adj_dense, norm, pos_weight
# ==========================================
# 训练主程序
# ==========================================
def train_model():
    SAVE_PATH = "CompareMethod/GBCTD/vgae_model_checkpoint.pth"
    
    EPOCHS = 200
    LR = 0.01

    data = load_and_build_graph()
    if data is None: return
    adj, features, adj_label, norm, pos_weight = data
    
    model = VGAE(input_dim=32)
    optimizer = Adam(model.parameters(), lr=LR)

    print(f"[*] Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        z, mu, logstd = model(features, adj)
        preds = torch.matmul(z, z.t())
        
        loss = loss_function(preds, adj_label, mu, logstd, features.shape[0], norm, pos_weight)
        
        if torch.isnan(loss):
            print(f"[!] Loss is NaN at epoch {epoch}. Stopping.")
            break
            
        loss.backward()
        
        # [关键] 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

    torch.save({'state_dict': model.state_dict(), 'input_dim': 32}, SAVE_PATH)
    print(f"[*] Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train_model()