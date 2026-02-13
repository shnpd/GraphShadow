import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
import networkx as nx
import json
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import gc
import math

# ==========================================
# 1. 图处理核心 (纯拓扑 - One-Hot Degree)
# ==========================================
class GraphProcessor:
    @staticmethod
    def _add_tx_to_graph(G, tx):
        tx_hash = tx.get('hash', 'unknown')
        G.add_node(tx_hash, node_type='transaction')
        
        inputs = []
        inp_field = tx.get('inputs', []) or tx.get('input_addrs', [])
        for inp in inp_field:
            if isinstance(inp, str): inputs.append(inp)
            elif isinstance(inp, dict): inputs.extend(inp.get('addresses', []))
        for addr in inputs:
            G.add_node(addr, node_type='address')
            G.add_edge(addr, tx_hash, direction='input')
            
        outputs = []
        out_field = tx.get('outputs', []) or tx.get('output_addrs', [])
        for out in out_field:
            if isinstance(out, str): outputs.append(out)
            elif isinstance(out, dict): outputs.extend(out.get('addresses', []))
        for addr in outputs:
            G.add_node(addr, node_type='address')
            G.add_edge(tx_hash, addr, direction='output')

    @staticmethod
    def convert_subgraph_to_heterodata(sub_G, label):
        data = HeteroData()
        tx_nodes = [n for n, d in sub_G.nodes(data=True) if d.get('node_type') == 'transaction']
        addr_nodes = [n for n, d in sub_G.nodes(data=True) if d.get('node_type') == 'address']
        
        if len(tx_nodes) == 0: return None
        
        # === [核心特征] One-Hot Degree (纯拓扑结构) ===
        def get_degree_one_hot(nodes, max_degree=64):
            degrees = [sub_G.degree(n) for n in nodes]
            deg_tensor = torch.tensor(degrees, dtype=torch.long).clamp(max=max_degree-1)
            return F.one_hot(deg_tensor, num_classes=max_degree).float()

        data['transaction'].x = get_degree_one_hot(tx_nodes)
        data['address'].x = get_degree_one_hot(addr_nodes)
        
        tx_mapping = {n: i for i, n in enumerate(tx_nodes)}
        addr_mapping = {n: i for i, n in enumerate(addr_nodes)}
        
        input_edges = []
        output_edges = []
        for u, v, d in sub_G.edges(data=True):
            edge_dir = d.get('direction')
            if edge_dir == 'input' and u in addr_mapping and v in tx_mapping:
                input_edges.append([addr_mapping[u], tx_mapping[v]])
            elif edge_dir == 'output' and u in tx_mapping and v in addr_mapping:
                output_edges.append([tx_mapping[u], addr_mapping[v]])
        
        if input_edges:
            data['address', 'input', 'transaction'].edge_index = torch.tensor(input_edges, dtype=torch.long).t().contiguous()
        else:
            data['address', 'input', 'transaction'].edge_index = torch.empty((2, 0), dtype=torch.long)
        if output_edges:
            data['transaction', 'output', 'address'].edge_index = torch.tensor(output_edges, dtype=torch.long).t().contiguous()
        else:
            data['transaction', 'output', 'address'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        data.y = torch.tensor([label], dtype=torch.float)
        return data

# ==========================================
# 2. 数据加载器 (保持不变)
# ==========================================
class FixedLoader:
    @staticmethod
    def load_bg_data(files, label=0, min_nodes=3, max_nodes=500, target_samples=50000):
        if not files: return []
        dataset = []
        current_count = 0
        batch_size = 20
        total_batches = (len(files) + batch_size - 1) // batch_size
        
        pbar = tqdm(total=target_samples, desc=f"Loading BG ({len(files)} files)", leave=False)
        for i in range(total_batches):
            if current_count >= target_samples: break
            batch_files = files[i*batch_size : (i+1)*batch_size]
            G = nx.DiGraph()
            for f in batch_files:
                try:
                    with open(f, 'r') as jf: tx_list = json.load(jf)
                    for tx in tx_list: GraphProcessor._add_tx_to_graph(G, tx)
                except: continue
            
            components = list(nx.weakly_connected_components(G))
            random.shuffle(components)
            for nodes in components:
                if current_count >= target_samples: break
                if len(nodes) < min_nodes or len(nodes) > max_nodes: continue
                sub_G = G.subgraph(nodes).copy()
                data = GraphProcessor.convert_subgraph_to_heterodata(sub_G, label)
                if data: 
                    # 关键步骤：在加载时直接转为均质图，方便 GCN 使用
                    # PyG 的 to_homogeneous 会自动合并节点特征和边索引
                    homo_data = data.to_homogeneous()
                    dataset.append(homo_data)
                    current_count += 1
                    pbar.update(1)
            del G, components
            gc.collect()
        pbar.close()
        return dataset

    @staticmethod
    def load_cov_data(files, label=1, min_nodes=3, max_nodes=500, target_samples=2100):
        if not files: return []
        dataset = []
        current_count = 0
        pbar = tqdm(total=target_samples, desc=f"Loading Cov ({len(files)} files)", leave=False)
        for f in files:
            if current_count >= target_samples: break
            try:
                with open(f, 'r') as jf: tx_list = json.load(jf)
                G = nx.DiGraph()
                for tx in tx_list: GraphProcessor._add_tx_to_graph(G, tx)
                components = list(nx.weakly_connected_components(G))
                for nodes in components:
                    if current_count >= target_samples: break
                    if len(nodes) < min_nodes or len(nodes) > max_nodes: continue
                    sub_G = G.subgraph(nodes).copy()
                    data = GraphProcessor.convert_subgraph_to_heterodata(sub_G, label)
                    if data:
                        # 同样转为均质图
                        homo_data = data.to_homogeneous()
                        dataset.append(homo_data)
                        current_count += 1
                        pbar.update(1)
                del G
            except: continue
        pbar.close()
        return dataset

# ==========================================
# 3. GCN 模型 (替代 HAN)
# ==========================================
class SimpleGCN(nn.Module):
    def __init__(self, hidden_channels=64, out_channels=1):
        super().__init__()
        # 假设输入特征是 One-Hot Degree (64维)
        # 均质图输入维度 = 64
        self.conv1 = GCNConv(64, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels) # 加深一层以捕获更远结构
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x, edge_index, batch):
        # 1. GCN Layers
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Global Pooling (Readout)
        # 将所有节点（包括交易和地址）的 Embedding 平均
        x = global_mean_pool(x, batch)
        
        # 3. Classifier
        return self.classifier(x)

# ==========================================
# 4. 主程序
# ==========================================
def run_gcn_experiment():
    config = {
        "bg_pattern": "dataset/transactions_block_*.json",
        "methods": {
            "Normal": "dataset/transactions_block_*.json",
            "BlockWhisper": "CompareMethod/BlockWhisper/dataset/BlockWhisper_transactions_*.json",
            "GraphShadow": "CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json",
            "DDSAC": "CompareMethod/DDSAC/dataset/DDSAC_transactions_*.json",
            "GBCTD": "CompareMethod/GBCTD/dataset/GBCTD_transactions_*.json"
        },
        "train_pos": 2000,     
        "train_neg": 6000,     
        "test_pos": 500,      
        "test_neg": 500,     
        "min_nodes": 5, 
        "max_nodes": 100,
        "batch_size": 128,
        "epochs": 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device} | Model: GCN (Homogeneous) | Feat: One-Hot Degree")
    
    all_bg_files = glob.glob(config['bg_pattern'])
    try: all_bg_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    except: pass
    
    files_train_bg = all_bg_files[:50]
    files_test_bg = all_bg_files[50:100]
    files_normal_pos = all_bg_files[100:150]
    
    results_table = []

    for method_name, file_pattern in config['methods'].items():
        print(f"\n>>> Evaluating: {method_name}")
        
        if method_name == "Normal":
            files_train_pos = files_normal_pos[:len(files_normal_pos)//2]
            files_test_pos = files_normal_pos[len(files_normal_pos)//2:]
        else:
            cov_files = glob.glob(file_pattern)
            mid = int(len(cov_files) * 0.5)
            files_train_pos = cov_files[:mid]
            files_test_pos = cov_files[mid:]
        
        if not files_train_pos: continue

        # Loading Data
        print("    [Train] Loading...")
        train_bg_data = FixedLoader.load_bg_data(files_train_bg, label=0, target_samples=config['train_neg'])
        if method_name == "Normal":
            train_pos_data = FixedLoader.load_bg_data(files_train_pos, label=1, target_samples=config['train_pos'])
        else:
            train_pos_data = FixedLoader.load_cov_data(files_train_pos, label=1, target_samples=config['train_pos'])
        train_dataset = train_bg_data + train_pos_data
        
        print(f"    [Test]  Loading...")
        test_bg_data = FixedLoader.load_bg_data(files_test_bg, label=0, target_samples=config['test_neg'])
        if method_name == "Normal":
            test_pos_data = FixedLoader.load_bg_data(files_test_pos, label=1, target_samples=config['test_pos'])
        else:
            test_pos_data = FixedLoader.load_cov_data(files_test_pos, label=1, target_samples=config['test_pos'])
        test_dataset = test_bg_data + test_pos_data
        
        train_y = [d.y.item() for d in train_dataset]
        if train_y.count(1.0) == 0: continue
        
        # Training
        random.shuffle(train_dataset)
        w0 = 1.0 / (train_y.count(0.0) + 1e-5)
        w1 = 1.0 / (train_y.count(1.0) + 1e-5)
        weights = torch.tensor([w1 if t==1 else w0 for t in train_y])
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        # [修改] 使用 SimpleGCN
        model = SimpleGCN(hidden_channels=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(config['epochs']):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                # GCN 的调用方式略有不同：(x, edge_index, batch)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out.view(-1), batch.y)
                loss.backward()
                optimizer.step()
                
        # Evaluation
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                # GCN 推理
                out = model(batch.x, batch.edge_index, batch.batch)
                probs = torch.sigmoid(out).view(-1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy())
                y_probs.extend(probs)
        
        thresholds = [0.99, 0.95, 0.9, 0.7, 0.5, 0.3, 0.1]
        for thr in thresholds:
            y_pred = [1 if p >= thr else 0 for p in y_probs]
            tp = sum((p==1 and t==1) for p,t in zip(y_pred, y_true))
            fp = sum((p==1 and t==0) for p,t in zip(y_pred, y_true))
            tn = sum((p==0 and t==0) for p,t in zip(y_pred, y_true))
            fn = sum((p==0 and t==1) for p,t in zip(y_pred, y_true))
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            results_table.append({
                "Method": method_name, "Thr": thr,
                "Recall": rec, "Prec": prec, "F1": f1,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn
            })

    if results_table:
        df = pd.DataFrame(results_table)
        print("\n" + "="*100)
        print("FINAL RESULTS (GCN Model - Homogeneous Topology)")
        print("="*100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        df.to_csv("gcn_topology_results.csv", index=False)

if __name__ == "__main__":
    run_gcn_experiment()