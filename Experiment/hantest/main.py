import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, global_mean_pool
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
# 1. 图处理核心 (含 Ratio 特征)
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
    def calculate_directed_diameter(G):
        """快速估算有向直径 (采样20个点)"""
        nodes = list(G.nodes())
        if not nodes: return 0
        if len(nodes) > 20: target_sources = random.sample(nodes, 20)
        else: target_sources = nodes
        
        max_dist = 0
        for start_node in target_sources:
            try: 
                lengths = nx.single_source_shortest_path_length(G, start_node)
                if lengths:
                    current_max = max(lengths.values())
                    if current_max > max_dist: max_dist = current_max
            except: continue
        return max_dist

    @staticmethod
    def convert_subgraph_to_heterodata(sub_G, label):
        data = HeteroData()
        tx_nodes = [n for n, d in sub_G.nodes(data=True) if d.get('node_type') == 'transaction']
        addr_nodes = [n for n, d in sub_G.nodes(data=True) if d.get('node_type') == 'address']
        
        if len(tx_nodes) == 0: return None
        
        # --- [特征] 计算比值特征 Ratio (Diameter / NumNodes) ---
        num_nodes = sub_G.number_of_nodes()
        diameter = GraphProcessor.calculate_directed_diameter(sub_G)
        ratio = diameter / num_nodes if num_nodes > 0 else 0
        
        # 将标量扩展为 [N, 1] 的张量
        ratio_feat_tx = torch.tensor([[ratio]], dtype=torch.float).repeat(len(tx_nodes), 1)
        ratio_feat_addr = torch.tensor([[ratio]], dtype=torch.float).repeat(len(addr_nodes), 1)

        # --- [特征] 节点微观特征 (In/Out Degree + Log) ---
        def build_tx_features(nodes):
            LIMIT = 32
            in_degs = [sub_G.in_degree(n) for n in nodes]
            in_tensor = torch.tensor(in_degs, dtype=torch.long).clamp(max=LIMIT-1)
            one_hot_in = F.one_hot(in_tensor, num_classes=LIMIT).float()
            log_in = torch.tensor([math.log(d + 1) for d in in_degs], dtype=torch.float).unsqueeze(1)
            
            out_degs = [sub_G.out_degree(n) for n in nodes]
            out_tensor = torch.tensor(out_degs, dtype=torch.long).clamp(max=LIMIT-1)
            one_hot_out = F.one_hot(out_tensor, num_classes=LIMIT).float()
            log_out = torch.tensor([math.log(d + 1) for d in out_degs], dtype=torch.float).unsqueeze(1)
            
            # 基础维度: 32+1 + 32+1 = 66
            return torch.cat([one_hot_in, log_in, one_hot_out, log_out], dim=1)

        def build_addr_features(nodes):
            LIMIT = 64
            degs = [sub_G.degree(n) for n in nodes]
            deg_tensor = torch.tensor(degs, dtype=torch.long).clamp(max=LIMIT-1)
            one_hot_deg = F.one_hot(deg_tensor, num_classes=LIMIT).float()
            log_deg = torch.tensor([math.log(d + 1) for d in degs], dtype=torch.float).unsqueeze(1)
            # 基础维度: 64+1 = 65
            return torch.cat([one_hot_deg, log_deg], dim=1)

        # 拼接 Ratio 特征
        # Tx Dim: 66 + 1 = 67
        data['transaction'].x = torch.cat([build_tx_features(tx_nodes), ratio_feat_tx], dim=1)
        # Addr Dim: 65 + 1 = 66
        data['address'].x = torch.cat([build_addr_features(addr_nodes), ratio_feat_addr], dim=1)
        
        # 建立索引映射
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
# 2. 修复后的加载器 (分离加载逻辑)
# ==========================================
class FixedLoader:
    @staticmethod
    def load_bg_data(files, label=0, min_nodes=3, max_nodes=500, target_samples=50000):
        """背景数据：允许批量合并 (Batch=20)，为了重构大图网络"""
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
                    with open(f, 'r') as jf:
                        tx_list = json.load(jf)
                    for tx in tx_list:
                        GraphProcessor._add_tx_to_graph(G, tx)
                except: continue
            
            components = list(nx.weakly_connected_components(G))
            # 随机打乱，避免只取到头部数据
            random.shuffle(components)
            
            for nodes in components:
                if current_count >= target_samples: break
                if len(nodes) < min_nodes or len(nodes) > max_nodes: continue
                
                sub_G = G.subgraph(nodes).copy()
                data = GraphProcessor.convert_subgraph_to_heterodata(sub_G, label)
                if data: 
                    dataset.append(data)
                    current_count += 1
                    pbar.update(1)
            del G, components
            gc.collect()
        pbar.close()
        return dataset

    @staticmethod
    def load_cov_data(files, label=1, min_nodes=3, max_nodes=500, target_samples=2100):
        """[关键修复] 隐蔽数据：逐个文件加载，严禁合并，防止 BlockWhisper 变成巨型图"""
        if not files: return []
        dataset = []
        current_count = 0
        
        pbar = tqdm(total=target_samples, desc=f"Loading Cov ({len(files)} files)", leave=False)
        
        for f in files:
            if current_count >= target_samples: break
            try:
                with open(f, 'r') as jf:
                    tx_list = json.load(jf)
                
                G = nx.DiGraph()
                for tx in tx_list:
                    GraphProcessor._add_tx_to_graph(G, tx)
                
                components = list(nx.weakly_connected_components(G))
                for nodes in components:
                    if current_count >= target_samples: break
                    if len(nodes) < min_nodes or len(nodes) > max_nodes: continue
                        
                    sub_G = G.subgraph(nodes).copy()
                    data = GraphProcessor.convert_subgraph_to_heterodata(sub_G, label)
                    if data:
                        dataset.append(data)
                        current_count += 1
                        pbar.update(1)
                del G
            except: continue
            
        pbar.close()
        return dataset

# ==========================================
# 3. HAN 模型 (Input Dim = 67/66)
# ==========================================
class RatioHAN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=1):
        super().__init__()
        # Tx Input: 66 (Base) + 1 (Ratio) = 67
        self.proj_tx = nn.Linear(67, hidden_channels)
        # Addr Input: 65 (Base) + 1 (Ratio) = 66
        self.proj_addr = nn.Linear(66, hidden_channels)
        
        self.han_conv = HANConv(hidden_channels, hidden_channels, heads=4, dropout=0.2, metadata=metadata)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, out_channels)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict):
        x_dict['transaction'] = self.proj_tx(x_dict['transaction'])
        x_dict['address'] = self.proj_addr(x_dict['address'])
        out_dict = self.han_conv(x_dict, edge_index_dict)
        tx_feature = out_dict['transaction']
        
        if batch_dict is not None and 'transaction' in batch_dict:
            graph_embedding = global_mean_pool(tx_feature, batch_dict['transaction'])
        else:
            graph_embedding = tx_feature.mean(dim=0, keepdim=True)
            
        return self.classifier(graph_embedding)

# ==========================================
# 4. 主程序 (平衡测试 + 对照组)
# ==========================================
def run_balanced_control():
    config = {
        "bg_pattern": "dataset/transactions_block_*.json",
        "methods": {
            "Normal": "dataset/transactions_block_*.json",  # 对照组
            "BlockWhisper": "CompareMethod/BlockWhisper/dataset/BlockWhisper_transactions_*.json",
            "GraphShadow": "CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json",
            "DDSAC": "CompareMethod/DDSAC/dataset/DDSAC_transactions_*.json",
            "GBCTD": "CompareMethod/GBCTD/dataset/GBCTD_transactions_*.json"
        },
        # 训练集规模
        "train_pos": 2000,     
        "train_neg": 6000,     
        
        # 测试集规模 (平衡 1:1)
        "test_pos": 500,      
        "test_neg": 500,     
        
        "min_nodes": 3, 
        "max_nodes": 500,
        "batch_size": 128,
        "epochs": 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device} | Test Ratio: 1:1 (Balanced) | Control Group: Normal")
    
    # 1. 准备背景文件
    all_bg_files = glob.glob(config['bg_pattern'])
    try: all_bg_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    except: pass
    
    if len(all_bg_files) < 100:
        print(f"[!] Warning: Only {len(all_bg_files)} BG files found.")
    
    # 划分文件集：防止数据泄露
    # 前50个用于训练背景
    files_train_bg = all_bg_files[:50]
    # 中间50个用于测试背景 (作为 Neg)
    files_test_bg = all_bg_files[50:100]
    # 后50个用于 Normal 对照组 (作为 Pos)
    files_normal_pos = all_bg_files[100:150]
    
    results_table = []

    for method_name, file_pattern in config['methods'].items():
        print(f"\n>>> Evaluating: {method_name}")
        
        # 确定正样本来源
        if method_name == "Normal":
            # 对照组：正样本来自 Normal 专用的背景文件
            files_train_pos = files_normal_pos[:len(files_normal_pos)//2]
            files_test_pos = files_normal_pos[len(files_normal_pos)//2:]
            
            # 注意：如果是 Normal 测试，训练阶段我们通常需要模拟“已知攻击”来测试对 Normal 的误报
            # 或者我们模拟“Normal vs Normal”来测试区分度
            # 这里我们采用 A/A Test 逻辑：训练集也是 Normal vs BG，测试集也是 Normal vs BG
            # 理论上模型应该学不到任何区别。
        else:
            cov_files = glob.glob(file_pattern)
            mid = int(len(cov_files) * 0.5)
            files_train_pos = cov_files[:mid]
            files_test_pos = cov_files[mid:]
        
        if not files_train_pos: continue

        # --- 2. Train Loading ---
        print("    [Train] Loading...")
        # 训练背景
        train_bg_data = FixedLoader.load_bg_data(
            files_train_bg, label=0, 
            min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['train_neg']
        )
        # 训练正样本
        if method_name == "Normal":
             # 对照组特殊处理：用 load_bg_data 加载 Normal (因为它们本质是 BG)
             train_pos_data = FixedLoader.load_bg_data(
                files_train_pos, label=1,
                min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['train_pos']
            )
        else:
            train_pos_data = FixedLoader.load_cov_data(
                files_train_pos, label=1, 
                min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['train_pos']
            )
        train_dataset = train_bg_data + train_pos_data
        
        # --- 3. Test Loading ---
        print(f"    [Test]  Loading Target: Pos={config['test_pos']}, Neg={config['test_neg']}")
        # 测试背景
        test_bg_data = FixedLoader.load_bg_data(
            files_test_bg, label=0, 
            min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['test_neg']
        )
        # 测试正样本
        if method_name == "Normal":
             test_pos_data = FixedLoader.load_bg_data(
                files_test_pos, label=1,
                min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['test_pos']
            )
        else:
            test_pos_data = FixedLoader.load_cov_data(
                files_test_pos, label=1, 
                min_nodes=config['min_nodes'], max_nodes=config['max_nodes'], target_samples=config['test_pos']
            )
        test_dataset = test_bg_data + test_pos_data
        
        train_y = [d.y.item() for d in train_dataset]
        test_y = [d.y.item() for d in test_dataset]
        print(f"    -> Train Actual: {len(train_dataset)} (Pos={train_y.count(1.0)})")
        print(f"    -> Test Actual:  {len(test_dataset)} (Pos={test_y.count(1.0)}, Neg={test_y.count(0.0)})")
        
        if train_y.count(1.0) == 0: continue

        # --- 4. Training ---
        random.shuffle(train_dataset)
        w0 = 1.0 / (train_y.count(0.0) + 1e-5)
        w1 = 1.0 / (train_y.count(1.0) + 1e-5)
        weights = torch.tensor([w1 if t==1 else w0 for t in train_y])
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        model = RatioHAN(train_dataset[0].metadata()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        for epoch in range(config['epochs']):
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                loss = criterion(out.view(-1), batch.y)
                loss.backward()
                optimizer.step()
                
        # --- 5. Evaluation ---
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                probs = torch.sigmoid(out).view(-1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy())
                y_probs.extend(probs)
        
        # 只取几个关键阈值
        thresholds = [0.99, 0.9, 0.5]
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

    # --- 6. Output ---
    if results_table:
        df = pd.DataFrame(results_table)
        print("\n" + "="*100)
        print("FINAL RESULTS (Balanced 1:1 with Control Group)")
        print("="*100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        df.to_csv("balanced_control_results.csv", index=False)

if __name__ == "__main__":
    run_balanced_control()