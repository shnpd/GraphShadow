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
# 1. 图处理核心 (全图直径计算)
# ==========================================
class GraphProcessor:
    @staticmethod
    def build_graph_from_tx_list(tx_list):
        """将交易列表直接构建为 NetworkX 图"""
        G = nx.DiGraph()
        for tx in tx_list:
            tx_hash = tx.get('hash', 'unknown')
            G.add_node(tx_hash, node_type='transaction')
            
            # 处理输入
            inputs = tx.get('inputs', []) or tx.get('input_addrs', [])
            for inp in inputs:
                addr = inp if isinstance(inp, str) else inp.get('addresses', ['?'])[0]
                G.add_node(addr, node_type='address')
                G.add_edge(addr, tx_hash, direction='input')
            
            # 处理输出
            outputs = tx.get('outputs', []) or tx.get('output_addrs', [])
            for out in outputs: 
                addr = out if isinstance(out, str) else out.get('addresses', ['?'])[0]
                G.add_node(addr, node_type='address')
                G.add_edge(tx_hash, addr, direction='output')
        return G

    @staticmethod
    def calculate_full_diameter(G):
        """
        [核心修改] 全图扫描计算最大直径
        逻辑：对图中每一个节点进行 BFS，找到它能到达的最远距离。
        这能确保捕捉到任何因隐蔽交易插入而延长的路径。
        """
        nodes = list(G.nodes())
        num_nodes = len(nodes)
        
        if num_nodes == 0: return 0
        
        # 策略：如果图不算太大 (<2000节点)，全量计算
        # 如果图极大，为了防止训练卡死，回退到高比例采样 (例如采样 500 个点)
        if num_nodes > 2000:
            target_sources = random.sample(nodes, 500)
        else:
            target_sources = nodes
            
        max_dist = 0
        
        # 遍历所有源点
        for start_node in target_sources:
            try:
                # single_source_shortest_path_length 返回一个字典 {target: distance}
                # 这比 diameter() 函数更鲁棒，因为交易图通常不是强连通的
                lengths = nx.single_source_shortest_path_length(G, start_node)
                
                if lengths:
                    # 找到该节点能到达的最远距离
                    current_max = max(lengths.values())
                    if current_max > max_dist:
                        max_dist = current_max
            except:
                continue
                
        return max_dist

    @staticmethod
    def convert_nx_to_heterodata(G, label):
        """将 NetworkX 图转换为 PyG HeteroData"""
        if G.number_of_nodes() < 5: return None

        data = HeteroData()
        tx_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'transaction']
        addr_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'address']
        
        if len(tx_nodes) == 0: return None
        
        # === 1. 局部拓扑: One-Hot Degree (64维) ===
        def get_degree_one_hot(nodes, max_degree=64):
            degrees = [G.degree(n) for n in nodes]
            deg_tensor = torch.tensor(degrees, dtype=torch.long).clamp(max=max_degree-1)
            return F.one_hot(deg_tensor, num_classes=max_degree).float()

        tx_deg_feat = get_degree_one_hot(tx_nodes)     # [N_tx, 64]
        addr_deg_feat = get_degree_one_hot(addr_nodes) # [N_addr, 64]

        # === 2. 全局拓扑: Full Diameter (1维) ===
        # 使用全图扫描计算出的直径
        diameter = GraphProcessor.calculate_full_diameter(G)
        
        # 扩展特征
        tx_dia_feat = torch.tensor([[diameter]], dtype=torch.float).repeat(len(tx_nodes), 1)
        addr_dia_feat = torch.tensor([[diameter]], dtype=torch.float).repeat(len(addr_nodes), 1)

        # === 3. 拼接 (65维) ===
        data['transaction'].x = torch.cat([tx_deg_feat, tx_dia_feat], dim=1)
        data['address'].x = torch.cat([addr_deg_feat, addr_dia_feat], dim=1)
        
        # === 4. 边索引 ===
        tx_mapping = {n: i for i, n in enumerate(tx_nodes)}
        addr_mapping = {n: i for i, n in enumerate(addr_nodes)}
        
        input_edges = []
        output_edges = []
        for u, v, d in G.edges(data=True):
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
# 2. 嵌入式数据加载器
# ==========================================
class EmbeddingLoader:
    @staticmethod
    def load_pure_normal(normal_files, max_samples=None):
        dataset = []
        if max_samples is None: max_samples = len(normal_files)
        
        pbar = tqdm(normal_files[:max_samples], desc="Loading Pure Normal (Label 0)", leave=False)
        for f in pbar:
            try:
                with open(f, 'r') as jf: tx_list = json.load(jf)
                G = GraphProcessor.build_graph_from_tx_list(tx_list)
                data = GraphProcessor.convert_nx_to_heterodata(G, label=0)
                if data: dataset.append(data)
            except Exception as e:
                continue
        return dataset

    @staticmethod
    def load_embedded_normal(normal_files, covert_files, max_samples=None):
        dataset = []
        num_pairs = min(len(normal_files), len(covert_files))
        if max_samples: num_pairs = min(num_pairs, max_samples)
        
        pairs = zip(normal_files[:num_pairs], covert_files[:num_pairs])
        
        pbar = tqdm(pairs, total=num_pairs, desc="Loading Embedded (Label 1)", leave=False)
        for bg_file, cov_file in pbar:
            try:
                with open(bg_file, 'r') as f1: bg_txs = json.load(f1)
                with open(cov_file, 'r') as f2: cov_txs = json.load(f2)
                
                merged_txs = bg_txs + cov_txs
                G = GraphProcessor.build_graph_from_tx_list(merged_txs)
                data = GraphProcessor.convert_nx_to_heterodata(G, label=1)
                if data: dataset.append(data)
            except Exception as e:
                continue
        return dataset

# ==========================================
# 3. HAN 模型 (Input Dim = 65)
# ==========================================
class PureHAN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=1):
        super().__init__()
        # Input: 64(Degree) + 1(Full Diameter) = 65
        self.proj_tx = nn.Linear(65, hidden_channels)
        self.proj_addr = nn.Linear(65, hidden_channels)
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
# 4. 主程序
# ==========================================
def run_full_diameter_experiment():
    config = {
        "bg_pattern": "dataset/transactions_block_*.json",
        "methods": {
            # "BlockWhisper": "CompareMethod/BlockWhisper/dataset/BlockWhisper_transactions_*.json",
            # "GraphShadow": "CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json",
            # "DDSAC": "CompareMethod/DDSAC/dataset/DDSAC_transactions_*.json",
            "GBCTD": "CompareMethod/GBCTD/dataset/GBCTD_transactions_*.json"
        },
        # 样本量分配
        "train_pairs": 300,  
        "test_pairs": 210,   
        
        "batch_size": 16,
        "epochs": 15
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Device: {device} | Feat: Degree + Full Diameter (No Sampling) | File Embedding")
    
    all_bg_files = glob.glob(config['bg_pattern'])
    random.seed(42)
    random.shuffle(all_bg_files)
    
    n1 = config['train_pairs']
    n2 = config['train_pairs']
    n3 = config['test_pairs']
    n4 = config['test_pairs']
    total_needed = n1 + n2 + n3 + n4
    
    if len(all_bg_files) < total_needed:
        print(f"[!] Warning: Not enough BG files. Scaling down.")
        limit = len(all_bg_files) // 4
        n1 = n2 = n3 = n4 = limit
        config['train_pairs'] = limit
        config['test_pairs'] = limit

    idx = 0
    bg_train_pure = all_bg_files[idx : idx+n1]; idx += n1
    bg_train_embed_base = all_bg_files[idx : idx+n2]; idx += n2
    bg_test_pure = all_bg_files[idx : idx+n3]; idx += n3
    bg_test_embed_base = all_bg_files[idx : idx+n4]; idx += n4
    
    print(f"[*] Split: Train={n1} pairs, Test={n3} pairs")

    results_table = []

    for method_name, file_pattern in config['methods'].items():
        if method_name == "Normal": continue
        
        print(f"\n>>> Evaluating Embedding: {method_name}")
        
        cov_files = glob.glob(file_pattern)
        random.shuffle(cov_files)
        
        if len(cov_files) < (n1 + n3):
             print(f"[!] Not enough covert files. Duplicating.")
             cov_files = cov_files * ((n1+n3)//len(cov_files) + 1)
        
        cov_train_files = cov_files[:n1]
        cov_test_files = cov_files[n1 : n1+n3]

        print("    [Train] Generating dataset...")
        train_data_0 = EmbeddingLoader.load_pure_normal(bg_train_pure)
        train_data_1 = EmbeddingLoader.load_embedded_normal(bg_train_embed_base, cov_train_files)
        train_dataset = train_data_0 + train_data_1
        
        print("    [Test]  Generating dataset...")
        test_data_0 = EmbeddingLoader.load_pure_normal(bg_test_pure)
        test_data_1 = EmbeddingLoader.load_embedded_normal(bg_test_embed_base, cov_test_files)
        test_dataset = test_data_0 + test_data_1
        
        train_y = [d.y.item() for d in train_dataset]
        if len(train_dataset) == 0: continue

        # --- Training ---
        random.shuffle(train_dataset)
        w0 = 1.0 / (train_y.count(0.0) + 1e-5)
        w1 = 1.0 / (train_y.count(1.0) + 1e-5)
        weights = torch.tensor([w1 if t==1 else w0 for t in train_y])
        sampler = WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
        
        model = PureHAN(train_dataset[0].metadata()).to(device)
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
                
        # --- Evaluation ---
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                probs = torch.sigmoid(out).view(-1).cpu().numpy()
                y_true.extend(batch.y.cpu().numpy())
                y_probs.extend(probs)
        
        thresholds = [0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
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
        print("FINAL RESULTS (Full Diameter Embedding)")
        print("="*100)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.to_string(index=False))
        
        # 保存 CSV (已有)
        df.to_csv("full_diameter_results.csv", index=False)
        
        # === [新增] 保存为 Excel ===
        excel_path = "full_diameter_results.xlsx"
        try:
            # 需要安装 openpyxl: pip install openpyxl
            df.to_excel(excel_path, index=False, sheet_name='Experiment Results')
            print(f"[*] Excel file saved successfully to: {excel_path}")
        except ImportError:
            print("[!] Error: 'openpyxl' library is missing. Install it using: pip install openpyxl")
        except Exception as e:
            print(f"[!] Error saving Excel: {e}")

if __name__ == "__main__":
    run_full_diameter_experiment()