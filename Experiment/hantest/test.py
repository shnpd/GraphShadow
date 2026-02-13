import json
import networkx as nx
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from collections import Counter

# === 配置 ===
files_bg = glob.glob("dataset/transactions_block_*.json")[:20]  # 取20个背景文件
files_gs = glob.glob("CompareMethod/GraphShadow/dataset/GraphShadow_transactions_*.json")[:20] # 取20个GraphShadow

def get_degree_distribution(files):
    all_degrees = []
    
    for f in files:
        try:
            with open(f, 'r') as jf: tx_list = json.load(jf)
            G = nx.DiGraph()
            # 简易构图
            for tx in tx_list:
                tx_hash = tx.get('hash', 'unknown')
                G.add_node(tx_hash)
                inputs = tx.get('inputs', []) or tx.get('input_addrs', [])
                for inp in inputs:
                    addr = inp if isinstance(inp, str) else inp.get('addresses', ['?'])[0]
                    G.add_edge(addr, tx_hash)
                outputs = tx.get('outputs', []) or tx.get('output_addrs', [])
                for out in outputs:
                    addr = out if isinstance(out, str) else out.get('addresses', ['?'])[0]
                    G.add_edge(tx_hash, addr)
            
            # 提取连通分量的度数
            components = list(nx.weakly_connected_components(G))
            for nodes in components:
                sub_G = G.subgraph(nodes)
                # 只统计稍微大一点的图，排除噪音
                if sub_G.number_of_nodes() >= 3:
                    degs = [d for n, d in sub_G.degree()]
                    all_degrees.extend(degs)
        except: continue
        
    return all_degrees

def analyze_degrees():
    print("[-] Analyzing Degree Distributions...")
    deg_bg = get_degree_distribution(files_bg)
    deg_gs = get_degree_distribution(files_gs)
    
    # 统计频率
    count_bg = Counter(deg_bg)
    count_gs = Counter(deg_gs)
    
    # 转换为概率分布
    total_bg = sum(count_bg.values())
    total_gs = sum(count_gs.values())
    
    max_deg = 10 # 只看前10个度数，通常这就够了
    
    print(f"\n{'Degree':<6} | {'Background %':<12} | {'GraphShadow %':<12} | {'Diff':<10}")
    print("-" * 50)
    
    diff_sum = 0
    for d in range(1, max_deg + 1):
        p_bg = count_bg.get(d, 0) / total_bg * 100
        p_gs = count_gs.get(d, 0) / total_gs * 100
        diff = p_gs - p_bg
        diff_sum += abs(diff)
        print(f"{d:<6} | {p_bg:<12.2f} | {p_gs:<12.2f} | {diff:<+10.2f}")
        
    print("-" * 50)
    print(f"Total Absolute Difference (Top {max_deg}): {diff_sum:.2f}")
    
    if diff_sum > 15:
        print("\n[CONCLUSION] SIGNIFICANT DEGREE LEAKAGE DETECTED!")
        print("GraphShadow has a distinct degree fingerprint.")
        print("The model is using 'One-Hot Degree' to cheat.")

if __name__ == "__main__":
    analyze_degrees()