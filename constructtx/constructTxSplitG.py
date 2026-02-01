import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampleaddressdegree import AddressCentralDegreeSampler
from txgraph.main import BitcoinTransactionGraph
from sampletxinout import TxInOutSampler
from samplediameter import DiameterSampler
import utils
import pandas as pd
import re
import networkx as nx

# =========================================================
# 初始化采样器（原样）
# =========================================================

def init_TxInOutSampler():
    discrete_date = {(1, 2): 15796, (1, 1): 11722}
    continuous_params = {
        'A': 391696615.17,
        'alpha': 14.04, 'beta': 17.33,
        'lam1': 1.7308, 'lam2': 3.1599, 'lam3': 12.5264
    }
    lambda_mix = (15796 + 11722) / (15796 + 11722 + 3394)
    return TxInOutSampler(discrete_date, continuous_params, lambda_mix, max_range=5)

def init_AddressCentralDegreeSampler():
    params = {'A': 397541.13, 'alpha': -5.39}
    return AddressCentralDegreeSampler(params, min_x=2, max_x=100)

# =========================================================
# 期望计算（原样）
# =========================================================

def calculate_utxos(msgSizePerBit):
    return msgSizePerBit / 256

def calculate_txinout_expectations(sampler):
    disc_n = np.array([p[0] for p in sampler.discrete_points])
    disc_m = np.array([p[1] for p in sampler.discrete_points])
    e_n_disc = np.sum(disc_n * sampler.discrete_probs)
    e_m_disc = np.sum(disc_m * sampler.discrete_probs)

    p_n = np.sum(sampler.cont_probs_grid, axis=1)
    p_m = np.sum(sampler.cont_probs_grid, axis=0)
    e_n_cont = np.sum(sampler.cont_n_vals * p_n)
    e_m_cont = np.sum(sampler.cont_m_vals * p_m)

    lam = sampler.lambda_mix
    return lam * e_n_disc + (1 - lam) * e_n_cont, \
           lam * e_m_disc + (1 - lam) * e_m_cont

def calculate_addresscentrality_expectations(sampler):
    return np.sum(sampler.x_vals * sampler.probs)

# =========================================================
# >>> MODIFIED <<< 中心地址初始化（引入 component_id）
# =========================================================

def init_center_addresses(N_center, N_comm, centrality_sampler, n_components=5):
    central_states = []
    for i in range(int(N_center)):
        state = {
            'address': utils.generate_random_address(),
            'bal': 0,
            'd_tgt': int(centrality_sampler.sample()[0]),
            'component_id': i % n_components   # <<< 核心修改
        }
        central_states.append(state)
    return central_states

# =========================================================
# 分量信息更新（原样）
# =========================================================

def update_central_states_with_components(graph, central_states):
    if graph.graph.number_of_nodes() == 0:
        for s in central_states:
            s['comp_size'] = 0
        return

    components = list(nx.weakly_connected_components(graph.graph))
    node_size = {}
    for comp in components:
        for n in comp:
            node_size[n] = len(comp)

    for s in central_states:
        s['comp_size'] = node_size.get(s['address'], 0)

# =========================================================
# >>> MODIFIED <<< Phase 1：中心 → 通信（通信地址带 component_id）
# =========================================================

def construct_token_distribute_transaction(tx_sampler, central_states, graph):
    tx_id = utils.generate_tx_id()
    n, m = tx_sampler.sample(size=1)[0]

    candidates = [i for i, s in enumerate(central_states) if s['bal'] > 0]
    if not candidates:
        return 'Done'

    sender_idx = np.random.choice(candidates)
    sender = central_states[sender_idx]

    outputs = []
    for _ in range(m):
        addr = utils.generate_random_address()
        outputs.append(addr)
        # 给通信地址绑定 component_id
        COMM_ADDR_COMPONENT[addr] = sender['component_id']

    graph.add_transaction(tx_id, [sender['address']], outputs)
    sender['bal'] -= 1

    return {'txid': tx_id, 'inputs': [sender['address']], 'outputs': outputs}

# =========================================================
# >>> MODIFIED <<< Phase 2：通信 → 中心（组件内回流）
# =========================================================

def construct_message_communication_transaction(tx_sampler, comm_addresses, central_states, graph):
    tx_id = utils.generate_tx_id()
    n, m = tx_sampler.sample(size=1)[0]

    n = min(n, len(comm_addresses))
    inputs = list(np.random.choice(comm_addresses, n, replace=False))

    # 所有输入通信地址必须属于同一 component
    comp_id = COMM_ADDR_COMPONENT[inputs[0]]

    candidates = [
        i for i, s in enumerate(central_states)
        if s['component_id'] == comp_id
    ]
    if not candidates:
        return 'Exceed'

    outputs = [central_states[i]['address'] for i in
               np.random.choice(candidates, m, replace=True)]

    graph.add_transaction(tx_id, inputs, outputs)

    for i in outputs:
        for s in central_states:
            if s['address'] == i:
                s['bal'] += 1

    for addr in inputs:
        comm_addresses.remove(addr)
        del COMM_ADDR_COMPONENT[addr]

    return {'txid': tx_id, 'inputs': inputs, 'outputs': outputs}

# =========================================================
# 主程序
# =========================================================

COMM_ADDR_COMPONENT = {}

if __name__ == "__main__":

    N_utxo = int(calculate_utxos(512 * 8))
    tx_sampler = init_TxInOutSampler()
    addr_sampler = init_AddressCentralDegreeSampler()

    central_states = init_center_addresses(
        N_center=10,
        N_comm=N_utxo,
        centrality_sampler=addr_sampler,
        n_components=5
    )

    central_states[0]['bal'] = 2
    central_states[1]['bal'] = 2
    central_states[2]['bal'] = 2

    btg = BitcoinTransactionGraph()

    round_idx = 0
    while N_utxo > 0:
        round_comm = []

        while N_utxo > 0:
            update_central_states_with_components(btg, central_states)
            tx = construct_token_distribute_transaction(
                tx_sampler, central_states, btg
            )
            if tx == 'Done':
                break
            round_comm.extend(tx['outputs'])
            N_utxo -= len(tx['outputs'])

        while round_comm:
            update_central_states_with_components(btg, central_states)
            tx = construct_message_communication_transaction(
                tx_sampler, round_comm, central_states, btg
            )
            if tx == 'Exceed':
                break

        round_idx += 1

    btg.visualize()
    print("Final diameter:", btg.calculate_diameter())
