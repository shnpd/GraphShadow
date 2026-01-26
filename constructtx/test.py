import random
from collections import Counter

import numpy as np

from constructtx.utils import generate_random_address
from crawler.crawler import save_block_transaction
from graphanalysis.path_length import get_max_diameter
from txgraph.main import BitcoinTransactionGraph


def weighted_random_choice(items, weights, num_selections=100):
    # 计算累积权重
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("总权重必须大于0")
    # 归一化权重
    cumulative_weights = []
    cumulative_sum = 0
    for weight in weights:
        cumulative_sum += weight / total_weight
        cumulative_weights.append(cumulative_sum)

    # 执行加权随机选择
    selections = []
    for _ in range(num_selections):
        rand_val = random.random()  # 生成[0, 1)之间的随机数

        # 根据权重选择元素
        for i, cum_weight in enumerate(cumulative_weights):
            if rand_val <= cum_weight:
                selections.append(items[i])
                break

    # 统计选择结果
    stats = {
        'total_selections': num_selections,
        'counts': dict(Counter(selections)),
        'percentages': {},
        'expected_percentages': {}
    }

    # 计算实际百分比和期望百分比
    for i, item in enumerate(items):
        if item in stats['counts']:
            stats['percentages'][item] = stats['counts'][item] / num_selections * 100
        else:
            stats['percentages'][item] = 0

        stats['expected_percentages'][item] = weights[i] / total_weight * 100

    return selections, stats


if __name__ == "__main__":
    # tmp = [2,1,3]
    # tmp.sort()
    # print(tmp)
    save_block_transaction(923800, 923900)

    # btg = BitcoinTransactionGraph()
    # btg.add_transaction('1',['a'],['b'])
    # btg.add_transaction('2',['b'],['a'])
    # btg.add_transaction('3',['a'],['c'])
    # btg.add_transaction('4',['c'],['a'])
    # print(get_max_diameter(btg))
