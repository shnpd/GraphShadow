import re

import numpy as np
import pandas as pd


class DiameterSampler:
    def __init__(self, csv_path):
        """
        初始化直径采样器
        :param csv_path: 包含概率分布的 CSV 文件路径
        """
        self.df = pd.read_excel(csv_path)
        # 1. 提取直径候选值 (第一列)
        # 假设第一列名为 'Path Length'，如果不确定可以用 iloc
        self.diameter_values = self.df.iloc[:, 0].values

        # 2. 解析分箱区间 (从第二列开始的表头)
        self.intervals = []
        self.column_names = []

        # 正则表达式匹配 "[0,30)" 格式
        pattern = re.compile(r"\[(\d+),(\d+)\)")

        for col in self.df.columns[1:]:
            match = pattern.search(col)
            if match:
                lower = int(match.group(1))
                upper = int(match.group(2))
                self.intervals.append((lower, upper))
                self.column_names.append(col)
            else:
                print(f"警告: 无法解析列名 '{col}'，已跳过。")

        # 3. 归一化概率，保存列名与概率分布的映射
        self.prob_map = {}
        for col in self.column_names:
            raw_probs = self.df[col].values
            total = np.sum(raw_probs)
            self.prob_map[col] = raw_probs / total

    def get_interval_column(self, num_nodes):
        """根据节点数查找对应的区间列名"""
        # 1. 遍历区间查找
        for (lower, upper), col_name in zip(self.intervals, self.column_names):
            if lower <= num_nodes < upper:
                return col_name

        # 2. 边界处理
        # 如果小于最小值 (比如负数或 0 如果区间从1开始)，返回第一个区间
        if num_nodes < self.intervals[0][0]:
            return self.column_names[0]

        # 如果大于最大值 (比如 350)，返回最后一个区间 (假设分布趋于稳定)
        if num_nodes >= self.intervals[-1][1]:
            return self.column_names[-1]

        return None

    def sample(self, num_nodes, size=1):
        """
        核心采样函数
        :param num_nodes: 交易图的节点个数
        :param size: 采样样本数
        :return: 采样得到的直径 (int 或 numpy array)
        """
        # 1. 确定使用哪一列的概率分布
        target_col = self.get_interval_column(num_nodes)

        if target_col is None:
            raise ValueError(f"无法为节点数 {num_nodes} 找到合适的区间。")

        # 2. 获取概率分布
        probs = self.prob_map[target_col]

        # 3. 执行采样
        sampled_diameters = np.random.choice(
            self.diameter_values,
            size=size,
            p=probs
        )

        return sampled_diameters

if __name__ == "__main__":
    csv_file_path = "path_length_probability_matrix.xlsx"
    diameter_sampler = DiameterSampler(csv_file_path)
    samples = diameter_sampler.sample(30,1)
    print(samples)
