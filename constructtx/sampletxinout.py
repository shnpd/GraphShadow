import numpy as np


class TxInOutSampler:
    def __init__(self, discrete_map, params, lambda_mix, max_range=100):
        """
            初始化交易输入输出数量采样器
            混合分布：𝑃_𝑡𝑥(𝑛,𝑚)=𝜆⋅𝑃_𝑑𝑖𝑠𝑐𝑟𝑒𝑡𝑒(𝑛,𝑚)+(1−𝜆)⋅𝑃_𝑐𝑜𝑛𝑡𝑖𝑛𝑢𝑜𝑢𝑠(𝑛,𝑚)
            :param discrete_map: 字典，包含特殊点的相对频次，如 {(1,1): 10, (1,2): 1}
            :param params: 字典，包含拟合函数𝑦(𝑛,𝑚)=𝐴⋅𝑛^𝛼⋅𝑚^𝛽⋅𝑒^(−(𝜆_1 𝑛+𝜆_2 𝑚+𝜆_3 √𝑛𝑚) )的参数 alpha, beta, lam1, lam2, lam3
            :param lambda_mix: 混合系数 (0 <= lambda <= 1)，表示采样落入离散区域的概率
            :param max_range: 连续分布采样的截断范围
        """
        self.discrete_map = discrete_map
        self.params = params
        self.lambda_mix = lambda_mix
        self.max_range = max_range

        # 预计算概率表，避免重复计算
        self._prepare_discrete_probs()
        self._prepare_continuous_probs()

    def _prepare_discrete_probs(self):
        """预处理 P_discrete：归一化经验频次"""
        points = list(self.discrete_map.keys())
        counts = np.array(list(self.discrete_map.values()), dtype=float)

        # 归一化，使其成为一个合法的 PMF
        probs = counts / counts.sum()

        self.discrete_points = points  # list of tuples [(1,1), (1,2)]
        self.discrete_probs = probs  # numpy array [0.909, 0.091]

    def _prepare_continuous_probs(self):
        """预处理 P_continuous：计算max_range内每个点的概率分布"""
        # 1. 生成数据点
        n_vals = np.arange(1, self.max_range + 1)
        m_vals = np.arange(1, self.max_range + 1)
        # 生成网格（一次性矩阵计算，不需要使用两层for循环计算）
        N, M = np.meshgrid(n_vals, m_vals, indexing="ij")

        # 2. 计算函数值
        A, alpha, beta = self.params["A"], self.params["alpha"], self.params["beta"]
        l1, l2, l3 = self.params["lam1"], self.params["lam2"], self.params["lam3"]

        exponent = -(l1 * N + l2 * M + l3 * np.sqrt(N * M))
        weights = A * (N ** alpha) * (M ** beta) * np.exp(exponent)

        # 3. 挖空离散点，将 discrete_map 中存在的点在 continuous 分布中的权重设为 0
        for (dn, dm) in self.discrete_points:
            weights[dn - 1, dm - 1] = 0.0  # -1 因为索引从0开始

        # 4. 归一化
        self.cont_probs_grid = weights / np.sum(weights)

        # 5. 展平以便于 random.choice 使用（二维转一维）
        self.cont_flat_probs = self.cont_probs_grid.flatten()
        self.cont_n_vals = n_vals
        self.cont_m_vals = m_vals

    def P_discrete(self, size=1):
        """
        仅从经验分布中采样
        """
        # 随机选择索引
        indices = np.random.choice(len(self.discrete_points), size=size, p=self.discrete_probs)
        # 根据索引找回 (n, m)
        samples = [self.discrete_points[i] for i in indices]
        return np.array(samples)

    def P_continuous(self, size=1):
        """
        仅从拟合分布中采样 (已排除离散点)
        """
        # 1. 在展平的网格上采样索引（从索引范围 [0, len(self.cont_flat_probs)-1]中，按照给定的概率分布 self.cont_flat_probs，随机抽取 size个索引。）
        flat_indices = np.random.choice(
            len(self.cont_flat_probs),
            size=size,
            p=self.cont_flat_probs
        )

        # 2. 将一维索引还原为二维坐标索引
        n_idx, m_idx = np.unravel_index(flat_indices, self.cont_probs_grid.shape)

        # 3. 映射回真实值
        samples_n = self.cont_n_vals[n_idx]
        samples_m = self.cont_m_vals[m_idx]

        return np.column_stack((samples_n, samples_m))

    def sample(self, size=1):
        """
        混合采样：主入口
        逻辑：以概率 lambda 选择 P_discrete，以概率 (1-lambda) 选择 P_continuous
        """
        # 1. 生成一个掩码，决定每个样本来源
        # True 代表来自 discrete, False 代表来自 continuous
        # np.random.random(size) 会生成一个包含 size 个随机数的一维数组，这些随机数均匀分布在 [0, 1) 区间内。每个随机数对应一个样本，用于决定该样本的来源。
        source_mask = np.random.random(size) < self.lambda_mix
        # 计算离散分布的样本数量(source_mask中true的数量)
        num_discrete = np.sum(source_mask)
        # 计算连续分布的样本数量
        num_continuous = size - num_discrete

        # 2. 初始化结果数组
        results = np.zeros((size, 2), dtype=int)

        # 3. 分别采样并填入
        if num_discrete > 0:
            results[source_mask] = self.P_discrete(size=num_discrete)

        if num_continuous > 0:
            results[~source_mask] = self.P_continuous(size=num_continuous)

        return results


if __name__ == "__main__":
    # 离散分布
    discrete_date = {(1, 2): 15796, (1, 1): 11722}

    # 连续分布
    continuous_params = {
        'A': 391696615.17,
        'alpha': 14.04, 'beta': 17.33,
        'lam1': 1.7308, 'lam2': 3.1599, 'lam3': 12.5264
    }

    # 离散分布权重
    lambda_mix = (15796 + 11722) / (15796 + 11722 + 3394)
    # lambda_mix = 0.5
    sampler = TxInOutSampler(discrete_date, continuous_params, lambda_mix, max_range=5)
    samples = sampler.sample(size=1)
    print(samples)
