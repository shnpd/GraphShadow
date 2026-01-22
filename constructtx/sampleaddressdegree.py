import numpy as np


class AddressCentralDegreeSampler:
    def __init__(self, param, min_x=1, max_x=10000):
        self.param = param
        self.min_x = min_x
        self.max_x = max_x
        # 预计算概率表
        self._prepare_probs()

    def _prepare_probs(self):
        # 1. 生成所有可能的中心度值 x [min_x, ..., max_x]
        x_vals = np.arange(self.min_x, self.max_x + 1)


        # 2. 计算拟合函数值
        A, alpha = self.param["A"],self.param["alpha"]
        weights = A * x_vals ** alpha

        # 3. 归一化得到概率分布 (PMF)
        self.probs = weights / np.sum(weights)
        self.x_vals = x_vals

    def sample(self, size=1):
        """
        执行采样
        :param size: 采样数量
        :return: 采样得到的中心度数组
        """
        # np.random.choice 在 1D 情况下非常高效
        # replace=True 表示允许重复抽样 (符合独立同分布)
        return np.random.choice(self.x_vals, size=size, p=self.probs)

if __name__ == "__main__":
    params = {
        'A': 397541.13,
        'alpha': -5.39
    }

    sampler = AddressCentralDegreeSampler(params,min_x=2,max_x=100)
    samples = sampler.sample(100)
    print(samples)