import random


def generate_random_address():
    """
    生成模拟的区块链地址字符串
    :return: 随机地址字符串
    """
    # 预定义 Base58 字符集 (去掉了容易混淆的 0, O, I, l)
    BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    # 模拟比特币 P2PKH 地址: 以 '1' 开头，长度通常为 26-35 位
    # 使用 secrets 库比 random 更适合生成唯一 ID，虽然在模拟中 random 也够用
    # 随机长度 33 或 34 是最常见的
    length = 33
    # random.choices 在 Python 3.6+ 非常快
    suffix = ''.join(random.choices(BASE58_ALPHABET, k=length))
    return f"1{suffix}"

