import random
import secrets
import time


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


def generate_tx_id(simple=True, index=None):
    """
    生成模拟的交易 ID (TxID)

    :param simple: 如果为 True，生成形如 'tx_001' 的简单 ID (便于调试和肉眼观察)
    :param index: 配合 simple=True 使用，指定序号
    :return: 交易 ID 字符串
    """
    if simple:
        if index is not None:
            return f"tx_{index}"
        else:
            # 使用时间戳后6位作为简易ID
            return f"tx_{int(time.time() * 1000) % 1000000}"

    else:
        # --- 生产级模拟：生成 64 字符的十六进制哈希 ---
        # 方式 1: 完全随机 (速度最快，推荐)
        return secrets.token_hex(32)
