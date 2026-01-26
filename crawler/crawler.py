import requests
import json
import random
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import sys
import os

def fetch_block_transactions(block_number):
    """
    调用 rawblock 接口，一次性获取该区块的所有交易详情。
    """
    all_transactions = []
    url = f"https://blockchain.info/rawblock/{block_number}?cors=true"
    resp = requests.get(url)
    resp.raise_for_status()
    block_data = resp.json()

    tx_list = block_data.get("tx", [])
    print(f"区块 {block_number}（{block_number}）共含 {len(tx_list)} 笔交易，开始处理...")
    for tx in tx_list:
        all_transactions.append(extract_transaction_field(tx))
    return all_transactions

def extract_transaction_field(tx):
    """
    提取交易中的关键字段，只保留交易哈希，输入地址，输出地址
    """
    newTx = {
        'hash': tx.get('hash', ''),
        'input_addrs': [],
        'output_addrs': []
    }

    # 提取输入地址
    inputs = tx.get('inputs', [])
    for input_item in inputs:
        prev_out = input_item.get('prev_out', [])
        addr = prev_out.get('addr', '')
        if addr:
            newTx['input_addrs'].append(addr)

    # 提取输出地址
    outputs = tx.get('out', [])
    for output_item in outputs:
        addr = output_item.get('addr', '')
        if addr:
            newTx['output_addrs'].append(addr)
    return newTx



def save_block_transaction(startId, endId):
    all_blocks = []
    for i in range(startId, endId):
        all_blocks.append({
            'number': str(i),
            'link': f"/explorer/blocks/btc/{i}"
        })
    MAX_RETRIES = 5
    # ============================== 交易数据获取（改为 API 一次性获取） ==============================
    for idx, blk in enumerate(all_blocks, start=1):
        file_path = f"../dataset/transactions_block_{blk['number']}.json"
        # 检查文件是否存在
        if os.path.exists(file_path):
            print(f"{file_path} 文件已存在，跳过获取。")
            continue

        # === 开始重试逻辑 ===
        success = False
        for attempt in range(MAX_RETRIES):
            try:
                # 尝试获取数据
                all_transactions = fetch_block_transactions(blk['number'])
                # 每个区块处理完都保存一次，防止丢失
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(all_transactions, f, ensure_ascii=False)
                print(f"[{idx}/{len(all_blocks)}] 已保存至 transactions_block_{blk['number']}.json")
                # 成功后标记并退出重试循环
                success = True
                break
            except Exception as e:
                # 捕获错误，打印提示
                wait_time = (attempt + 1) * 5  # 失败次数越多，等待时间越长 (5s, 10s, 15s...)
                print(f"⚠️ 处理区块 {blk['number']} 失败 (第 {attempt + 1}/{MAX_RETRIES} 次重试): {e}")
                print(f"   将在 {wait_time} 秒后重试...")
                time.sleep(wait_time)
        # === 重试多次依然失败的处理 ===
        if not success:
            print(f"❌ 严重错误：区块 {blk['number']} 在重试 {MAX_RETRIES} 次后依然失败，程序终止。")
            # 建议直接 break 停止程序，防止产生大量断点；或者记录到 log 文件后 continue
            break
        time.sleep(random.uniform(1, 3))

def get_blockId_list():
    global all_blocks, f
    # ============================== 区块列表爬取 ==============================
    blocks_base_url = "https://www.blockchain.com/explorer/blocks/btc?page="
    all_blocks = []
    max_pages = 1  # 根据需要调整要爬的区块列表页数
    for page in range(1, max_pages + 1):
        print(f"正在爬取区块列表第 {page} 页...")
        driver.get(f"{blocks_base_url}{page}")
        time.sleep(8)
        # 通过CSS选择器拿到区块链接
        block_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/explorer/blocks/btc/"]')
        # 每个元素会重复筛选，只取前一半
        block_links = block_links[:len(block_links) // 2]
        print(f"找到 {len(block_links)} 个区块链接")
        for link in block_links:
            # 解析href，href示例：/explorer/blocks/btc/927931
            href = link.get_attribute('href')
            block_number = href.split('/')[-1]
            all_blocks.append({
                'number': block_number,
                'link': href
            })
        time.sleep(random.uniform(5, 10))
    print(f"共爬取到 {len(all_blocks)} 个区块信息")
    with open("blocks_info.json", "w", encoding="utf-8") as f:
        json.dump(all_blocks, f, ensure_ascii=False, indent=4)
    print("区块列表已保存至 blocks_info.json")
    driver.quit()
    if not all_blocks:
        print("未爬取到任何区块，退出。")
        sys.exit(1)


def init_parameter():
    global driver
    sys.stdout.reconfigure(encoding='utf-8')  # 适用于 Python 3.7+
    # ============================== 浏览器配置 ==============================
    proxy = "socks5://127.0.0.1:7890"  # 代理地址（如有需要）
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # 去掉注释可开启无头模式
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument(f'--proxy-server={proxy}')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    service = Service(executable_path=r"/Users/shn/Desktop/AntiGraph/chromedriver-mac-arm64/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1920, 1080)


if __name__ == "__main__":
    init_parameter()

    # get_blockId_list()

    # save_block_transaction(927000, 927001)

# # 最终以行分隔格式保存所有交易
# with open("all_blocks_transactions_lines.json", "w", encoding="utf-8") as f:
#     for tx in all_transactions:
#         f.write(json.dumps(tx, ensure_ascii=False) + "\n")
# print("所有交易（行分隔）已保存至 all_blocks_transactions_lines.json")
