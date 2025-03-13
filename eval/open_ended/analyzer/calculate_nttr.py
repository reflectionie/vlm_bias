import os
import json
import math
import argparse
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk

# 下载 'punkt' 资源
nltk.download('punkt')

def calculate_normalized_ttr(text, case_sensitive=False):
    """
    计算给定文本的标准化类型-词汇比 (NTTR)。

    NTTR 是一种词汇多样性度量，它通过将文本中的唯一词汇数（类型）除以词汇总数的平方根进行归一化。
    这种方法可以在评估词汇多样性时减少文本长度的影响。

    参数：
        text (str): 要计算 NTTR 的输入文本。
        case_sensitive (bool): 是否区分大小写。

    返回：
        float: 计算得到的 NTTR 值。较高的值表示更高的词汇多样性。
        对于大多数文本，NTTR 值通常介于 0 和 1 之间，但在词汇极其丰富或文本非常短的情况下可能超过 1。
    """
    if not case_sensitive:
        text = text.lower()
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    if total_tokens == 0:  # 避免除以零
        return 0
    freq_dist = FreqDist(tokens)
    num_types = len(freq_dist)
    normalized_ttr = num_types / math.sqrt(total_tokens)
    return normalized_ttr

def process_json_file(file_path):
    """
    处理 JSON 文件，将所有数据作为一个整体计算标准化类型-词汇比 (NTTR)，并保存结果。

    参数：
        file_path (str): 输入 JSON 文件的路径。
    """
    combined_text = ""

    # 加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 合并所有记录中的 'response' 字段文本
    for record in data:
        if 'response' in record and isinstance(record['response'], str):
            combined_text += record['response'] + " "

    # 计算整体的 NTTR
    nttr = calculate_normalized_ttr(combined_text)

    # 准备输出文件路径
    dir_path, file_name = os.path.split(file_path)
    output_file_name = f"nttr_{file_name}"
    output_file_path = os.path.join(dir_path, output_file_name)

    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"normalized_ttr": nttr}, output_file, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 文件以计算整体标准化类型-词汇比 (NTTR)。")
    parser.add_argument("input_json_path", type=str, help="输入 JSON 文件的路径。")
    args = parser.parse_args()

    process_json_file(args.input_json_path)
