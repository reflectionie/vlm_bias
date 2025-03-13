import json
import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import nltk
import argparse

nltk.download('stopwords')

# 加载停用词
stop_words = set(stopwords.words('english'))

# 定义计算 PMI 的函数
def calculate_pmi(word_freq_cd, word_freq_total, total_words_cd, total_words_t):
    """
    计算 Pointwise Mutual Information (PMI)。
    
    参数：
    - word_freq_cd: 目标组语料（CD）中词的出现频率。
    - word_freq_total: 总语料（T）中词的出现频率。
    - total_words_cd: 目标组语料（CD）的总词数。
    - total_words_t: 总语料（T）的总词数。

    返回值：
    - PMI 值
    """
    p_w_cd = word_freq_cd / total_words_cd
    p_w = word_freq_total / total_words_t
    if p_w > 0:
        return np.log2(p_w_cd / p_w)
    else:
        return 0

# 定义处理 JSON 文件并计算 PMI 的函数
def process_and_calculate_pmi(json_file_path, min_freq=1, pmi_threshold=1, top_k=None):
    """
    处理 JSON 文件，根据社会属性分组并计算 PMI。
    
    输入：
    - json_file_path: JSON 文件路径。
    - min_freq: 最小出现频率（默认值为1）。
    - pmi_threshold: PMI 阈值（默认值为1）。
    - top_k: 每个属性保留的最大词数（默认值为 None，表示不限制）。
    
    输出：
    - 在输入文件同目录下保存一个包含 PMI 计算结果的 CSV 文件，命名为 'pmi_{输入文件名}.csv'。
    """
    # 加载 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 根据 'a1a2' 属性对数据分组
    grouped_data = defaultdict(list)
    for item in data:
        attribute = item['a1a2']  # 通过 'a1a2' 确定社会属性
        response = item['response']  # 提取响应文本
        grouped_data[attribute].append(response)

    # 构建语料库
    total_corpus = []  # 总语料库
    group_corpora = {}  # 按属性分组的子语料库

    for attribute, responses in grouped_data.items():
        group_text = " ".join(responses)  # 将响应文本拼接为单个字符串
        group_words = [word for word in group_text.split() if word.lower() not in stop_words]  # 过滤停用词
        group_corpora[attribute] = group_words
        total_corpus.extend(group_words)

    # 计算词频
    total_word_freq = Counter(total_corpus)  # 总语料的词频
    total_word_count = sum(total_word_freq.values())  # 总词数

    pmi_results = []  # 存储 PMI 结果

    for attribute, words in group_corpora.items():
        group_word_freq = Counter(words)  # 目标组的词频
        group_word_count = sum(group_word_freq.values())  # 目标组的总词数

        attribute_results = []

        for word in group_word_freq:
            if group_word_freq[word] >= min_freq:  # 检查最小出现频率
                pmi_score = calculate_pmi(
                    group_word_freq[word],
                    total_word_freq[word],
                    group_word_count,
                    total_word_count
                )
                if pmi_score >= pmi_threshold:  # 检查 PMI 阈值
                    attribute_results.append({
                        'Attribute': attribute,  # 社会属性
                        'Word': word,  # 词
                        'PMI': pmi_score  # PMI 值
                    })

        # 如果指定了 top_k，则按 PMI 降序保留前 top_k 个词
        if top_k:
            attribute_results = sorted(attribute_results, key=lambda x: x['PMI'], reverse=True)[:top_k]

        pmi_results.extend(attribute_results)

    # 确定输出文件路径
    input_filename = os.path.basename(json_file_path)
    input_dir = os.path.dirname(json_file_path)
    output_filename = f"pmi_{os.path.splitext(input_filename)[0]}.csv"
    output_csv_path = os.path.join(input_dir, output_filename)

    # 保存结果到 CSV 文件
    pmi_df = pd.DataFrame(pmi_results)
    pmi_df = pmi_df.sort_values(by=['Attribute', 'PMI'], ascending=[True, False])  # 按属性和 PMI 值排序
    pmi_df.to_csv(output_csv_path, index=False)  # 保存为 CSV 文件
    print(f"PMI 计算完成。结果已保存到 '{output_csv_path}'。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 文件并计算每组的 PMI 值")
    parser.add_argument("input_json_path", type=str, help="输入 JSON 文件的路径。")
    parser.add_argument("--min_freq", type=int, default=1, help="词的最小出现频率，默认值为 1")
    parser.add_argument("--pmi_threshold", type=float, default=1.0, help="PMI 阈值，默认值为 1")
    parser.add_argument("--top_k", type=int, default=None, help="每组保留的最大词数，默认值为 10")
    args = parser.parse_args()

    process_and_calculate_pmi(
        args.input_json_path,
        min_freq=args.min_freq,
        pmi_threshold=args.pmi_threshold,
        top_k=args.top_k
    )
