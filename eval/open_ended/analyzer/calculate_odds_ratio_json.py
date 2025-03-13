import json
import os
import sys
from collections import Counter
from odds_ratio import WordExtraction, odds_ratio

def update_dict(json_file, word_extractor, target_dict):
    """
    逐条读取 JSON 文件，并提取 response 字段中的单词，更新词频字典。

    参数：
        json_file: str，JSON 文件路径。
        word_extractor: WordExtraction，用于提取单词。
        target_dict: Counter，用于累计词频。
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)  # 解析整个 JSON 文件
            for entry in data:  # 假设 JSON 文件是一个数组
                if "response" in entry:
                    words = word_extractor.extract_words(entry["response"])
                    target_dict.update(words)
        except json.JSONDecodeError:
            print("JSONDecodeError: 文件格式有误，无法解析。")
            return


def save_results(results, output_file):
    """
    将结果保存到指定文件。

    参数：
        results: dict，结果数据。
        output_file: str，输出文件路径。
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def main(json1_path, json2_path, topk=50, threshold=20):
    """
    主函数，用于计算两个 JSON 文件之间的 odds ratio。

    参数：
        json1_path: str，第一个 JSON 文件路径。
        json2_path: str，第二个 JSON 文件路径。
        topk: int，返回的 OR 排名前 k 的键值。
            - 表示仅输出 OR 最大和最小的前 k 个词汇。
            - 例如 topk=10，将输出 OR 最大和最小的 10 个词汇。
        threshold: int，筛选的最小频次阈值。
            - 仅统计在两个类别中均出现至少 threshold 次的词汇。
            - 例如 threshold=5，仅统计出现次数不少于 5 次的词汇，忽略低频词。
    """
    # 初始化词频字典和单词提取器
    cat1_dict = Counter()
    cat2_dict = Counter()
    extractor = WordExtraction(word_types=['noun', 'adj'])

    # 更新词频字典
    print(f"正在处理 {json1_path}...")
    update_dict(json1_path, extractor, cat1_dict)

    print(f"正在处理 {json2_path}...")
    update_dict(json2_path, extractor, cat2_dict)

    # 计算 OR
    top_positive, top_negative = odds_ratio(cat1_dict, cat2_dict, topk=topk, threshold=threshold)

    # 准备结果
    results = {
        "Top Positive OR": top_positive,
        "Top Negative OR": top_negative
    }

    # 构造输出文件路径
    json1_name = os.path.splitext(os.path.basename(json1_path))[0]
    json2_name = os.path.splitext(os.path.basename(json2_path))[0]
    output_dir = os.path.dirname(json1_path)
    output_file = os.path.join(output_dir, f"odds_ratio_{json1_name}_{json2_name}.json")

    # 保存结果
    save_results(results, output_file)
    print(f"结果已保存到 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python calculate_odds_ratio_json.py <json1_path> <json2_path> [--topk <int>] [--threshold <int>]")
        sys.exit(1)

    json1_path = sys.argv[1]
    json2_path = sys.argv[2]

    # 默认值
    topk = 50
    threshold = 1

    # 解析额外参数
    for i, arg in enumerate(sys.argv):
        if arg == '--topk' and i + 1 < len(sys.argv):
            topk = int(sys.argv[i + 1])
        if arg == '--threshold' and i + 1 < len(sys.argv):
            threshold = int(sys.argv[i + 1])

    main(json1_path, json2_path, topk=topk, threshold=threshold)
