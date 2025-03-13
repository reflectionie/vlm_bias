import json
import os
import sys
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# 确保下载 VADER 词典
nltk.download('vader_lexicon')

"""
情感分析字段解释：
- neg（negative）：负面情感的强度分数，范围为 [0, 1]。值越大，表示负面情感越强。
- neu（neutral）：中性情感的强度分数，范围为 [0, 1]。值越大，表示中性情感越强。
- pos（positive）：正面情感的强度分数，范围为 [0, 1]。值越大，表示正面情感越强。
- compound：综合情感得分，范围为 [-1, 1]。
  - 复合得分通过标准化算法综合了 neg、neu 和 pos 的值。
  - 复合得分大于 0.05 表示整体情感偏正面；
    小于 -0.05 表示整体情感偏负面；
    介于 -0.05 和 0.05 之间表示中性情感。
"""

def analyze_sentiment(input_file):
    # 加载输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化 VADER 分析器
    analyzer = SentimentIntensityAnalyzer()

    # 存储每条数据的情感分数
    results = []
    total_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}

    for item in data:
        text = item.get('response', '')
        scores = analyzer.polarity_scores(text)

        # 创建一个新的字典，移除 response 字段，并添加 sentiment 字段
        filtered_item = {k: v for k, v in item.items() if k != 'response'}
        filtered_item['sentiment'] = scores
        results.append(filtered_item)

        # 累加分数
        for key in total_scores:
            total_scores[key] += scores[key]

    # 计算平均分数
    num_items = len(data)
    average_scores = {key: (value / num_items) for key, value in total_scores.items()}

    # 输出结果文件路径
    directory, filename = os.path.split(input_file)
    output_file = os.path.join(directory, f"VADER_{filename}")

    # 保存结果到文件，将 average_scores 写在最前面
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {"average_scores": average_scores, "results": results},
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"情感分析结果已保存至: {output_file}")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("用法: python analyze_sentiment.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在。")
        sys.exit(1)

    # 执行情感分析
    analyze_sentiment(input_file)
