import os
import json
import re
import argparse
from textblob import TextBlob

def analyze_sentiment_textblob(text):
    """
    使用 TextBlob 计算文本的情感极性和主观性。

    参数：
        text (str): 输入文本。

    返回：
        tuple: 极性 (polarity) 和主观性 (subjectivity)。

        极性 (polarity)：
            - 表示文本的情感倾向。
            - 取值范围为 [-1.0, 1.0]。
            - -1.0 表示完全负面情感，1.0 表示完全正面情感，0 表示中性。

        主观性 (subjectivity)：
            - 表示文本的主观性程度。
            - 取值范围为 [0.0, 1.0]。
            - 0.0 表示完全客观，1.0 表示完全主观。
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

def process_json_file_with_sentiment(file_path):
    """
    处理 JSON 文件，将所有记录拼接为一个文本并进行情感分析，保存整体结果。

    参数：
        file_path (str): 输入 JSON 文件的路径。

    功能：
        - 从 JSON 文件中提取每条记录的 'response' 字段。
        - 将所有文本拼接为一个整体进行分析。
        - 使用 TextBlob 分析文本的整体情感。
        - 分析结果包括：
            - 极性 (combined_polarity)：整体文本的情感倾向，取值范围 [-1.0, 1.0]。
            - 主观性 (combined_subjectivity)：整体文本的主观性程度，取值范围 [0.0, 1.0]。
    """
    combined_text = ""

    # 加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 拼接所有记录中的 'response' 字段文本
    for record in data:
        if 'response' in record and isinstance(record['response'], str):
            combined_text += record['response'] + " "

    # 进行整体情感分析
    polarity, subjectivity = analyze_sentiment_textblob(combined_text)

    # 准备输出文件路径
    dir_path, file_name = os.path.split(file_path)
    output_file_name = f"sentiment_{file_name}"
    output_file_path = os.path.join(dir_path, output_file_name)

    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({
            "combined_polarity": polarity,
            "combined_subjectivity": subjectivity
        }, output_file, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 文件以计算整体情感分析结果。")
    parser.add_argument("input_json_path", type=str, help="输入 JSON 文件的路径。")
    args = parser.parse_args()

    process_json_file_with_sentiment(args.input_json_path)
