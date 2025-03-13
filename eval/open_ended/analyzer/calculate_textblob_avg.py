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
    处理 JSON 文件，对每条记录分别进行情感分析，然后计算平均值，保存结果。

    参数：
        file_path (str): 输入 JSON 文件的路径。

    功能：
        - 从 JSON 文件中提取每条记录的 'response' 字段。
        - 分别对每条记录进行情感分析。
        - 计算所有记录的情感极性和主观性的平均值。
        - 保存每条记录的分析结果和总体平均值。
    """
    results = []
    total_polarity = 0
    total_subjectivity = 0
    count = 0

    # 加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 遍历每条记录并分析情感
    for record in data:
        if 'response' in record and isinstance(record['response'], str):
            text = record['response']
            polarity, subjectivity = analyze_sentiment_textblob(text)
            # 将原始记录内容写入，并附加情感分析结果
            updated_record = {key: value for key, value in record.items() if key != 'response'}
            updated_record.update({
                "polarity": polarity,
                "subjectivity": subjectivity
            })
            results.append(updated_record)
            total_polarity += polarity
            total_subjectivity += subjectivity
            count += 1

    # 计算平均值
    average_polarity = total_polarity / count if count > 0 else 0
    average_subjectivity = total_subjectivity / count if count > 0 else 0

    # 准备输出文件路径
    dir_path, file_name = os.path.split(file_path)
    output_file_name = f"textblob_sentiment_avg_{file_name}"
    output_file_path = os.path.join(dir_path, output_file_name)

    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({
            "average_polarity": average_polarity,
            "average_subjectivity": average_subjectivity,
            "individual_results": results
        }, output_file, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 文件以分别计算每条记录的情感分析结果并计算平均值。")
    parser.add_argument("input_json_path", type=str, help="输入 JSON 文件的路径。")
    args = parser.parse_args()

    process_json_file_with_sentiment(args.input_json_path)
