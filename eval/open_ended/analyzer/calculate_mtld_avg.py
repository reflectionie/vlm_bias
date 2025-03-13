import os
import json
import re
import argparse
from nltk.corpus import stopwords
from lexical_diversity import lex_div as ld
import nltk

# 下载必要资源
nltk.download('stopwords')

def calculate_mtld(text, remove_stopwords=False):
    """
    计算给定文本的 MTLD（Measure of Textual Lexical Diversity）得分。

    MTLD 取值范围：
        - 较高值：指示文本词汇多样性较高，词汇重复较少。
        - 较低值：指示文本词汇多样性较低，词汇重复较多。

    参数：
        text (str): 输入文本。
        remove_stopwords (bool): 是否移除停用词，默认为 False。

    返回：
        float: 计算的 MTLD 值。
    """
    # 转为小写
    text_lower = text.lower()

    # 去除标点符号
    text_clean = re.sub(r'[^\w\s]', '', text_lower)

    # 拆分为单词
    words = text_clean.split()

    # 可选：移除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

    # 计算 MTLD
    mtld_score = ld.mtld(words)
    return mtld_score

def process_json_file(file_path, remove_stopwords=False):
    """
    处理 JSON 文件，对每条记录的 'response' 字段分别计算 MTLD，最后取平均值，并保存结果。

    参数：
        file_path (str): 输入 JSON 文件的路径。
        remove_stopwords (bool): 是否移除停用词。
    """
    mtld_scores = []

    # 加载 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 对每条记录的 'response' 字段分别计算 MTLD
    for record in data:
        if 'response' in record and isinstance(record['response'], str):
            response_text = record['response']
            mtld_score = calculate_mtld(response_text, remove_stopwords=remove_stopwords)
            mtld_scores.append(mtld_score)

    # 计算平均 MTLD
    if mtld_scores:
        average_mtld = sum(mtld_scores) / len(mtld_scores)
    else:
        average_mtld = 0.0

    # 准备输出文件路径
    dir_path, file_name = os.path.split(file_path)
    output_file_name = f"mtld_avg_{file_name}"
    output_file_path = os.path.join(dir_path, output_file_name)

    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump({"average_mtld": average_mtld, "individual_mtlds": mtld_scores}, output_file, ensure_ascii=False, indent=4)

    print(f"结果已保存到 {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 JSON 文件以计算每条记录的 MTLD 并取平均值。")
    parser.add_argument("input_json_path", type=str, help="输入 JSON 文件的路径。")
    parser.add_argument("--remove_stopwords", action="store_true", help="是否移除停用词。")
    args = parser.parse_args()

    process_json_file(args.input_json_path, remove_stopwords=args.remove_stopwords)
