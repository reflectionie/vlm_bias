from collections import Counter
from operator import itemgetter
import spacy
from spacy.matcher import Matcher

class WordExtraction:
    def __init__(self, word_types=None):
        """
        初始化一个单词提取器，用于从文本中提取特定词性。

        参数：
            word_types: list，指定需要提取的词性，如 ['noun', 'adj', 'verb']。
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        patterns = []

        for word_type in word_types:
            if word_type == 'noun':
                patterns.append([{'POS':'NOUN'}])
            elif word_type == 'adj':
                patterns.append([{'POS':'ADJ'}])
            elif word_type == 'verb':
                patterns.append([{"POS": "VERB"}])
        self.matcher.add("word_extraction", patterns)

    def extract_words(self, text):
        """
        从输入文本中提取指定词性的单词。

        参数：
            text: str，输入文本。

        返回：
            list，包含提取的单词。
        """
        doc = self.nlp(text)
        matches = self.matcher(doc)
        return [doc[start:end].text for _, start, end in matches]

def calculate_dict(category1_list, category2_list):
    """
    计算两个类别的词频，并补齐缺失的键值。
    """
    counter1 = Counter(category1_list)
    counter2 = Counter(category2_list)
    all_keys = set(counter1.keys()).union(set(counter2.keys()))
    for key in all_keys:
        counter1.setdefault(key, 0)
        counter2.setdefault(key, 0)
    return counter1, counter2

def odds_ratio(cat1_dict, cat2_dict, topk=50, threshold=20, very_small_value=1e-5):
    """
    根据两个类别的词频计算 Odds Ratio（OR）。

    OR 的意义：
        OR 衡量了某个词在类别2中出现的相对可能性与类别1的对比。
        - 如果 OR > 1，说明该词更倾向于出现在类别2中。
        - 如果 OR < 1，说明该词更倾向于出现在类别1中。
        - 如果 OR 接近 1，说明该词在两个类别中的出现频率接近。

    参数：
        cat1_dict: dict，第一个类别的词频统计。
        cat2_dict: dict，第二个类别的词频统计。
        topk: int，返回的 OR 排名前 k 的键值。
        threshold: int，筛选的最小频次阈值。
        very_small_value: float，防止除零的小值。

    返回：
        top_positive: dict，OR 最大的 topk 键值对。
        top_negative: dict，OR 最小的 topk 键值对。
    """
    odds_ratio = {}
    total_num_cat1 = sum(cat1_dict.values())
    total_num_cat2 = sum(cat2_dict.values())

    for key in cat1_dict.keys():
        cat1_count = max(cat1_dict[key], very_small_value)
        cat2_count = max(cat2_dict[key], very_small_value)
        non_cat1_count = total_num_cat1 - cat1_count
        non_cat2_count = total_num_cat2 - cat2_count

        # 仅计算满足阈值的词汇
        if cat1_count >= threshold and cat2_count >= threshold:
            odds_ratio[key] = round((cat2_count / cat1_count) / (non_cat2_count / non_cat1_count), 2)

    sorted_odds = sorted(odds_ratio.items(), key=itemgetter(1), reverse=True)
    return dict(sorted_odds[:topk]), dict(sorted_odds[-topk:])

# 示例用法
if __name__ == "__main__":
    # 示例文本
    category1_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A happy dog is a friendly companion."
    ]
    category2_texts = [
        "The strong bold eagle flies over the vast mountains.",
        "A smart cat is a curious explorer."
    ]

    # 使用单词提取器提取词汇
    extractor = WordExtraction(word_types=['noun', 'adj'])
    category1_words = []
    category2_words = []

    for text in category1_texts:
        category1_words.extend(extractor.extract_words(text))

    for text in category2_texts:
        category2_words.extend(extractor.extract_words(text))

    # 计算词频
    cat1_dict, cat2_dict = calculate_dict(category1_words, category2_words)

    # 计算 OR
    top_positive, top_negative = odds_ratio(cat1_dict, cat2_dict, threshold=1)
    print("Top Positive OR:", top_positive)
    print("Top Negative OR:", top_negative)
