import random
from collections import defaultdict
from datasets import Dataset

def balanced_sample(dataset, field_name, sample_ratio, random_seed=None):
    """
    对 Hugging Face 数据集进行采样，使得采样后 `field_name` 的不同取值的个数相同。

    参数：
        dataset (Dataset): 要采样的数据集。
        field_name (str): 要平衡的字段名。
        sample_ratio (float): 采样比例 0 到 1 之间。
        random_seed (int, optional): 随机数种子，确保采样结果可重复。

    返回：
        Dataset: 采样后的数据集。
    """
    # 设置随机数种子
    if random_seed is not None:
        random.seed(random_seed)

    # 检查采样比例是否合理
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio 必须在 (0, 1] 范围内。")

    # 将数据按 `field_name` 的取值分组
    grouped_data = defaultdict(list)
    for idx, example in enumerate(dataset):
        value = example[field_name]
        grouped_data[value].append(idx)

    # 找到每组的最小样本数量
    min_group_size = min(len(indices) for indices in grouped_data.values())

    # 自动调整采样比例，以确保每组样本数量足够
    max_possible_ratio = min_group_size / len(dataset)
    effective_ratio = min(sample_ratio, max_possible_ratio)

    # 打印警告，如果未达到设定的采样比例
    if effective_ratio < sample_ratio:
        print(f"警告：采样比例设置为 {sample_ratio}，但实际有效比例为 {effective_ratio}，已自动调整。")

    # 计算每组实际采样数量
    sample_size_per_group = int(len(dataset) * effective_ratio / len(grouped_data))
    group_len = len(grouped_data)
    print(f"Sampler: sample_size_per_group: {sample_size_per_group}, group_len: {group_len}")

    # 从每组中随机采样
    sampled_indices = []
    for indices in grouped_data.values():
        sampled_indices.extend(random.sample(indices, min(sample_size_per_group, len(indices))))

    # 根据采样的索引创建新的数据集
    sampled_dataset = dataset.select(sampled_indices)

    # 打印原始数据集和采样后数据集的大小
    print(f"原始数据集大小: {len(dataset)}，采样后数据集大小: {len(sampled_dataset)}")

    return sampled_dataset


# 示例用法
# ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")
# sampled_ds = balanced_sample(ds, 'a2', 0.1, random_seed=42)
# print(sampled_ds)


from collections import Counter
from datasets import Dataset, load_from_disk
import pandas as pd

def ratio_sample(dataset, column_name, elements, sample_ratio):
    """
    对 HuggingFace 数据集的特定列中的指定元素进行采样。

    Args:
        dataset (Dataset): 输入的数据集
        column_name (str): 需要采样的列名 ('a1' 或 'a2')
        elements (list): 需要采样的元素名列表
        sample_ratio (float): 采样比例 (0-1)

    Returns:
        Dataset: 采样后数据集
    """
    # 打印采样前的统计信息
    def print_stats(title, data, columns):
        print(f"{title}:")
        for col in columns:
            counts = Counter(data[col])
            print(f"[{col}] 属性值分布:")
            for value, count in counts.items():
                print(f"值: {value}, 数量: {count}")

    print_stats("采样前数据集统计信息", dataset, ['a1', 'a2'])

    # 筛选需要采样的行
    sampled_dataframes = []

    for element in elements:
        element_data = dataset.filter(lambda x: x[column_name] == element)
        other_column = 'a1' if column_name == 'a2' else 'a2'

        # 确保其他列的值分布均匀
        counts = Counter(element_data[other_column])
        min_count = min(counts.values())
        target_count = int(min_count * sample_ratio)

        grouped_data = element_data.to_pandas().groupby(other_column)
        sampled_groups = [group.sample(n=min(len(group), target_count), random_state=42) for _, group in grouped_data]
        sampled_dataframes.append(pd.concat(sampled_groups, ignore_index=True))

    # 保留未被采样的其他数据
    remaining_data = dataset.filter(lambda x: x[column_name] not in elements)
    remaining_df = remaining_data.to_pandas()

    # 合并所有采样的结果和未采样数据
    sampled_df = pd.concat(sampled_dataframes + [remaining_df], ignore_index=True)

    # 将采样结果转换回 HuggingFace 数据集
    sampled_dataset = Dataset.from_pandas(sampled_df)

    # 打印采样后的统计信息
    print_stats("采样后数据集统计信息", sampled_dataset, ['a1', 'a2'])

    return sampled_dataset


def num_sample(dataset, column_name, elements, sample_nums):
    """
    对 HuggingFace 数据集的特定列中的指定元素分别采样固定数量的数据，并保证另一列的类别分布平衡。

    Args:
        dataset (Dataset): 输入的数据集
        column_name (str): 需要采样的列名 ('a1' 或 'a2')
        elements (list): 需要采样的元素名列表
        sample_nums (list): 每个元素采样的行数，与 elements 等长

    Returns:
        Dataset: 采样后数据集
    """
    if len(elements) != len(sample_nums):
        raise ValueError("elements 和 sample_nums 的长度必须相等！")

    # 打印采样前的统计信息
    def print_stats(title, data, columns):
        print(f"{title}:")
        for col in columns:
            counts = Counter(data[col])
            print(f"[{col}] 属性值分布:")
            for value, count in counts.items():
                print(f"值: {value}, 数量: {count}")

    print_stats("采样前数据集统计信息", dataset, ['a1', 'a2'])

    # 筛选需要采样的行
    sampled_dataframes = []

    for element, sample_num in zip(elements, sample_nums):
        if sample_num == 0:
            # 跳过采样数为 0 的元素
            continue

        element_data = dataset.filter(lambda x: x[column_name] == element)
        other_column = 'a1' if column_name == 'a2' else 'a2'

        # 确保另一列的值分布平衡
        grouped_data = element_data.to_pandas().groupby(other_column)
        total_count = sum(len(group) for _, group in grouped_data)
        sampled_groups = []

        for _, group in grouped_data:
            # 按比例分配采样数量
            proportion = len(group) / total_count
            group_sample_num = int(round(proportion * sample_num))
            sampled_groups.append(group.sample(n=min(len(group), group_sample_num), random_state=42))

        sampled_dataframes.append(pd.concat(sampled_groups, ignore_index=True))

    # 保留未被采样的其他数据
    remaining_data = dataset.filter(lambda x: x[column_name] not in elements)
    remaining_df = remaining_data.to_pandas()

    # 合并所有采样的结果和未采样数据
    sampled_df = pd.concat(sampled_dataframes + [remaining_df], ignore_index=True)

    # 将采样结果转换回 HuggingFace 数据集
    sampled_dataset = Dataset.from_pandas(sampled_df)

    # 打印采样后的统计信息
    print_stats("采样后数据集统计信息", sampled_dataset, ['a1', 'a2'])

    return sampled_dataset






# 加载数据集
# data_path = "/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test"
# dataset = load_from_disk(data_path)

# # 调用函数对'a2'列中的'male'元素进行80%的采样
# sampled_dataset = ratio_sample(dataset, 'a1', ['young','skinny'], 0.5)


