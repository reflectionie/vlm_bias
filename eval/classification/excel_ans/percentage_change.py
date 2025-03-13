import pandas as pd

# 读取Excel文件
file_path = 'bias.xlsx'
df = pd.read_excel(file_path)

# 提取基准值（第一行数据，跳过第一列）
base_values = df.iloc[0, 1:]  # 第二行作为基准值
columns = df.columns[1:]  # 数据列

# 计算变化百分比
percentage_change = pd.DataFrame(index=df.index[1:], columns=columns)  # 创建空表
for col in columns:
    base_value = base_values[col]
    if base_value != 0:
        percentage_change[col] = (df[col].iloc[1:] - base_value) / base_value * 100
    else:
        # 如果基准值为 0，则用 NaN 表示
        percentage_change[col] = None

# 将结果保存到一个新的 Excel 文件
output_file_path = 'percentage_change.xlsx'
percentage_change.insert(0, df.columns[0], df.iloc[1:, 0])  # 保留第一列作为行名
percentage_change.to_excel(output_file_path, index=False)

print(f"变化百分比表已保存为 {output_file_path}")
