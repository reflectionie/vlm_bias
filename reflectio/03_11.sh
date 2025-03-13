#!/bin/bash

echo "当前工作目录: $(pwd)"
# 定义要替换的数字列表
numbers=("010000" "001000" "000100" "000010" "000001")

# 循环遍历每个数字，依次执行命令
for num in "${numbers[@]}"; do
    echo "正在处理 bias_wlbism_$num 目录..."
    
  
    # 使用绝对路径执行 sample_dataset.py
    PYTHONPATH=. python /net/graphium/storage3/tingyuan/vlm_bias/sft_dataset/race_gender/bias_wlbism_${num}/sample_dataset.py
    
    # 执行 build_sft_pairs.py（执行两次）
    PYTHONPATH=. python /net/graphium/storage3/tingyuan/vlm_bias/sft_dataset/race_gender/bias_wlbism_${num}/build_sft_pairs.py
    
    # 执行 build_sft_dataset.py
    PYTHONPATH=. python /net/graphium/storage3/tingyuan/vlm_bias/sft_dataset/race_gender/bias_wlbism_${num}/build_sft_dataset.py
    
    echo "bias_wlbism_$num 处理完毕。"
    echo "-------------------------------------"
done
