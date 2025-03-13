#!/bin/bash

# 获取当前日期和时间
current_datetime=$(date +"%Y-%m-%d_%H-%M-%S")

# 检查是否提供了 Python 文件名
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <python_file> [args...]"
  exit 1
fi

# 获取 Python 文件名和基础名
python_file=$1
python_base_name=$(basename "$python_file" .py)

# 获取 Python 文件所在的目录
python_dir=$(dirname "$python_file")

# 创建 log 文件夹（如果不存在）
log_dir="$python_dir/log"
mkdir -p "$log_dir"

# 设置日志文件路径
log_file="$log_dir/${current_datetime}_${python_base_name}.log"

# 移除第一个参数（Python 文件名），其余参数作为 Python 文件的命令行参数
shift

# 将运行命令写入日志文件
{
  echo "Starting script: $python_file"
  echo "Arguments: $@"
  echo "Log file: $log_file"
  echo "PYTHONPATH: /net/papilio/storage7/tingyuan/llama/bias/vlm_bias/"
} > "$log_file"

# 打印调试信息
cat "$log_file"

# 运行 Python 脚本，确保实时写入日志
PYTHONPATH=/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/ \
python -u "$python_file" "$@" 2>&1 | stdbuf -oL tee -a "$log_file"

# 检查运行状态
if [ $? -ne 0 ]; then
  echo "Python script failed. Check the log file: $log_file"
  exit 1
fi

# 提示完成
echo "Log saved to $log_file"
