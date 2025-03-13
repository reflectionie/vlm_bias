import os
import shutil

def move_json_to_output():
    # 获取当前脚本所在的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 定义目标文件夹路径
    output_dir = os.path.join(current_dir, "output")

    # 如果 output 文件夹不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历当前目录下的所有文件
    for file_name in os.listdir(current_dir):
        # 检查文件是否为 JSON 文件
        if file_name.endswith(".json"):
            # 获取文件的完整路径
            source_path = os.path.join(current_dir, file_name)
            # 定义目标路径
            destination_path = os.path.join(output_dir, file_name)

            # 移动文件
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name} -> {output_dir}")

if __name__ == "__main__":
    move_json_to_output()
