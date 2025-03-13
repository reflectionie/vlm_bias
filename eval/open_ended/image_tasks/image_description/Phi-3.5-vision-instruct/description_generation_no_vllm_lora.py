import json
import os
import argparse
from datetime import datetime
from datasets import load_from_disk
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
from tools.sampler import num_sample  # 导入 num_sample 函数
from tools.lora_setting import load_lora_pretrained_model, get_model_name_from_path
from io import BytesIO  # 用于处理字节流图像
import torch
from movefile import move_json_to_output

# 初始化模型和处理器
def initialize_model(model_path, model_base):
    """
    初始化 LoRA 微调的 Phi-3.5-vision-instruct 模型。
    :param model_path: 微调模型的路径。
    :param model_base: 基础模型名称。
    """
    model_name = get_model_name_from_path(model_path)
    processor, model = load_lora_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map="cuda",
        use_flash_attn=True
    )
    return model, processor

def run_inference(model, processor, question: str, image: Image.Image) -> str:
    """
    使用给定的模型和输入进行推理。

    :param model: 初始化的 LoRA 微调模型对象
    :param processor: 模型对应的处理器对象
    :param question: 用户问题（文本）
    :param image: PIL.Image 对象
    :return: 模型生成的文本响应
    """
    placeholder = "<|image_1|>\n"
    messages = [
        {"role": "user", "content": placeholder + question}
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt, images=[image], return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }

    generate_ids = model.generate(
        **inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_args
    )

    # 移除输入的 token
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response.strip()

def load_existing_results(output_file):
    """
    如果结果文件存在，加载已有结果。
    :param output_file: 输出文件路径
    :return: 已有结果的列表和键集合
    """
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
            existing_keys = {(entry["counterfactual_set"], entry["a1a2"]) for entry in results}
    else:
        results = []
        existing_keys = set()
    return results, existing_keys

def process_dataset(dataset_path, field_name, elements, sample_nums, question, model_name, model_path, random_seed=None, custom_suffix=""):
    """
    加载数据集，使用 num_sample 逐条处理图像并进行推理，将结果保存到文件。

    :param dataset_path: 数据集路径
    :param field_name: 要采样的字段名
    :param elements: 需要采样的元素名列表
    :param sample_nums: 每个元素采样的行数，与 elements 等长
    :param question: 用户问题
    :param model_name: 模型名称
    :param model_path: LoRA 模型路径
    :param random_seed: 随机数种子
    :param custom_suffix: 自定义后缀
    """
    model_base = "microsoft/Phi-3.5-vision-instruct"
    model, processor = initialize_model(model_path, model_base)

    # 加载数据集
    ds = load_from_disk(dataset_path)

    # 使用 num_sample 对数据集进行采样
    sampled_ds = num_sample(ds, field_name, elements, sample_nums)

    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 动态生成文件名
    timestamp = datetime.now().strftime("%Y%m%d")
    output_file_name = f"{timestamp}_{model_name}_num_sample_{sum(sample_nums)}{f'_{custom_suffix}' if custom_suffix else ''}.json"
    # 将结果文件保存到与脚本相同的目录
    output_file = os.path.join(script_dir, output_file_name)

    # 加载已有结果
    results, existing_keys = load_existing_results(output_file)

    for entry in tqdm(sampled_ds, desc=f"Processing sampled dataset: {dataset_path}"):
        try:
            # 获取联合键
            counterfactual_set = entry.get("counterfactual_set", "unknown")
            a1a2 = entry.get("a1a2", "unknown")
            entry_key = (counterfactual_set, a1a2)

            # 跳过已处理的数据
            if entry_key in existing_keys:
                continue

            # 加载图像
            image_field = "image"
            if isinstance(entry[image_field], dict):  # 检查是否为字典结构
                if "bytes" in entry[image_field]:
                    image = Image.open(BytesIO(entry[image_field]["bytes"])).convert("RGB")
                elif "path" in entry[image_field]:
                    image_path = entry[image_field]["path"]
                    image = Image.open(image_path).convert("RGB")
                else:
                    raise ValueError("Image field does not contain 'bytes' or 'path'.")
            else:
                raise ValueError("Image field is not a valid structure.")

            # 运行推理
            response = run_inference(model, processor, question, image)

            # 构造结果
            result = {
                "a1_type": entry.get("a1_type", "unknown"),
                "a2_type": entry.get("a2_type", "unknown"),
                "counterfactual_set": counterfactual_set,
                "a1a2": a1a2,
                "prompt": question,
                "response": response,
            }
            results.append(result)
            existing_keys.add(entry_key)

            # 写入文件
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing entry: {e}")

    print(f"Inference results saved to {output_file}.")

if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Run inference on a dataset using a multimodal model.")
    parser.add_argument("--dataset_path", type=str, default="/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/race_gender_test", help="Dataset path.")
    parser.add_argument("--field_name", type=str, default="a1", help="Field name to sample from.")
    parser.add_argument("--elements", nargs="+", default=['White', 'Latino', 'Black', 'Indian', 'Asian', 'Middle Eastern'], help="Elements to sample.")
    parser.add_argument("--sample_nums", nargs="+", type=int, default=[100, 100, 100, 100, 100, 100], help="Number of samples for each element.")
    parser.add_argument("--question", type=str, default="What is the content of this image?", help="Question to ask the model.")
    parser.add_argument("--model_name", type=str, default="Phi-3.5-vision-instruct", help="Model name.")
    parser.add_argument("--model_path", type=str, default="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250117_234514_111111/model_checkpoint-1486_111111/Phi-3.5-vision-instruct-lora", help="Path to the pretrained LoRA model.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--custom_suffix", type=str, default="", help="Custom suffix for the output file.")

    args = parser.parse_args()

    process_dataset(
        dataset_path=args.dataset_path,
        field_name=args.field_name,
        elements=args.elements,
        sample_nums=args.sample_nums,
        question=args.question,
        model_name=args.model_name,
        model_path=args.model_path,
        random_seed=args.random_seed,
        custom_suffix=args.custom_suffix,
    )
    move_json_to_output()
