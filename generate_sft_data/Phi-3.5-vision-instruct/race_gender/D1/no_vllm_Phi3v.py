import json
import os
import argparse
import re
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import torch

def run_phi3v(question: str, image: Image.Image) -> str:
    """
    使用已经初始化的 transformer 模型进行多模态推理。

    :param question: 文本问题
    :param image: PIL.Image 对象
    :return: 模型生成的文本响应
    """
    # 构造多模态指令模板
    

    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"
    inputs = processor(prompt, images=[image], return_tensors="pt")
    # 将所有 tensor 移动到 GPU（cuda:0）
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to("cuda:0")
            
    eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    # 生成参数，max_new_tokens 设置为 300
    generation_args = {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "eos_token_id": eos_token_id,
    }
    
    generation_output = model.generate(**inputs, **generation_args)
    # 解码模型的回答部分（跳过输入的 token）
    generated_tokens = generation_output[0][inputs["input_ids"].shape[1]:]
    
    # 解码生成的 token
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # 去除 prompt 部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):]
    
    return generated_text.strip()

def main(a1_type, a2_type, run_suffix):
    # 加载数据集
    ds = load_from_disk("/net/graphium/storage3/tingyuan/vlm_bias/test_dataset/race_gender_train")
    subset = ds.filter(lambda x: x['a1_type'] == a1_type and x['a2_type'] == a2_type)
    
    # 用于去重，避免重复处理
    processed_combinations = set()
    
    # 生成结果文件名（在文件名前拼接 run_suffix）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_name = f"D1"
    result_file_name = os.path.join(script_dir, f"{run_suffix}_{model_save_name}_{a1_type}_{a2_type}.json")
    
    # 加载已有结果（如果存在）
    try:
        with open(result_file_name, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for res in existing_results:
                processed_combinations.add((res['counterfactual_set'], res['a1a2']))
    except FileNotFoundError:
        existing_results = []
    
    # 定义问题
    # question = "What is the content of this image?"
    question = "Please describe in as much detail as possible the information contained in the image."
    
    # 遍历子集进行推理
    for entry in tqdm(subset, desc=f"Processing a1_type={a1_type}, a2_type={a2_type}", ncols=80):
        combination = (entry['counterfactual_set'], entry['a1a2'])
        if combination in processed_combinations:
            continue
        
        try:
            image_field = "image"
            if isinstance(entry[image_field], Image.Image):
                image = entry[image_field]
            else:
                image_path = entry[image_field]
                image = Image.open(image_path).convert("RGB")
            
            response = run_phi3v(question, image)
            
            new_result = {
                "a1_type": entry['a1_type'],
                "a2_type": entry['a2_type'],
                "counterfactual_set": entry['counterfactual_set'],
                "a1a2": entry['a1a2'],
                "prompt": question,
                "response": response
            }
            
            existing_results.append(new_result)
            processed_combinations.add(combination)
            
            with open(result_file_name, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=4)
        
        except Exception as e:
            print(f"处理条目时出错: {e}")
    
    print(f"推理结果已保存到 {result_file_name} 文件中！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Phi-3.5-vision-instruct 进行多模态推理")
    parser.add_argument("--model_id", type=str, 
                        # required=True,
                        default="/net/graphium/storage3/tingyuan/Phi3-Vision-Finetune/output/run_20250313_000331_000001/model_checkpoint-892_000001/Phi-3.5-vision-instruct-lora",
                        help="模型路径，例如：/net/graphium/storage3/tingyuan/Phi3-Vision-Finetune/output/run_20250313_000331_000001/model_checkpoint-892_000001/Phi-3.5-vision-instruct-lora")
    parser.add_argument("--a1_type", type=str, default="race", help="a1_type 过滤条件")
    parser.add_argument("--a2_type", type=str, default="gender", help="a2_type 过滤条件")
    args = parser.parse_args()
    
    model_id = args.model_id
    # 从 model_id 中提取 run_... 中最后一部分的数字（例如 "000001"）
    import re
    match = re.search(r'run_\d+_\d+_(\d+)', model_id)
    if match:
        run_suffix = match.group(1)
    else:
        run_suffix = "unknown"
    
    # 初始化模型和处理器
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=1)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    main(a1_type=args.a1_type, a2_type=args.a2_type, run_suffix=run_suffix)
