import json
import os
import argparse
import re
import math
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import torch
import gc

def run_phi3v_batch(questions, images, model, processor, tokenizer):
    """
    批量多模态推理函数：
    :param questions: list[str]，每个样本的文本问题
    :param images: list[PIL.Image]，与 questions 对应的一批图像
    :param model: 已加载到 GPU 上的 CausalLM 模型
    :param processor: 对应模型的 Processor（能处理图像和文本），仅支持单样本文本输入
    :param tokenizer: 对应模型的 Tokenizer
    :return: list[str]，与输入一一对应的生成回答
    """
    # 对于每个样本，单独调用 processor 得到输入，然后手动拼接成一个 batch
    batch_inputs = []
    # 构造每个样本的 prompt
    prompts = [
        f"<|user|>\n<|image_1|>\n{q}<|end|>\n<|assistant|>\n"
        for q in questions
    ]
    
    for prompt, img in zip(prompts, images):
        # 每次只处理单个文本和单个图像，注意 images 参数传入一个列表
        inputs = processor(prompt, images=[img], return_tensors="pt")
        batch_inputs.append(inputs)
    
    # 手动将各个样本的输入拼接成 batch
    collated_inputs = {}
    for key in batch_inputs[0]:
        # 假设每次返回的 tensor 第一维大小均为1，则直接 cat
        collated_inputs[key] = torch.cat([x[key] for x in batch_inputs], dim=0)
    
    # 将所有 tensor 移动到 GPU（cuda:0）
    for key in collated_inputs:
        if torch.is_tensor(collated_inputs[key]):
            collated_inputs[key] = collated_inputs[key].to("cuda:0")
    
    # 获取 <|end|> 的 token_id
    eos_token_id = tokenizer.convert_tokens_to_ids("<|end|>")
    
    generation_args = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "do_sample": True,
        "eos_token_id": eos_token_id,
    }
    
    with torch.no_grad():
        generation_outputs = model.generate(**collated_inputs, **generation_args)
    
    # 解码每个样本的生成结果（跳过输入的 token）
    responses = []
    input_len = collated_inputs["input_ids"].shape[1]
    for i, output_ids in enumerate(generation_outputs):
        generated_tokens = output_ids[input_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # 如果生成文本中包含 prompt，则去掉（一般情况不应该再包含）
        if text.startswith(prompts[i]):
            text = text[len(prompts[i]):]
        responses.append(text.strip())
    
    return responses

def main(a1_type, a2_type, run_suffix, batch_size=2):
    # 1. 加载数据集并根据 a1_type, a2_type 做过滤
    ds = load_from_disk("/net/graphium/storage3/tingyuan/vlm_bias/test_dataset/race_gender_train")
    subset = ds.filter(lambda x: x['a1_type'] == a1_type and x['a2_type'] == a2_type)
    
    # 2. 用于去重，避免重复处理
    processed_combinations = set()

    # 3. 生成结果文件名
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_name = "D1"
    result_file_name = os.path.join(script_dir, f"{run_suffix}_{model_save_name}_{a1_type}_{a2_type}.json")

    # 4. 尝试加载已有结果（如果存在），把已处理的 combo 记下来
    try:
        with open(result_file_name, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for res in existing_results:
                processed_combinations.add((res['counterfactual_set'], res['a1a2']))
    except FileNotFoundError:
        existing_results = []

    # 5. 定义要问的问题
    question = "Please describe in as much detail as possible the information contained in the image."

    # 6. 通过分批方式对 subset 进行遍历
    total_len = len(subset)
    num_batches = math.ceil(total_len / batch_size)

    for batch_idx in tqdm(range(num_batches), desc="Processing", ncols=80):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_len)

        # 当前批次数据（使用 select 按索引取子集）
        batch_entries = subset.select(range(start, end))
        if len(batch_entries) == 0:
            continue

        # 筛选未处理过的样本在本批次内的索引
        valid_indices = []
        for i_in_batch, entry in enumerate(batch_entries):
            combo = (entry['counterfactual_set'], entry['a1a2'])
            if combo not in processed_combinations:
                valid_indices.append(i_in_batch)

        if len(valid_indices) == 0:
            continue
        
        # 7. 收集本批次需要处理的图像和文本
        batch_images = []
        batch_questions = []
        
        for i_in_batch in valid_indices:
            entry = batch_entries[i_in_batch]
            image_field = "image"
            if isinstance(entry[image_field], Image.Image):
                img = entry[image_field]
            else:
                img = Image.open(entry[image_field]).convert("RGB")
            batch_images.append(img)
            batch_questions.append(question)

        # 8. 进行推理
        responses = run_phi3v_batch(
            questions=batch_questions,
            images=batch_images,
            model=model,
            processor=processor,
            tokenizer=tokenizer
        )

        # 9. 保存推理结果并更新已处理组合
        for j, i_in_batch in enumerate(valid_indices):
            entry = batch_entries[i_in_batch]
            combo = (entry['counterfactual_set'], entry['a1a2'])
            
            new_result = {
                "a1_type": entry['a1_type'],
                "a2_type": entry['a2_type'],
                "counterfactual_set": entry['counterfactual_set'],
                "a1a2": entry['a1a2'],
                "prompt": question,
                "response": responses[j]
            }
            existing_results.append(new_result)
            processed_combinations.add(combo)

        # 10. 每批处理完后将结果写回 JSON 文件
        with open(result_file_name, "w", encoding="utf-8") as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=4)

        # 11. 清理本批次数据，释放内存
        for img in batch_images:
            img.close()
        del batch_images, batch_questions, responses, batch_entries
        gc.collect()
        torch.cuda.empty_cache()

    print(f"推理结果已保存到 {result_file_name} 文件中！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Phi-3.5-vision-instruct 进行多模态推理")
    parser.add_argument("--model_id", type=str, 
                        default="/net/graphium/storage3/tingyuan/Phi3-Vision-Finetune/output/run_20250313_000331_000001/model_checkpoint-892_000001/Phi-3.5-vision-instruct-lora",
                        help="模型路径，例如：/net/graphium/storage3/tingyuan/Phi3-Vision-Finetune/output/run_20250313_000331_000001/model_checkpoint-892_000001/Phi-3.5-vision-instruct-lora")
    parser.add_argument("--a1_type", type=str, default="race", help="a1_type 过滤条件")
    parser.add_argument("--a2_type", type=str, default="gender", help="a2_type 过滤条件")
    parser.add_argument("--batch_size", type=int, default=20, help="推理时的 batch 大小")
    args = parser.parse_args()

    model_id = args.model_id
    # 从 model_id 中提取 run_... 中最后一部分的数字（例如 "000001"）
    match = re.search(r'run_\d+_\d+_(\d+)', model_id)
    if match:
        run_suffix = match.group(1)
    else:
        run_suffix = "unknown"

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        _attn_implementation='flash_attention_2'
    )
    # 可选：如果环境支持，可以尝试 torch.compile(model)
    # model = torch.compile(model)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=1)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    main(
        a1_type=args.a1_type,
        a2_type=args.a2_type,
        run_suffix=run_suffix,
        batch_size=args.batch_size
    )
