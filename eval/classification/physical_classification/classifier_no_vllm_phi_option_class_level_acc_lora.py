from datasets import load_from_disk
from transformers import AutoProcessor
from vllm.entrypoints.openai.logits_processors import get_logits_processors as get_openai_logits_processors
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd
from tools.lora_setting import load_lora_pretrained_model, get_model_name_from_path
import argparse

# 初始化模型和处理器
model_base = "microsoft/Phi-3.5-vision-instruct"

def run_phi3v_with_allowed_tokens(model, processor, tokenizer, question, image, allowed_token_ids):
    """
    使用 Phi-3.5-Vision 模型进行推理，并限制生成的 token 范围。

    :param model: 已初始化的模型对象。
    :param processor: 已初始化的处理器对象。
    :param tokenizer: 模型的 tokenizer。
    :param question: 输入问题。
    :param image: 输入图像。
    :param allowed_token_ids: 限制的 token ID 列表。
    :return: 模型生成的结果。
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

    logits_processors = get_openai_logits_processors(
        logit_bias=None,
        allowed_token_ids=allowed_token_ids,
        tokenizer=tokenizer
    )

    generation_args = {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "do_sample": False,
        "logits_processor": logits_processors
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # 移除输入的 token
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Run physical characteristics classification with Phi-3.5-Vision model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the LoRA pretrained model.")
    args = parser.parse_args()

    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)

    processor, model = load_lora_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map="cuda",
        use_flash_attn=True
    )

    tokenizer = processor.tokenizer

    # 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_race_test")

    options = ["A", "B", "C", "D", "E"]
    option_token_ids = [tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
    print(f"Option Token IDs: {option_token_ids}")

    # 定义选项与标签的映射
    option_to_label = {
        "A": "skinny",
        "B": "obese",
        "C": "young",
        "D": "old",
        "E": "tattooed"
    }
    labels = list(option_to_label.values())  # ['skinny', 'obese', 'young', 'old', 'tattooed']

    y_true = []
    y_pred = []

    for example in tqdm(ds):
        ground_truth = example['a1'].lower()  # 将 ground_truth 转为小写
        image = example['image']
        question = "What is the physical characteristic of the person in this image?\nOptions: A. Skinny B. Obese C. Young D. Old E. Tattooed.\nAnswer (provide only one letter from the options):"

        output = run_phi3v_with_allowed_tokens(model, processor, tokenizer, question, image, option_token_ids)

        if output in options:
            pred_label = option_to_label[output]
        else:
            pred_label = None

        # 添加到预测列表和真实列表
        y_true.append(ground_truth)
        y_pred.append(pred_label)

    # 计算总体准确率
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    acc = correct / total if total > 0 else 0
    print(f"\nOverall Physical Characteristics Classification Accuracy = {acc:.4f}")

    # 计算每个类别的 Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # 打印每个类别的指标
    metrics_table = pd.DataFrame({
        "Class": labels,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    print("\nClass-level Metrics:")
    print(metrics_table)

    # 打印混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(conf_matrix, index=labels, columns=labels))

    # 计算宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    print("\nMacro Average Metrics:")
    print(f"Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1 Score: {macro_f1:.4f}")

    print("\nWeighted Average Metrics:")
    print(f"Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1 Score: {weighted_f1:.4f}")

if __name__ == "__main__":
    main()
