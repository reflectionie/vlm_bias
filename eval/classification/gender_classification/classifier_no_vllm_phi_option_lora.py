import argparse
from datasets import load_from_disk
from transformers import AutoProcessor
from vllm.entrypoints.openai.logits_processors import get_logits_processors as get_openai_logits_processors
from PIL import Image
from tqdm import tqdm
import torch
from tools.lora_setting import load_lora_pretrained_model, get_model_name_from_path

# 初始化模型和处理器
def initialize_model(model_path, model_base):
    model_name = get_model_name_from_path(model_path)
    processor, model = load_lora_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        device_map="cuda",
        use_flash_attn=True
    )
    return processor, model

def run_phi3v(model, processor, tokenizer, question, image, allowed_token_ids):
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
        "temperature": 0.0,
        "do_sample": False,
        "logits_processor": logits_processors
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # 移除输入的 token
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response.strip()

def main():
    parser = argparse.ArgumentParser(description="Run Phi-3.5-Vision Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model")
    args = parser.parse_args()

    model_base = "microsoft/Phi-3.5-vision-instruct"
    model_path = args.model_path

    processor, model = initialize_model(model_path, model_base)

    # 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")

    tokenizer = processor.tokenizer
    option_a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
    allowed_token_ids = [option_a_token_id, option_b_token_id]

    correct = 0
    total = 0

    for example in tqdm(ds):
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"
        image = example['image']
        question = "What is the gender of the person in this image?\nOptions: A. Male B. Female.\nAnswer (provide only one letter from the options):"

        output = run_phi3v(model, processor, tokenizer, question, image, allowed_token_ids)

        # 解析模型输出的答案
        pred_label = "male" if output == "A" else "female"

        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
