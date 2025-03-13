import argparse
from datasets import load_from_disk 
from transformers import AutoProcessor
from PIL import Image
from tqdm import tqdm
import torch
from tools.lora_setting import load_lora_pretrained_model, get_model_name_from_path

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

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]

        # 只保留 allowed_token_ids 对应的 logits
        logits = logits[allowed_token_ids]
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # 获取概率最高的 token
        max_prob_index = torch.argmax(probs).item()
        pred_token_id = allowed_token_ids[max_prob_index]

    response = tokenizer.decode([pred_token_id]).strip()
    return response

def main():
    parser = argparse.ArgumentParser(description="Run gender classification with Phi-3.5-Vision model.")
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

    # 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")

    tokenizer = processor.tokenizer
    male_token_id = tokenizer.encode("male", add_special_tokens=False)[0]
    female_token_id = tokenizer.encode("female", add_special_tokens=False)[0]
    allowed_token_ids = [male_token_id, female_token_id]

    correct = 0
    total = 0

    for example in tqdm(ds):
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"
        image = example['image']
        question = "The gender of the person in this image is:"

        output = run_phi3v_with_allowed_tokens(model, processor, tokenizer, question, image, allowed_token_ids)

        total += 1
        if output.lower() == ground_truth.lower():
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
