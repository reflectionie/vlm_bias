from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor
from vllm.entrypoints.openai.logits_processors import get_logits_processors as get_openai_logits_processors
from PIL import Image
from tqdm import tqdm
import torch

# 初始化模型和处理器
model_id = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=1)

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
    # 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_race_test")

    tokenizer = processor.tokenizer
    options = ["A", "B", "C", "D", "E"]
    option_token_ids = [tokenizer.encode(option, add_special_tokens=False)[0] for option in options]
    print(f"Option Token IDs: {option_token_ids}")

    correct = 0
    total = 0

    for example in tqdm(ds):
        ground_truth = example['a1']  # a1 是 ground_truth 的身体特征
        image = example['image']
        question = "What is the physical characteristic of the person in this image?\nOptions: A. Skinny B. Obese C. Young D. Old E. Tattooed.\nAnswer (provide only one letter from the options):"

        output = run_phi3v_with_allowed_tokens(model, processor, tokenizer, question, image, option_token_ids)

        # 解析模型输出的答案
        option_probs = {
            options[i]: float("-inf") for i in range(len(options))
        }

        if output in options:
            option_probs[output] = 0  # 模拟概率以选择正确的选项

        pred_option = max(option_probs, key=option_probs.get)
        option_to_label = {
            "A": "skinny",
            "B": "obese",
            "C": "young",
            "D": "old",
            "E": "tattooed"
        }
        pred_label = option_to_label[pred_option]

        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Physical Characteristics Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
