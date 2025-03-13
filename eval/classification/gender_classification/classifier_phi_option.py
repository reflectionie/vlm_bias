from datasets import load_from_disk
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import logging

llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=2048,
    max_num_seqs=2,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    enforce_eager=True
)

def run_phi3v_with_allowed_tokens(llm, question, image, allowed_token_ids):
    """
    使用 Phi-3.5-Vision 模型进行推理，并限制生成的 token 范围。

    :param llm: 已初始化的 LLM 对象。
    :param question: 输入问题。
    :param image: 输入图像。
    :param allowed_token_ids: 限制的 token ID 列表。
    :return: 模型生成的结果。
    """
    prompt = f"<|user|>\n<|image_1|>\n{question}\n<|end|>\n<|assistant|>\n"

    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=1.0,
        logprobs=len(allowed_token_ids),
        allowed_token_ids=allowed_token_ids
    )
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params
    )
    
    return outputs[0]

def main():
    # 1. 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")

    correct = 0
    total = 0
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-vision-instruct")

    # 获取选项的 token IDs
    option_a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
    allowed_token_ids = [option_a_token_id, option_b_token_id]
    print(f"Option A Token ID: {option_a_token_id}, Option B Token ID: {option_b_token_id}")

    for example in tqdm(ds):
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"
        image = example['image']
        question = "What is the gender of the person in this image?\nOptions: A. Male B. Female.\nAnswer (provide only one letter from the options):"

        output = run_phi3v_with_allowed_tokens(llm, question, image, allowed_token_ids)

        allowed_token_info = output.outputs[0].logprobs[0]
        
        # 比较每个选项的对数概率
        option_probs = {
            "A": getattr(allowed_token_info.get(option_a_token_id, None), "logprob", float("-inf")),
            "B": getattr(allowed_token_info.get(option_b_token_id, None), "logprob", float("-inf"))
        }

        # 选择概率最大的选项
        pred_option = max(option_probs, key=option_probs.get)
        pred_label = "male" if pred_option == "A" else "female"

        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
