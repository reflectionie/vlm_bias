



"""
我有一个数据集：ds = load_from_disk('physical_gender_test')，这个数据集中的属性列有：'image', 'dataset_segment', 'a2_type', 'a2_type', 'counterfactual_set', 'caption', 'a2', 'a2', 'a2a2']；我希望你读取这个数据集，然后使用我的llama对数据进行推理，具体来说推理代码可以参考这个：import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm

# ===== 在函数外部初始化 LLM =====
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    max_num_seqs=2,
    max_model_len=2048,
    dtype="bfloat16",
)

def run_mllama(llm, question, image):

    使用已经初始化好的 llm 对象进行推理，避免每次都重新初始化。

    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|><|image|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    sampling_params = SamplingParams(
    max_tokens=300,
    min_tokens=50,
    stop_token_ids=[128001, 128008, 128009],  # 使用配置中的 eos_token_id
    temperature=0.7,  # 中等随机性
    # top_k=50,         # 限制候选 token 的范围
    top_p=0.9,        # 累积概率裁剪
    repetition_penalty=1.0,  # 防止重复
    # num_beams=5       # 启用 beam search 提高质量
)，我希望你进行Gender Classification，数据集中，a2是ground_truth的gender，gender的可选是male和female，请构造一个问题，比如”这张图片里的人物的性别是：“，然后让模型去推理，拿到推理问题后生成的下一个token是male和是female的概率（这涉及到vllm如何拿到模型输出的token的probability的问题，具体方法是这样，SamplingParams有一个参数logprobs: Optional[int] = None，它的介绍是logprobs: Number of log probabilities to return per output token.
            When set to None, no probability is returned. If set to a non-None
            value, the result includes the log probabilities of the specified
            number of most likely tokens, as well as the chosen tokens.
            Note that the implementation follows the OpenAI API: The API will
            always return the log probability of the sampled token, so there
            may be up to `logprobs+1` elements in the response.我认为用这个可以拿到probs），其中概率大的就认为是模型的回答，然后将回答和ground_truth进行比较，计算acc。
"""

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
        # mm_processor_kwargs={"num_crops": 16},
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
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=1.0,
        logprobs=1,
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

    male_token_id = tokenizer.encode("male", add_special_tokens=False)[0]
    female_token_id = tokenizer.encode("female", add_special_tokens=False)[0]
    allowed_token_ids = [male_token_id, female_token_id]
    print(f"Male Token ID: {male_token_id}, Female Token ID: {female_token_id}")


    for example in tqdm(ds):
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"
        image = example['image']
        question = "The gender of the person in this image is:"

        output = run_phi3v_with_allowed_tokens(llm, question, image, allowed_token_ids)

        allowed_token_info = output.outputs[0].logprobs[0]
        pred_label = None
        max_logprob = float("-inf")

        for token_id, token_logprob in allowed_token_info.items():
            t = token_logprob.decoded_token.strip().lower()
            if t in ["male", "female"]:
                if token_logprob.logprob > max_logprob:
                    max_logprob = token_logprob.logprob
                    pred_label = t

        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
