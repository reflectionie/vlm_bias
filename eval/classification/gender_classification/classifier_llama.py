



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

# 禁止打印 info 级别的日志
logging.getLogger("vllm").setLevel(logging.WARNING)


# 假设模型已经在函数外部初始化过，这里仅做示例
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = LLM(
    model=model_name,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    max_num_seqs=2,
    max_model_len=2048,
    dtype="bfloat16",
)

def run_mllama_with_allowed_tokens(llm, question, image, allowed_token_ids):
    """
    使用 llm 对象进行推理，并限制生成的 token 范围。
    
    :param llm: 已初始化的 LLM 对象。
    :param question: 输入问题。
    :param image: 输入图像。
    :param allowed_token_ids: 限制的 token ID 列表。
    :return: 模型生成的结果。
    """
    # 构造 Prompt
    prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|><|image|>{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    # 构造 SamplingParams，限制 token id
    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=1.0,
        logprobs=1,  # 保留候选 token 的 logprobs 信息
        allowed_token_ids=allowed_token_ids  # 限制生成的 token id
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
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    male_token_id = tokenizer.encode("male", add_special_tokens=False)[0]
    female_token_id = tokenizer.encode("female", add_special_tokens=False)[0]
    allowed_token_ids = [male_token_id, female_token_id]
    print(f"Male Token ID: {male_token_id}, Female Token ID: {female_token_id}")


    for example in tqdm(ds):
        # 从数据集中取出 ground truth gender
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"

        # 获取图像，这里假设 example['image'] 里是一个图像路径或者 PIL.Image 对象
        # 如果是路径，需要先用 PIL.Image.open() 打开；如果已经是 Image object，可直接使用。
        # 下面示例假设是已经是 PIL Image：
        image = example['image']
        
        # 2. 构造问题
        question = "The gender of the person in this image is:"
        
        # 3. 调用推理函数，并拿到输出
        output = run_mllama_with_allowed_tokens(llm, question, image, allowed_token_ids)

        
        allowed_token_info = output.outputs[0].logprobs[0]  # 取第一个输出 token 的logprobs
        # 这里是对数概率，需要先转为概率再比较或者直接比较对数概率大小也可以（对数概率大的就是概率更大）。

        # 先设一个默认值
        pred_label = None
        max_logprob = float("-inf")

        # 遍历 top_token_info，找出 "male"/"female" 对应的对数概率最大的那个
        for token_id, token_logprob in allowed_token_info.items():
            # 为了安全，先把 token_str 做个小写去空格处理
            t = token_logprob.decoded_token.strip().lower()
            if t in ["male", "female"]:
                if token_logprob.logprob > max_logprob:
                    max_logprob = token_logprob.logprob
                    pred_label = t

        # 5. 比较预测与 ground_truth
        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    # 6. 计算并输出最终准确率
    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
