import json
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from adj_prompt import prompts, adj_list
import logging
import os

# 禁止打印 info 级别的日志
logging.getLogger("vllm").setLevel(logging.WARNING)

# 假设模型已经在函数外部初始化过，这里仅做示例
model_name = "microsoft/Phi-3.5-vision-instruct"
llm = LLM(
    model=model_name,
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
    # 加载数据集
    # ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/race_gender_test")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-vision-instruct")

    # 获取选项的 token IDs
    option_a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
    allowed_token_ids = [option_a_token_id, option_b_token_id]

    results = []

    for example in tqdm(ds):
        counterfactual_set = example.get("counterfactual_set", "")
        a1a2 = example.get("a1a2", "")
        image = example['image']

        for prompt_template in prompts:
            for adj in adj_list:
                # 第一次推理
                question1 = prompt_template.format(adj) + "\nOptions: A. Yes B. No.\nAnswer (provide only one letter from the options):"
                output1 = run_phi3v_with_allowed_tokens(llm, question1, image, allowed_token_ids)

                allowed_token_info1 = output1.outputs[0].logprobs[0]

                probs1 = {
                    "Yes": allowed_token_info1.get(option_a_token_id, None).logprob if allowed_token_info1.get(option_a_token_id, None) else float("-inf"),
                    "No": allowed_token_info1.get(option_b_token_id, None).logprob if allowed_token_info1.get(option_b_token_id, None) else float("-inf")
                }

                # 第二次推理
                question2 = prompt_template.format(adj) + "\nOptions: A. No B. Yes.\nAnswer (provide only one letter from the options):"
                output2 = run_phi3v_with_allowed_tokens(llm, question2, image, allowed_token_ids)

                allowed_token_info2 = output2.outputs[0].logprobs[0]

                probs2 = {
                    "Yes": allowed_token_info2.get(option_b_token_id, None).logprob if allowed_token_info2.get(option_b_token_id, None) else float("-inf"),
                    "No": allowed_token_info2.get(option_a_token_id, None).logprob if allowed_token_info2.get(option_a_token_id, None) else float("-inf")
                }

                # 累加 logprob
                total_probs = {
                    "Yes": probs1["Yes"] + probs2["Yes"],
                    "No": probs1["No"] + probs2["No"]
                }

                # 选择最终选项
                choice = max(total_probs, key=total_probs.get)

                result = {
                    "counterfactual_set": counterfactual_set,
                    "a1a2": a1a2,
                    "prompt": prompt_template,
                    "adj": adj,
                    "choice": choice,
                    "probs": {
                        "first_log_probs": probs1,
                        "second_log_probs": probs2,
                        "total_log_probs": total_probs
                    }
                }

                results.append(result)

    # 写入结果文件到当前脚本所在路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    result_file = os.path.join(script_dir, f"adj_{model_name.replace('/', '_')}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
