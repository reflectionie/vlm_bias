import json
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor
from vllm.entrypoints.openai.logits_processors import get_logits_processors as get_openai_logits_processors
from tqdm import tqdm
from adj_prompt import prompts, adj_list  # 保留导入
import os

# 初始化模型和处理器
model_name = "microsoft/Phi-3.5-vision-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, num_crops=1)

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

    # 构造 prompt
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 处理输入
    inputs = processor(prompt, images=[image], return_tensors="pt").to("cuda:0")

    # 获取 logits_processor
    logits_processors = get_openai_logits_processors(
        logit_bias=None,
        allowed_token_ids=allowed_token_ids,
        tokenizer=tokenizer
    )

    # 设置生成参数
    generation_args = {
        "max_new_tokens": 1,
        "temperature": 1.0,
        "do_sample": False,
        "logits_processor": logits_processors
    }

    # 生成答案
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    # 获取生成结果
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return response.strip()

def main():
    # 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/physical_gender_test")

    tokenizer = processor.tokenizer
    option_a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
    allowed_token_ids = [option_a_token_id, option_b_token_id]

    results = []

    for example in tqdm(ds):
        counterfactual_set = example.get("counterfactual_set", "")
        a1a2 = example.get("a1a2", "")
        image = example['image']

        for prompt_template in prompts:  # 使用从 adj_prompt 导入的 prompts
            for adj in adj_list:  # 使用从 adj_prompt 导入的 adj_list
                # 第一次推理
                question1 = prompt_template.format(adj) + "\nOptions: A. Yes B. No.\nAnswer (provide only one letter from the options):"
                output1 = run_phi3v_with_allowed_tokens(model, processor, tokenizer, question1, image, allowed_token_ids)

                # 模拟 log 概率分布（根据实际需要可改为更精确的计算逻辑）
                probs1 = {"Yes": 0 if output1 == "A" else float("-inf"), "No": 0 if output1 == "B" else float("-inf")}

                # 第二次推理
                question2 = prompt_template.format(adj) + "\nOptions: A. No B. Yes.\nAnswer (provide only one letter from the options):"
                output2 = run_phi3v_with_allowed_tokens(model, processor, tokenizer, question2, image, allowed_token_ids)

                probs2 = {"Yes": 0 if output2 == "B" else float("-inf"), "No": 0 if output2 == "A" else float("-inf")}

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
