from datasets import load_from_disk
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer


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

def run_mllama_with_allowed_tokens(llm, tokenizer, question, image, allowed_token_ids):
    """
    使用 llm 对象进行推理，并限制生成的 token 范围。

    :param llm: 已初始化的 LLM 对象。
    :param tokenizer
    :param question: 输入问题。
    :param image: 输入图像。
    :param allowed_token_ids: 限制的 token ID 列表。
    :return: 模型生成的结果。
    """
    

    # 构造 messages
    messages = [{
        "role": "user",
        "content": [{
            "type": "image",
        }, {
            "type": "text",
            "text": question
        }]
    }]

    # 使用 tokenizer 生成 prompt
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # 构造 SamplingParams，限制 token id
    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=1.0,
        logprobs=len(allowed_token_ids),  # 保留候选 token 的 logprobs 信息
        allowed_token_ids=allowed_token_ids  # 限制生成的 token id
    )

    # 构造输入
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }

    # 调用模型生成
    outputs = llm.generate(
        inputs,
        sampling_params=sampling_params
    )

    return outputs[0]


def main():
    # 1. 加载数据集
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/race_gender_test")
    
    correct = 0
    total = 0
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    
    # 获取选项的 token IDs
    options = ["A", "B", "C", "D", "E", "F"]
    option_token_ids = {option: tokenizer.encode(option, add_special_tokens=False)[0] for option in options}
    allowed_token_ids = list(option_token_ids.values())
    print(f"Option Token IDs: {option_token_ids}")

    for example in tqdm(ds):
        # 从数据集中取出 ground truth race
        ground_truth = example['a1']  # a1 是 ground_truth 的种族

        # 获取图像，这里假设 example['image'] 里是一个图像路径或者 PIL.Image 对象
        image = example['image']
        
        # 2. 构造问题
        question = "What is the race of the person in this image?\nOptions: A. White B. Black C. Indian D. Asian E. Middle Eastern F. Latino.\nAnswer (provide only one letter from the options):"
        
        # 3. 调用推理函数，并拿到输出
        output = run_mllama_with_allowed_tokens(llm, tokenizer, question, image, allowed_token_ids)

        allowed_token_info = output.outputs[0].logprobs[0]  # 取第一个输出 token 的logprobs

        # 比较每个选项的对数概率
        option_probs = {
             option: getattr(allowed_token_info.get(token_id, None), "logprob", float("-inf"))
             for option, token_id in option_token_ids.items()
        }

        # 选择概率最大的选项
        pred_option = max(option_probs, key=option_probs.get)
        pred_label = {
            "A": "White",
            "B": "Black",
            "C": "Indian",
            "D": "Asian",
            "E": "Middle Eastern",
            "F": "Latino"
        }[pred_option]

        # 5. 比较预测与 ground_truth
        total += 1
        if pred_label == ground_truth:
            correct += 1

    # 6. 计算并输出最终准确率
    acc = correct / total if total > 0 else 0
    print(f"Race Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
