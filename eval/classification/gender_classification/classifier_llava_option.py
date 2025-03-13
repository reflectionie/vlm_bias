from datasets import load_from_disk
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# 假设模型已经在函数外部初始化过，这里仅做示例
llava_model_name = "llava-hf/llava-1.5-7b-hf"
llm = LLM(
    model=llava_model_name,
    max_model_len=4096,
    trust_remote_code=True,
    dtype="bfloat16",
)

def run_llava_with_allowed_tokens(llm, question, image, allowed_token_ids):
    """
    使用 llm 对象进行推理，并限制生成的 token 范围。
    
    :param llm: 已初始化的 LLM 对象。
    :param question: 输入问题。
    :param image: 输入图像。
    :param allowed_token_ids: 限制的 token ID 列表。
    :return: 模型生成的结果。
    """
    # 构造 Prompt
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    
    sampling_params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=1.0,
        logprobs=len(allowed_token_ids),
        allowed_token_ids=allowed_token_ids  # 限制生成的 token id
    )
    # 组织要传给 llm.generate 的输入
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
    tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

    # 获取选项的 token IDs
    option_a_token_id = tokenizer.encode("A", add_special_tokens=False)[0]
    option_b_token_id = tokenizer.encode("B", add_special_tokens=False)[0]
    allowed_token_ids = [option_a_token_id, option_b_token_id]
    print(f"Option A Token ID: {option_a_token_id}, Option B Token ID: {option_b_token_id}")

    for example in tqdm(ds):
        # 从数据集中取出 ground truth gender
        ground_truth = example['a2']  # a2 是 ground_truth 的性别，"male" 或 "female"

        # 获取图像，这里假设 example['image'] 里是一个图像路径或者 PIL.Image 对象
        image = example['image']
        
        # 2. 构造问题
        question = "What is the gender of the person in this image?\nOptions: A. Male B. Female.\nAnswer (provide only one letter from the options):"
        
        # 3. 调用推理函数，并拿到输出
        output = run_llava_with_allowed_tokens(llm, question, image, allowed_token_ids)

        allowed_token_info = output.outputs[0].logprobs[0]  # 取第一个输出 token 的logprobs

        # 比较每个选项的对数概率
        option_probs = {
            "A": getattr(allowed_token_info.get(option_a_token_id, None), "logprob", float("-inf")),
            "B": getattr(allowed_token_info.get(option_b_token_id, None), "logprob", float("-inf"))
        }

        # 选择概率最大的选项
        pred_option = max(option_probs, key=option_probs.get)
        pred_label = "male" if pred_option == "A" else "female"

        # 5. 比较预测与 ground_truth
        total += 1
        if pred_label == ground_truth.lower():
            correct += 1

    # 6. 计算并输出最终准确率
    acc = correct / total if total > 0 else 0
    print(f"Gender Classification Accuracy = {acc:.4f}")

if __name__ == "__main__":
    main()
