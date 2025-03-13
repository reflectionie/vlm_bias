from datasets import load_from_disk
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# 初始化 Phi-3.5-Vision 模型
llm = LLM(
    model="microsoft/Phi-3.5-vision-instruct",
    trust_remote_code=True,
    max_model_len=2048,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    enforce_eager=True
)

# 初始化 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-vision-instruct")

def compute_candidate_logprob_step_by_step(
    llm,
    question: str,
    image,
    candidate_str: str,
    tokenizer,
    union_ids
) -> float:
    """
    在仅允许 union_ids 的前提下，逐 token 强制生成 candidate_str，
    计算 log P(candidate_str | prompt)。
    """
    # 0. 分词
    candidate_ids = tokenizer.encode(candidate_str, add_special_tokens=False)

    # 1. 构造初始 Prompt
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

    total_logprob = 0.0

    # 2. 逐 token 做 teacher forcing
    for token_id in candidate_ids:
        # a) 只生成 1 个 token，限制在 union_ids 中
        sampling_params = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            logprobs=len(union_ids),
            allowed_token_ids=union_ids
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            }
        }
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        completion = outputs[0].outputs[0]

        candidate_token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

        # 这里假设 completion.top_logprobs[0] 是一个字典: {token_str: logprob} (仅这一步的)
        step_top_logprob_dict = completion.logprobs[0]  # vLLM 通常会在这里存"第0步"的 top tokens
        step_logprob = step_top_logprob_dict.get(token_id, float('-inf'))

        # c) 累加
        total_logprob += step_logprob.logprob

        # d) 将“我们想要的那个 token”加到 prompt，进入下一 step
        #    （无论它是不是最高概率，我们都把它视为下一步输出的 token，这才是 teacher forcing）
        prompt += candidate_token_text

    return total_logprob

def main():
    # 假设 dataset 已经加载好了, 这里略
    # ds = load_from_disk(...)
    ds = load_from_disk("/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/test_dataset/race_gender_test")

    allowed_tokens = ["White", "Black", "Indian", "Asian", "Middle Eastern", "Latino"]
    # 预先做并集
    union_token_ids = set()
    for word in allowed_tokens:
        tks = tokenizer.encode(word, add_special_tokens=False)
        union_token_ids.update(tks)
    union_token_ids = list(union_token_ids)

    correct = 0
    total = 0
    
    
    for example in ds:
        image = example["image"]
        ground_truth = example["a1"]
        question = "The race of the person in this image is:"

        # 对每个候选词计算其 logprob
        best_candidate = None
        best_logp = float("-inf")
        
        # hashmap = {key: float('-inf') for key in allowed_tokens}
        for candidate_str in allowed_tokens:
            lp = compute_candidate_logprob_step_by_step(
                llm, question, image, candidate_str, tokenizer, union_token_ids
            )
            if lp > best_logp:
                best_logp = lp
                best_candidate = candidate_str
                
            # hashmap[candidate_str] = lp
        
        # sorted_items = sorted(hashmap.items(), key=lambda x: x[1])
        # 对比 ground_truth
        total += 1
        if best_candidate.lower() == ground_truth.lower():
            correct += 1

    acc = correct / total if total else 0.0
    print(f"Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
