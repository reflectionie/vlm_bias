import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from PIL import Image
from tqdm import tqdm

# ======= 在函数外部初始化 LLM（外部） =======
model_name = "microsoft/Phi-3.5-vision-instruct"
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    max_model_len=4096,
    dtype="bfloat16"
)

def run_phi3v(llm, question: str, image: Image.Image) -> str:
    """
    使用在外部已初始化的 llm 对象进行多模态推理的函数。

    :param llm: 已经初始化的 vllm.LLM 对象
    :param question: 文本问题
    :param image: PIL.Image 对象
    :return: 模型生成的文本响应
    """
    # 构造适用于 Phi-3.5-vision-instruct 的多模态指令模板
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n"

    # 根据需求设定推理的采样参数
    sampling_params = SamplingParams(
        max_tokens=300,
        min_tokens=50,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.0,
    )

    # 构造 vllm.generate 的输入
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }

    # 调用已初始化的 llm 对象进行推理
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # 获取生成的文本
    response = outputs[0].outputs[0].text
    return response


def main(a1_type, a2_type):
    # 加载数据集
    ds = load_dataset("/net/graphium/storage3/tingyuan/vlm_bias/test_dataset/race_gender_train")

    # 提取子集
    subset = ds['train'].filter(
        lambda x: x['a1_type'] == a1_type and x['a2_type'] == a2_type
    )

    # 用于去重，避免重复处理
    processed_combinations = set()

    # 生成结果文件名（并与当前脚本所在目录拼接）
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录
    model_save_name = "D1-Phi-3.5-vision-instruct"
    result_file_name = os.path.join(script_dir, f"{model_save_name}_{a1_type}_{a2_type}.json")

    # 加载已有结果（如果存在）
    try:
        with open(result_file_name, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
            for res in existing_results:
                processed_combinations.add((res['counterfactual_set'], res['a1a2']))
    except FileNotFoundError:
        existing_results = []

    # 准备问题
    question = "What is the content of this image?"

    # 遍历子集并进行推理
    for entry in tqdm(subset, desc=f"Processing a1_type={a1_type}, a2_type={a2_type}"):
        combination = (entry['counterfactual_set'], entry['a1a2'])
        # 跳过已处理的数据
        if combination in processed_combinations:
            continue

        try:
            # 加载图像
            image_field = "image"
            if isinstance(entry[image_field], Image.Image):
                image = entry[image_field]
            else:
                image_path = entry[image_field]
                image = Image.open(image_path).convert("RGB")

            # ========== 调用 run_phi3v 进行推理，使用已经在外部初始化的 llm ==========
            response = run_phi3v(llm, question, image)

            # 构造结果
            new_result = {
                "a1_type": entry['a1_type'],
                "a2_type": entry['a2_type'],
                "counterfactual_set": entry['counterfactual_set'],
                "a1a2": entry['a1a2'],
                "prompt": question,
                "response": response
            }

            # 将新结果追加到 existing_results 中
            existing_results.append(new_result)
            processed_combinations.add(combination)

            # ========= 关键改动：每生成一条数据就把所有结果写回文件 =========
            with open(result_file_name, "w", encoding="utf-8") as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"处理条目时出错: {e}")

    print(f"推理结果已保存到 {result_file_name} 文件中！")


if __name__ == "__main__":
    # 调用主函数时传入需要处理的 a1_type 和 a2_type
    main(a1_type="race", a2_type="gender")
