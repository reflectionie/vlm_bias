from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import random
import requests
from datasets import load_dataset

def get_random_image(ds, image_field='image', save_path='random_image.jpg'):
    """
    从给定数据集中随机选择一张图片并保存到本地。

    参数：
    - ds: 数据集对象，假设为一个字典，其中包含数据条目。
    - image_field: 数据集中表示图像或图像路径的字段名。
    - save_path: 保存图片的路径。

    返回：
    - PIL.Image 对象。
    """
    try:
        # 随机选择一个数据条目
        random_index = random.randint(0, len(ds['train']) - 1)
        data_entry = ds['train'][random_index]

        # 获取图像
        if isinstance(data_entry[image_field], Image.Image):
            image = data_entry[image_field]
        else:
            image_path = data_entry[image_field]  # 假设这是图像路径或 URL
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw).convert("RGB")

        # 保存图像
        image.save(save_path)
        return image

    except KeyError:
        raise ValueError(f"数据条目中未找到字段: {image_field}")
    except Exception as e:
        raise ValueError(f"获取随机图像失败: {e}")

# LLama 3.2-Vision

def run_mllama(question: str, image: Image.Image):
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.95,  
        enforce_eager=True,
        max_num_seqs=2,
        max_model_len=2048,
        dtype="bfloat16",
        # disable_mm_preprocessor_cache=True
    )

    prompt = f"<|image|><|begin_of_text|>{question}"
    stop_token_ids = None
    return llm, prompt, stop_token_ids, image

def main():
    # 加载数据集
    ds = load_dataset("Intel/SocialCounterfactuals")

    # 获取随机图像
    image = get_random_image(ds, image_field="image")
    question = "What is the content of this image?"

    llm, prompt, stop_token_ids, image_data = run_mllama(question, image)

    sampling_params = SamplingParams(temperature=0.2, max_tokens=512, stop_token_ids=stop_token_ids)

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image_data
        },
    }

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    for o in outputs:
        print(o.outputs[0].text)

if __name__ == "__main__":
    main()
