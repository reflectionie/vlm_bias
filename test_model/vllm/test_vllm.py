from vllm import LLM, SamplingParams  # 修正为 vllm 文档中正确的接口
import random
from PIL import Image
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
    - 保存图片的路径。
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
            image = Image.open(response.raw)

        # 保存图像
        image.save(save_path)
        return save_path

    except KeyError:
        raise ValueError(f"数据条目中未找到字段: {image_field}")
    except Exception as e:
        raise ValueError(f"获取随机图像失败: {e}")

# 初始化 vLLM 接口
# model_path = "llava-hf/llava-1.5-7b-hf"  # 替换为支持的模型路径
# model_path = "microsoft/Phi-3-vision-128k-instruct"
model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
llm = LLM(model=model_path, 
          dtype="bfloat16", 
          gpu_memory_utilization=0.85,  
          enforce_eager=True,
          max_num_seqs=2,
          max_model_len=2048,
          trust_remote_code=True
          )

# 获取随机图片
ds = load_dataset("Intel/SocialCounterfactuals")  # 确保数据集正确加载
image_path = get_random_image(ds, image_field="image")  # 如果字段名不同，请替换

# 定义 Prompt 和推理函数
def run_inference(image_path, prompt, max_output_length=512):
    """
    使用 vLLM 推理接口，传入图片路径和文本描述 prompt。

    参数：
    - image_path: 图片路径。
    - prompt: 文本提示。
    - max_output_length: 模型输出的最大长度（以 token 为单位）。

    返回：
    - 模型生成的文本。
    """
    try:
        # 打开图像
        image = Image.open(image_path)

        # 定义 Prompt 和多模态输入
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

        # 设置采样参数，包括输出长度限制
        sampling_params = SamplingParams(
            # temperature=0.7,  # 设置生成温度
            max_tokens=max_output_length  # 设置输出最大 token 数
        )

        # 调用推理引擎
        result = llm.generate(inputs, sampling_params)
        return result[0].outputs[0].text  # 假设生成的结果在此路径
    except Exception as e:
        raise ValueError(f"推理失败: {e}")

# 调用推理
prompt = "USER: <image>\nPlease describe the person in this image.\nASSISTANT:"
try:
    output = run_inference(image_path, prompt, max_output_length=512)
    print("生成的描述: ", output)
except Exception as e:
    print(f"执行推理时出错: {e}")
