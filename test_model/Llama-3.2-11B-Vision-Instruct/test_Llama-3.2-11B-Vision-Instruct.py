from PIL import Image
import requests
from transformers import AutoProcessor, MllamaForConditionalGeneration
from datasets import load_dataset
import random
import torch

# 加载模型和处理器
model = MllamaForConditionalGeneration.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")

# 加载数据集
ds = load_dataset("Intel/SocialCounterfactuals")

# 查看数据集的字段信息（以第一个split为例）
print(ds['train'].features)
print(ds['train'][0])

# 随机选择一个数据条目
random_index = random.randint(0, len(ds['train']) - 1)
data_entry = ds['train'][random_index]

# 假设数据集中有一个字段包含图像或图像路径
# 这里需要根据实际数据集字段名进行调整
image_field = 'image'  # 请根据数据集实际字段名进行修改

# 检查数据条目是否为PIL.Image对象
if isinstance(data_entry[image_field], Image.Image):
    image = data_entry[image_field]
else:
    image_path = data_entry[image_field]  # 假设这是图像路径或URL
    image = Image.open(requests.get(image_path, stream=True).raw)

# 保存图像
image.save("random_image.jpg")

# 设置消息模板
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Please describe the person in this image."}
    ]}
]

# 生成输入文本
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

# 处理图像和文本输入
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt"
).to(model.device)

# 生成描述
generate_ids = model.generate(**inputs, max_new_tokens=300)
res = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# 打印结果
print(res)
