from PIL import Image, ImageOps
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset
import random

# 加载模型和处理器
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'  # use _attn_implementation='eager' to disable flash attention
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 加载数据集
ds = load_dataset("Intel/SocialCounterfactuals")

# 查看数据集的字段信息
print(ds['train'].features)
print(ds['train'][0])

# 随机选择一个数据条目
random_index = random.randint(0, len(ds['train']) - 1)
data_entry = ds['train'][random_index]

# 假设数据集中有一个字段包含图像或图像路径
image_field = 'image'  # 请根据实际数据集字段名进行修改

# 检查数据条目是否为PIL.Image对象
if isinstance(data_entry[image_field], Image.Image):
    image = data_entry[image_field]
else:
    image_path = data_entry[image_field]  # 假设这是图像路径或URL
    image = Image.open(requests.get(image_path, stream=True).raw)

# 图像预处理
image = ImageOps.exif_transpose(image).convert("RGB")

# 设置prompt
prompt = "<|user|>\n<|image_1|>\nPlease describe the person in this image.<|end|>\n<|assistant|>\n"

# 确保图像与文本正确传递
inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 300,
    "temperature": 0.7,
    "do_sample": True,
}

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# 解码生成的结果
res = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# 打印结果
print(res)
