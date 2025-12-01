import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 1. 指定你的模型配置文件夹路径 (就是第一步创建的文件夹)
model_path = "./yambda/model" 

# 2. 加载配置和分词器
print("正在加载配置...")
config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 核心步骤：使用 from_config 而不是 from_pretrained
# 这会根据配置创建模型结构，并使用随机权重初始化
print("正在随机初始化模型 (这可能需要几分钟)...")
model = AutoModelForCausalLM.from_config(config)

# 打印一下参数量确认大小
print(f"模型参数量: {model.num_parameters() / 1e9:.2f} B")

# 4. 保存为标准的 Hugging Face 格式
# 这一步会生成 model.safetensors 文件
print("正在保存随机初始化的模型...")
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print("完成！现在可以在 LLaMA Factory 中使用该路径作为 model_name_or_path 了。")