#测试
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 加载单个 LoRA
lora_path = "lora_models/VanGogh"
pipe.load_lora_weights(
    lora_path,              # 参数1: LoRA 文件所在的文件夹路径
    weight_name="lora_final_converted.safetensors"  # 参数2: LoRA 文件的准确文件名
)
# 生成图片
prompt = " A classroom , Van Gogh style"  # 建议加上风格触发词
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("output3.png")
