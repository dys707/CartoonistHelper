# pip install torch transformers pillow
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============== 配置 ==============
DATASET_ROOT = "./dataset_inner"
FORCE_OVERWRITE = False  # 是否覆盖已有txt，False=跳过已存在

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============== 加载模型 ==============
print(f"Loading BLIP on {DEVICE}...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# ============== 单张图生成标签 ==============
def generate_caption(img_path):
    img = Image.open(img_path).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(DEVICE)
    out = model.generate(**inputs, max_length=32, num_beams=2)
    return processor.decode(out[0], skip_special_tokens=True)

# ============== 批量处理 ==============
for style_folder in os.listdir(DATASET_ROOT):
    style_path = os.path.join(DATASET_ROOT, style_folder)
    if not os.path.isdir(style_path):
        continue

    print(f"Processing: {style_folder}")

    for fname in os.listdir(style_path):
        if not fname.lower().endswith(('jpg', 'jpeg', 'png', 'webp')):
            continue

        img_path = os.path.join(style_path, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"

        if os.path.exists(txt_path) and not FORCE_OVERWRITE:
            continue

        caption = generate_caption(img_path)
        caption = f"{caption}, style of {style_folder}"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

print("✅ All captions generated automatically!")