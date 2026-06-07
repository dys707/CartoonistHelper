from PIL import Image
import os
import sys

def resize_to_512(input_path, output_path=None, cover=False):
    """
    将图片转换为 512x512，保持比例 + 居中裁剪，不变形
    :param input_path: 输入图片路径
    :param output_path: 输出路径（不填则自动生成）
    :param cover: 是否覆盖原图
    """
    try:
        # 打开图片
        img = Image.open(input_path).convert("RGB")
        width, height = img.size

        # 计算居中裁剪的区域（保持比例）
        min_side = min(width, height)
        left = (width - min_side) / 2
        top = (height - min_side) / 2
        right = left + min_side
        bottom = top + min_side

        # 裁剪 + 缩放
        img_cropped = img.crop((left, top, right, bottom))
        img_512 = img_cropped.resize((512, 512), Image.Resampling.LANCZOS)

        # 自动生成输出路径
        if not output_path:
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_512x512{ext}"

        # 保存
        img_512.save(output_path, quality=95)
        print(f"✅ 成功转换：{output_path}")

    except Exception as e:
        print(f"❌ 处理失败：{input_path}，错误：{str(e)}")

def batch_resize(folder_path, cover=False):
    """批量转换文件夹内所有图片"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(exts):
            img_path = os.path.join(folder_path, filename)
            resize_to_512(img_path, cover=cover)

# ====================== 使用方式 ======================
if __name__ == "__main__":
    # 【单张图片转换】直接改这里
    image_path = "cat3.jpg"  # 你的图片路径
    resize_to_512(image_path)

    # 【批量转换文件夹】取消下面注释
    # batch_resize("./images")  # 你的图片文件夹路径