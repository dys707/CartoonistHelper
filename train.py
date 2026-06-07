# train_lora.py 参考训练文件，注意配置
import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
import shutil


class StyleDataset(Dataset):
    def __init__(self, style_dir, tokenizer, size=512, center_crop=True):
        self.style_dir = style_dir
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop

        self.image_paths = []
        self.caption_paths = []
        for f in os.listdir(style_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(style_dir, f)
                txt_path = os.path.join(style_dir, os.path.splitext(f)[0] + '.txt')
                if os.path.exists(txt_path):
                    self.image_paths.append(img_path)
                    self.caption_paths.append(txt_path)
                else:
                    print(f"Warning: missing caption for {img_path}")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        with open(self.caption_paths[idx], 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        # tokenize caption
        text_input = self.tokenizer(caption, max_length=self.tokenizer.model_max_length,
                                    padding="max_length", truncation=True, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": text_input.input_ids.squeeze(0),
        }


def train_lora(args):
    # 初始化 accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.report_to == "wandb" else None,
    )

    # 加载 tokenizer 和 text encoder
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path).unet

    # 冻结 vae 和 text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 配置 LoRA (只对 UNet 的注意力层)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 数据集
    dataset = StyleDataset(args.style_dir, tokenizer, size=args.resolution)
    train_dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True,
                                  num_workers=args.dataloader_num_workers)

    # 优化器
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 噪声调度器
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 准备加速器
    unet, optimizer, train_dataloader = accelerator.prepare(unet, optimizer, train_dataloader)

    # 学习率调度器
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 训练循环
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), desc="Training steps",
                        disable=not accelerator.is_local_main_process)
    for epoch in range(args.num_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 将像素值编码到潜空间
                latents = vae.encode(
                    batch["pixel_values"].to(accelerator.device, dtype=torch.float32)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=torch.float32)

                # 采样噪声
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                                          device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 编码文本
                encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]

                # 预测噪声
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 损失
                loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    # 保存 LoRA 权重
                    accelerator.wait_for_everyone()
                    unet = accelerator.unwrap_model(unet)
                    lora_state_dict = get_peft_model_state_dict(unet)
                    if accelerator.is_main_process:
                        lora_path = os.path.join(args.output_dir, f"lora_step_{global_step}.safetensors")
                        torch.save(lora_state_dict, lora_path)
                        print(f"Saved LoRA at {lora_path}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # 最终保存
    accelerator.wait_for_everyone()
    unet = accelerator.unwrap_model(unet)
    lora_state_dict = get_peft_model_state_dict(unet)
    if accelerator.is_main_process:
        final_lora_path = os.path.join(args.output_dir, "lora_final.safetensors")
        torch.save(lora_state_dict, final_lora_path)
        print(f"Training completed. Final LoRA saved at {final_lora_path}")
        # 同时也保存一份带风格名的副本
        style_name = os.path.basename(args.style_dir.rstrip('/'))
        shutil.copy(final_lora_path, os.path.join(args.output_dir, f"{style_name}_lora.safetensors"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style_dir", type=str, required=True,
                        help="某个风格文件夹的路径，如 /root/autodl-tmp/dataset_processed/水墨风")
    parser.add_argument("--output_dir", type=str, required=True, help="保存 LoRA 权重的目录")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="基础 SD 模型")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="none", choices=["wandb", "tensorboard", "none"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_lora(args)