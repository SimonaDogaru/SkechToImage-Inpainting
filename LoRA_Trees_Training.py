!unzip -q /content/LoRA_Trees.zip -d /content/LoRA_Trees

!pip install -q diffusers==0.33.1 transformers accelerate torchvision xformers datasets peft
!pip install -q --upgrade huggingface_hub

!pip install -U diffusers transformers accelerate

# Cell 1
import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model

# --- Config ---
class Config:
    pretrained_model    = "runwayml/stable-diffusion-v1-5"
    controlnet_model    = "lllyasviel/sd-controlnet-scribble"
    resolution          = 512
    batch_size          = 4
    grad_accum_steps    = 1
    lr                  = 2e-4
    lr_warmup           = 100
    max_train_steps     = 2000
    mixed_precision     = "fp16"
    seed                = 42
    lora_rank           = 32
    lora_alpha          = 64
    lora_dropout        = 0.1

    prompt              = "a professional high-quality photograph of a tree"
    negative_prompt     = "blurry, low quality, cartoon, drawing, painting"

    output_dir          = "/content/LoRA_Trees/output"
    train_sketch_dir    = "/content/LoRA_Trees/LoRA_Trees/Train/sketch_tree"
    train_image_dir     = "/content/LoRA_Trees/LoRA_Trees/Train/photo_tree"
    val_sketch_dir      = "/content/LoRA_Trees/LoRA_Trees/Validation/sketch_tree"
    val_image_dir       = "/content/LoRA_Trees/LoRA_Trees/Validation/photo_tree"

    checkpoint_steps    = 200
    validation_steps    = 100
    num_val_images      = 4

config = Config()
set_seed(config.seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Cell 2: LoRA setup + funcție de salvare
import os
from peft import LoraConfig, get_peft_model

# 2.1) Configurație LoRA — target doar pe Linear-urile din cross-attention
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    target_modules=["to_q", "to_k", "to_v"],  # strict Linear layers
    lora_dropout=config.lora_dropout,
    bias="none",
)

# 2.2) Funcție de salvare doar adapter LoRA
def save_lora_only(unet, controlnet, output_dir, step):
    save_dir = os.path.join(output_dir, f"lora_checkpoint_{step}")
    os.makedirs(save_dir, exist_ok=True)
    unet.save_pretrained(save_dir, safe_serialization=True)
    controlnet.save_pretrained(save_dir, safe_serialization=True)
    print(f"✅ LoRA adapters saved in: {save_dir}")

# Cell 3: Încarcă și wrap-uiește modelele cu LoRA

from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# 3.1) Încarcă componentele
tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
vae          = AutoencoderKL.from_pretrained(config.pretrained_model, subfolder="vae")
unet         = UNet2DConditionModel.from_pretrained(config.pretrained_model, subfolder="unet")
controlnet   = ControlNetModel.from_pretrained(config.controlnet_model)

# 3.2) Îmbracă cu LoRA (folosind doar target_modules=[\"to_q\",\"to_k\",\"to_v\"])
unet       = get_peft_model(unet, lora_config)
controlnet = get_peft_model(controlnet, lora_config)

print("✅ UNet și ControlNet wrap-uite cu LoRA pe to_q, to_k, to_v")

# Cell 4: Prepare DataLoader
train_ds = ControlNetDataset(
    config.train_sketch_dir,
    config.train_image_dir,
    tokenizer,
    size=config.resolution
)
print(f"🚀 Found {len(train_ds)} pairs")
assert len(train_ds) > 0, "No training pairs found!"

train_dl = DataLoader(
    train_ds,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

accelerator = Accelerator(
    gradient_accumulation_steps=config.grad_accum_steps,
    mixed_precision=config.mixed_precision,
    log_with="tensorboard",
    project_dir=os.path.join(config.output_dir, "logs"),
)

optimizer = torch.optim.AdamW(
    list(unet.parameters()) + list(controlnet.parameters()),
    lr=config.lr,
    weight_decay=0.01
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup,
    num_training_steps=config.max_train_steps,
)
noise_scheduler = DDPMScheduler.from_pretrained(
    config.pretrained_model, subfolder="scheduler"
)
ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
ema_unet.to(accelerator.device)

unet, controlnet, optimizer, train_dl, lr_scheduler = accelerator.prepare(
    unet, controlnet, optimizer, train_dl, lr_scheduler
)
text_encoder = text_encoder.to(accelerator.device, dtype=torch.float32)
vae          = vae.to(accelerator.device, dtype=torch.float32)

# Cell 5: save_checkpoint & validate_and_save
def save_checkpoint(accelerator, cfg, step):
    if accelerator.is_main_process:
        path = os.path.join(cfg.output_dir, f"checkpoint-{step}")
        accelerator.save_state(path)
        print(f"Saved full checkpoint at step {step}")

def validate_and_save(accelerator, cfg, models, step):
    tokenizer, text_encoder, vae, unet, controlnet = models
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        controlnet=accelerator.unwrap_model(controlnet),
        scheduler=DPMSolverMultistepScheduler.from_pretrained(cfg.pretrained_model, subfolder="scheduler"),
        safety_checker=None,
        feature_extractor=None,
    ).to(accelerator.device)

    os.makedirs(os.path.join(cfg.output_dir, f"validation_step_{step}"), exist_ok=True)
    sketch_files = sorted(os.listdir(cfg.val_sketch_dir))[:cfg.num_val_images]

    for i, fname in enumerate(sketch_files):
        try:
            sch = Image.open(os.path.join(cfg.val_sketch_dir, fname)).convert("L").resize((cfg.resolution, cfg.resolution))
            sketch_np = np.stack([np.array(sch)]*3, axis=-1)
            sketch_t = torch.from_numpy(sketch_np).permute(2,0,1).unsqueeze(0).to(accelerator.device)/255.0

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = pipe(
                    prompt=cfg.prompt,
                    negative_prompt=cfg.negative_prompt,
                    image=sketch_t,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=torch.Generator(device=accelerator.device).manual_seed(cfg.seed),
                ).images[0]
            out.save(os.path.join(cfg.output_dir, f"validation_step_{step}", f"{i}_{fname}"))

        except Exception as e:
            print(f"Eroare la validare pentru {fname}: {e}")

# Cell 6: Training loop cu salvare LoRA-only la final (cu excepție)
def train_loop(cfg, models, dataloader, accelerator):
    tok, txt, vae_m, unet_m, cn_m = models
    global_step = 0
    pbar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(100):
        unet_m.train()
        cn_m.train()
        total_loss = 0.0

        for batch in dataloader:
            with accelerator.accumulate(unet_m, cn_m):
                imgs = batch["image"].to(accelerator.device)
                sks  = batch["sketch"].to(accelerator.device)
                ids  = batch["input_ids"].to(accelerator.device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    lat = vae_m.encode(imgs).latent_dist.sample() * vae_m.config.scaling_factor
                    noise = torch.randn_like(lat)
                    ts = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (lat.size(0),),
                        device=accelerator.device
                    )
                    nlat = noise_scheduler.add_noise(lat, noise, ts)

                    enc = txt(ids)[0]
                    ds, ms = cn_m(
                        nlat, ts,
                        encoder_hidden_states=enc,
                        controlnet_cond=sks,
                        return_dict=False
                    )
                    pred = unet_m(
                        nlat, ts,
                        encoder_hidden_states=enc,
                        down_block_additional_residuals=ds,
                        mid_block_additional_residual=ms
                    ).sample

                    target = noise if noise_scheduler.config.prediction_type == "epsilon" \
                             else noise_scheduler.get_velocity(lat, noise, ts)
                    loss = torch.nn.functional.mse_loss(pred.float(), target.float())

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema_unet.step(unet_m.parameters())
                    total_loss += loss.item()
                    pbar.update(1)
                    global_step += 1

                    if global_step % cfg.checkpoint_steps == 0:
                        save_checkpoint(accelerator, cfg, global_step)
                    if global_step % cfg.validation_steps == 0:
                        validate_and_save(accelerator, cfg, models, global_step)

            if global_step >= cfg.max_train_steps:
                break

    accelerator.wait_for_everyone()

    # --- Aici salvăm doar LoRA-ul, prindem și orice eroare ---
    if accelerator.is_main_process:
        try:
            save_lora_only(unet_m, cn_m, cfg.output_dir, "final")
        except Exception as e:
            print(f"⚠️ Error saving LoRA-only adapter: {e}")

    accelerator.end_training()

# Cell 7: Start training
models = (tokenizer, text_encoder, vae, unet, controlnet)
notebook_launcher(lambda args: train_loop(*args),
                  args=((config, models, train_dl, accelerator),),
                  num_processes=1)

!zip -r /content/checkpoint-final.zip LoRA_Trees/output/checkpoint-final

from google.colab import files
files.download('/content/checkpoint-final.zip')

from google.colab import drive
drive.mount('/content/drive')

# (optional) make a folder in your Drive
!mkdir -p /content/drive/MyDrive/LoRATrees2

# copy the checkpoint-final directory
!cp -r LoRA_Trees/output/checkpoint-final \
      //content/drive/MyDrive/LoRATrees2
# copy the checkpoint-final directory
!cp -r /content/LoRA_Trees/output/lora_checkpoint_final \
      //content/drive/MyDrive/LoRATrees2
