
!pip install --upgrade git+https://github.com/huggingface/diffusers.git
!pip install accelerate safetensors transformers opencv-python pillow
!pip install --upgrade peft==0.15.1

# === 1.Instal the libralies for Fast API server===
!pip install fastapi uvicorn pyngrok python-multipart
!pip install simple-lama-inpainting

#Server config
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
import nest_asyncio
import shutil
from pyngrok import ngrok
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np
from simple_lama_inpainting import SimpleLama
import io
from PIL import Image, ImageEnhance, ImageFilter

from google.colab import drive
import os


drive.mount('/content/drive')


LORA_DIR = '/content/drive/MyDrive/LoRATrees2/lora_checkpoint_final'
assert os.path.isdir(LORA_DIR), f"❌ LORA dir not found: {LORA_DIR}"
print("✅ Found LoRA adapter in:", LORA_DIR)
print(os.listdir(LORA_DIR))

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
import torch


controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16
)


pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

lora_path = LORA_DIR


pipe.load_lora_weights(lora_path, adapter_name="custom_lora")

app = FastAPI()

@app.post("/generate")
async def generate(sketch: UploadFile = File(...)):
    sketch_path = "sketch.png"
    gen_with_bg_path = "gen_patch_withBG.png"
    final_output_path = "gen_patch.png"

    # 1. Save uploaded sketch
    with open(sketch_path, "wb") as f:
        shutil.copyfileobj(sketch.file, f)

    # 2. Preprocess sketch
    gray = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise HTTPException(status_code=400, detail="Failed to process sketch")

    _, bin_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    bin_mask = cv2.resize(bin_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    control_image = Image.fromarray(np.stack([bin_mask]*3, axis=-1))

    # 3. Generate image
    result = pipe(
        prompt="a professional high-quality photograph of a tree",
        negative_prompt="blurry, low quality, cartoon, drawing, painting, text, watermark",
        image=control_image,
        control_image=control_image,
        num_inference_steps=25,
        guidance_scale=14.0,
        controlnet_conditioning_scale=1.5,
        cross_attention_kwargs={"scale": 1.2}
    ).images[0]

    # 4. Enhance image quality
    result = ImageEnhance.Contrast(result).enhance(1.3)
    result = result.filter(ImageFilter.SHARPEN)
    result.save(final_output_path, quality=95)

    # 5. Verify output exists
    if not os.path.exists(final_output_path):
        raise HTTPException(status_code=500, detail="Failed to generate final output")

    # 6. Cleanup temp files
    for f in [sketch_path, gen_with_bg_path]:
        if os.path.exists(f):
            os.remove(f)

    return FileResponse(final_output_path, media_type="image/png")

@app.post("/inpaint")
async def inpaint(image: UploadFile = File(...), mask: UploadFile = File(...)):
    image_path = "final_result.png"
    mask_path = "new_mask_for_inpainting.png"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    with open(mask_path, "wb") as f:
        shutil.copyfileobj(mask.file, f)

    original = Image.open(image_path).convert("RGB").resize((512, 512))
    mask_img = Image.open(mask_path).convert("L").resize(original.size)

    model = SimpleLama()
    result = model(original, mask_img)

    output_path = "rezultat_final.jpg"
    result.save(output_path)

    return FileResponse(output_path, media_type="image/jpeg")

# Generate ngrok url
nest_asyncio.apply()
from pyngrok import conf, ngrok

conf.get_default().auth_token =  "NGROK TOKEN"

public_url = ngrok.connect(7860)
print("🔥 Public ngrok URL:", public_url)

# # Run server in thread async
nest_asyncio.apply()
uvicorn.run(app, port=7860)
