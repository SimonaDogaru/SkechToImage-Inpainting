from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import os
from pathlib import Path
import cv2
import numpy as np
import requests
from typing import Optional
import uuid
"""
from utils.image_utils import (
    calculate_mask_bbox,
    crop_and_resize,
    blend_images,
    save_image
)

router = APIRouter()

# Constants
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
COLAB_CONTROLNET_URL = os.getenv("COLAB_CONTROLNET_URL", "http://localhost:8888/generate")
COLAB_LAMA_URL = os.getenv("COLAB_LAMA_URL", "http://localhost:8888/inpaint")

@router.post("/process-image")
async def process_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    sketch: UploadFile = File(...),
    do_inpainting: bool = False
):
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        image_path = session_dir / "original.png"
        mask_path = session_dir / "mask.png"
        sketch_path = session_dir / "sketch.png"
        
        # Save files
        with open(image_path, "wb") as f:
            f.write(await image.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())
        with open(sketch_path, "wb") as f:
            f.write(await sketch.read())
        
        # Preprocess images
        bbox = calculate_mask_bbox(mask_path)
        cropped_image = crop_and_resize(image_path, bbox)
        cropped_mask = crop_and_resize(mask_path, bbox)
        cropped_sketch = crop_and_resize(sketch_path, bbox)
        
        # Save preprocessed images
        output_dir = OUTPUT_DIR / session_id
        output_dir.mkdir(exist_ok=True)
        
        save_image(cropped_image, output_dir / "cropped_image.png")
        save_image(cropped_mask, output_dir / "cropped_mask.png")
        save_image(cropped_sketch, output_dir / "cropped_sketch.png")
        
        # Send to ControlNet
        files = {
            'image': ('image.png', open(output_dir / "cropped_image.png", 'rb')),
            'mask': ('mask.png', open(output_dir / "cropped_mask.png", 'rb')),
            'sketch': ('sketch.png', open(output_dir / "cropped_sketch.png", 'rb'))
        }
        
        response = requests.post(COLAB_CONTROLNET_URL, files=files)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="ControlNet generation failed")
        
        # Save generated patch
        gen_patch_path = output_dir / "gen_patch.png"
        with open(gen_patch_path, "wb") as f:
            f.write(response.content)
        
        # Blend images
        original = cv2.imread(str(image_path))
        patch = cv2.imread(str(gen_patch_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        blended = blend_images(original, patch, mask, bbox)
        hybrid_path = output_dir / "hybrid.jpg"
        save_image(blended, hybrid_path)
        
        # Optional inpainting
        if do_inpainting:
            files = {
                'image': ('image.jpg', open(hybrid_path, 'rb')),
                'mask': ('mask.png', open(mask_path, 'rb'))
            }
            
            response = requests.post(COLAB_LAMA_URL, files=files)
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="LaMa inpainting failed")
            
            final_path = output_dir / "final.jpg"
            with open(final_path, "wb") as f:
                f.write(response.content)
            
            return FileResponse(final_path)
        
        return FileResponse(hybrid_path)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup files
        if 'files' in locals():
            for file in files.values():
                file[1].close() 
"""