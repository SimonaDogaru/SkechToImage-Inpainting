import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import requests


def call_colab_generate(colab_url: str, sketch_path: str, output_path: str) -> bool:
    try:
        with open(sketch_path, "rb") as sketch_file:
            files = {"sketch": sketch_file}
            response = requests.post(colab_url, files=files)
        
        print("[DEBUG] Colab response:", response.status_code)

        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print("[ERROR]", response.status_code, response.text[:200])
            return False
    except Exception as e:
        print("[EXCEPTION]", str(e))
        return False

def call_colab_inpaint(colab_url, image_path, mask_path, output_path):
    try:
        with open(image_path, "rb") as image_file, open(mask_path, "rb") as mask_file:
            files = {
                "image": image_file,
                "mask": mask_file
            }
            response = requests.post(f"{colab_url}/inpaint", files=files)
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            print("[✅ INPAINTING OK]")
            return True
        else:
            print(f"[❌ INPAINTING FAIL] {response.status_code} {response.text}")
            return False
    except Exception as e:
        print(f"[❌ INPAINT EXCEPTION] {e}")
        return False
