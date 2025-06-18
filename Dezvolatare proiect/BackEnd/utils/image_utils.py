import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import requests

def calculate_mask_bbox(mask_path: Path) -> Dict[str, int]:
    """Calculate bounding box from mask image."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Could not read mask image")
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    return {
        "x": x,
        "y": y,
        "width": w,
        "height": h
    }

def crop_and_resize(image_path: Path, bbox: Dict[str, int], target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """Crop image to bounding box and resize to target size."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Crop to bounding box
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    cropped = image[y:y+h, x:x+w]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    return resized

def blend_images(original: np.ndarray, patch: np.ndarray, mask: np.ndarray, bbox: Dict[str, int]) -> np.ndarray:
    """Blend patch into original image using seamless cloning."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    
    # Resize patch to match original mask size
    patch_resized = cv2.resize(patch, (w, h))
    
    # Create center point for seamless cloning
    center = (x + w//2, y + h//2)
    
    # Create a mask for the blending region
    mask_3channel = cv2.merge([mask, mask, mask])
    
    # Perform seamless cloning
    result = cv2.seamlessClone(
        patch_resized,
        original,
        mask_3channel,
        center,
        cv2.NORMAL_CLONE
    )
    
    return result

def save_image(image: np.ndarray, path: Path) -> None:
    """Save image to file."""
    cv2.imwrite(str(path), image)

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
