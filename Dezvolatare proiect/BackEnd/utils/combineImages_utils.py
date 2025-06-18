import cv2
import numpy as np
import os
import time

def wait_for_gen_patch(timeout=300):  # 5 minutes timeout
    """
    Waits for gen_patch.png to be available in the outputs directory
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists('outputs/gen_patch.png'):
            return True
        time.sleep(1)  # Check every second
    return False

def generate_inpainting_mask(mask, gen_patch, xmin, xmax, ymin, ymax, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Creează o mască nouă pentru inpainting: marchează tot ce este vizibil din patch dar nu era în masca originală.
    """
    # Verificare dimensiuni bounding box
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box dimensions: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    # Dimensiuni patch
    patch_width = xmax - xmin
    patch_height = ymax - ymin

    print(f"[DEBUG] patch bbox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    print(f"[DEBUG] patch size: width={patch_width}, height={patch_height}")

    if patch_width <= 0 or patch_height <= 0:
        raise ValueError("Invalid bounding box dimensions")

    # Redimensionează patch-ul generat și masca
    resized_patch = cv2.resize(gen_patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (patch_width, patch_height), interpolation=cv2.INTER_NEAREST)

    # Convertim patch-ul la grayscale pentru a extrage zonele vizibile
    patch_gray = cv2.cvtColor(resized_patch, cv2.COLOR_BGR2GRAY)

    # Determină zonele din patch care nu sunt acoperite de mască și conțin conținut (valori > 15)
    difference_area = ((resized_mask == 0) & (patch_gray > 15)).astype(np.uint8) * 255

    # Verificare dimensiuni difference_area
    if difference_area.shape[0] != (ymax - ymin) or difference_area.shape[1] != (xmax - xmin):
        print(f"[ERROR] Shape mismatch: diff_area={difference_area.shape}, target={(ymax - ymin, xmax - xmin)}")
        return None

    # Construim noua mască la dimensiunea inițială
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[ymin:ymax, xmin:xmax] = difference_area

    # Salvăm masca nouă
    cv2.imwrite(output_path, new_mask)

    return output_path

def combine_images():
    """
    Combines the original image with the generated patch using the mask.
    The function assumes that gen_patch.png is already generated and available in the output directory.
    """
    # Wait for gen_patch.png to be available
    if not wait_for_gen_patch():
        raise TimeoutError("Timeout waiting for gen_patch.png to be generated")

    # Read the input images
    original = cv2.imread('uploads/original.jpg')
    mask = cv2.imread('uploads/mask.png', cv2.IMREAD_GRAYSCALE)
    gen_patch = cv2.imread('outputs/gen_patch.png')
    
    if original is None or mask is None or gen_patch is None:
        raise FileNotFoundError("One or more input images could not be loaded")
    
    # Resize mask to match original image dimensions
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Debug prints for image shapes
    print(f"[DEBUG] original shape: {original.shape}")
    print(f"[DEBUG] mask shape: {mask.shape}")
    print(f"[DEBUG] gen_patch shape: {gen_patch.shape}")
    
    # Find the bounding box of the white area in the mask
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) == 0:
        raise ValueError("No white pixels found in the mask (255 values). Cannot compute bounding box.")
    
    # Dimensiuni mască și imagine originală
    mask_h, mask_w = mask.shape
    orig_h, orig_w = original.shape[:2]

    # Bounding box în coordonate de pe mască
    ymin_m, ymax_m = np.min(white_pixels[0]), np.max(white_pixels[0])
    xmin_m, xmax_m = np.min(white_pixels[1]), np.max(white_pixels[1])

    # Scaling bounding box la dimensiunea imaginii originale
    xmin = int(xmin_m * orig_w / mask_w)
    xmax = int(xmax_m * orig_w / mask_w)
    ymin = int(ymin_m * orig_h / mask_h)
    ymax = int(ymax_m * orig_h / mask_h)
    
    # Debug print for bounding box
    print(f"[DEBUG] bbox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    print(f"[DEBUG] patch size: width={xmax - xmin}, height={ymax - ymin}")
    
    # Validate bounding box
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    
    # Calculate the dimensions of the patch after scaling
    patch_height = ymax - ymin
    patch_width = xmax - xmin
    
    # Resize the generated patch to match the mask area
    resized_patch = cv2.resize(gen_patch, (patch_width, patch_height))
    
    # Create a copy of the original image
    result = original.copy()
    
    # Replace the area in the original image with the resized patch
    result[ymin:ymax, xmin:xmax] = resized_patch
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save the final result
    cv2.imwrite('outputs/final_result.png', result)
    
    # Generate and save the inpainting mask using the original gen_patch
    inpainting_mask_path = generate_inpainting_mask(mask, gen_patch, xmin, xmax, ymin, ymax)
    
    return 'outputs/final_result.png', inpainting_mask_path

def auto_combine_after_upload():
    """
    Automatically runs the combine_images function after upload
    """
    try:
        return combine_images()
    except Exception as e:
        print(f"Error during image combination: {str(e)}")
        return None
