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
    # 1. Resize mask to same size as original image if needed
    original_shape = (mask.shape[0], mask.shape[1])
    patch_h, patch_w = gen_patch.shape[:2]

    # 2. Resize original mask to match patch region
    mask_patch = cv2.resize(mask, (xmax - xmin, ymax - ymin), interpolation=cv2.INTER_NEAREST)
    
    # 3. Convert gen_patch to grayscale
    patch_gray = cv2.cvtColor(gen_patch, cv2.COLOR_BGR2GRAY)
    
    # 4. Crează noua mască (zero peste tot)
    new_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)

    # 5. Determină părțile din patch care nu sunt în mask (și sunt semnificative)
    difference_area = ((mask_patch == 0) & (patch_gray > 15)).astype(np.uint8) * 255

    # 6. Inserare în noua mască la poziția potrivită
    new_mask[ymin:ymax, xmin:xmax] = difference_area

    # 7. Salvează masca
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
    
    # Find the bounding box of the white area in the mask
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) == 0:
        raise ValueError("No white pixels found in the mask")
    
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
    
    # Generate and save the inpainting mask
    inpainting_mask_path = generate_inpainting_mask(mask, resized_patch, xmin, xmax, ymin, ymax)
    
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
