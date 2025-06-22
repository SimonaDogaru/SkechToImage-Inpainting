import cv2
import numpy as np
import os
import time

def wait_for_gen_patch(timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists('outputs/gen_patch.png'):
            return True
        time.sleep(1)
    return False

def generate_inpainting_mask(final_image, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Creates a mask from the final image. Pixels that are pure white 
    (255, 255, 255) will be white in the mask. All other pixels will be black.
    """
    # Create a binary mask where white pixels from the source are white, and all others are black.
    white_pixels_mask = cv2.inRange(final_image, (255, 255, 255), (255, 255, 255))
    
    cv2.imwrite(output_path, white_pixels_mask)
    return output_path

def adjust_colors_and_lighting(original_region, generated_region, mask_region):
    """
    Ajustează culorile și iluminarea pentru tranziții mai naturale.
    """
    # Convertește la float pentru calcule precise
    original_float = original_region.astype(np.float32) / 255.0
    generated_float = generated_region.astype(np.float32) / 255.0
    
    # Calculează statistici pentru fiecare canal
    mask_bool = mask_region > 0
    
    if np.any(mask_bool):
        # Calculează media și deviația standard pentru zona originală
        original_mean = np.mean(original_float[mask_bool], axis=0)
        original_std = np.std(original_float[mask_bool], axis=0)
        
        # Calculează media și deviația standard pentru zona generată
        generated_mean = np.mean(generated_float[mask_bool], axis=0)
        generated_std = np.std(generated_float[mask_bool], axis=0)
        
        # Aplică histogram matching pentru fiecare canal
        adjusted_generated = np.zeros_like(generated_float)
        
        for channel in range(3):
            if original_std[channel] > 0 and generated_std[channel] > 0:
                # Normalizează și rescalează
                normalized = (generated_float[:, :, channel] - generated_mean[channel]) / generated_std[channel]
                adjusted_generated[:, :, channel] = normalized * original_std[channel] + original_mean[channel]
            else:
                adjusted_generated[:, :, channel] = generated_float[:, :, channel]
        
        # Aplică ajustarea doar în zona măștii
        result = original_float.copy()
        result[mask_bool] = adjusted_generated[mask_bool]
        
        # Convertește înapoi la uint8
        return (result * 255).astype(np.uint8)
    
    return original_region

def combine_images():
    if not wait_for_gen_patch():
        raise TimeoutError("Timeout waiting for gen_patch.png to be generated")

    original = cv2.imread('uploads/original.jpg')
    mask = cv2.imread('uploads/mask.png', cv2.IMREAD_GRAYSCALE)
    gen_patch = cv2.imread('outputs/gen_patch.png')

    if original is None or mask is None or gen_patch is None:
        raise FileNotFoundError("One or more input images could not be loaded")

    # Resize both original and mask to 512x512
    original = cv2.resize(original, (512, 512))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

    print(f"[DEBUG] original shape: {original.shape}")
    print(f"[DEBUG] mask shape: {mask.shape}")
    print(f"[DEBUG] gen_patch shape: {gen_patch.shape}")

    # Găsește bounding box pentru optimizare
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) == 0:
        raise ValueError("No white pixels found in the mask")

    ymin, ymax = np.min(white_pixels[0]), np.max(white_pixels[0])
    xmin, xmax = np.min(white_pixels[1]), np.max(white_pixels[1])

    print(f"[DEBUG] bbox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    patch_width = xmax - xmin
    patch_height = ymax - ymin

    # Redimensionează patch-ul generat la dimensiunea bounding box-ului
    resized_patch = cv2.resize(gen_patch, (patch_width, patch_height))
    
    # Creează o mască pentru zona de îmbinare (doar zona alba din masca originală)
    blend_mask = mask[ymin:ymax, xmin:xmax]
    
    # Creează o mască cu gradient pentru tranziții mai naturale
    # Aplică blur pentru a crea tranziții smooth
    smooth_mask = cv2.GaussianBlur(blend_mask, (15, 15), 0)
    
    # Normalizează masca la 0-1 pentru blending
    blend_mask_normalized = smooth_mask.astype(np.float32) / 255.0
    
    # Extinde masca la 3 canale pentru blending
    blend_mask_3ch = np.stack([blend_mask_normalized] * 3, axis=2)
    
    # Creează o copie a imaginii originale
    result = original.copy()
    
    # Extrage zona din imaginea originală
    original_region = original[ymin:ymax, xmin:xmax]
    
    # Aplică blending cu tranziții naturale
    # Folosește o combinație de blending liniar și seamless cloning
    blended_region = (resized_patch * blend_mask_3ch + 
                     original_region * (1 - blend_mask_3ch)).astype(np.uint8)
    
    # Aplică un blur suplimentar la margini pentru tranziții mai smooth
    # Creează o mască pentru margini
    edge_mask = cv2.Canny(blend_mask, 50, 150)
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=2)
    
    # Aplică blur doar la margini
    if np.any(edge_mask > 0):
        edge_blur = cv2.GaussianBlur(blended_region, (5, 5), 0)
        edge_mask_3ch = np.stack([edge_mask.astype(np.float32) / 255.0] * 3, axis=2)
        blended_region = (edge_blur * edge_mask_3ch + 
                         blended_region * (1 - edge_mask_3ch)).astype(np.uint8)
    
    # Înlocuiește zona în rezultat
    result[ymin:ymax, xmin:xmax] = blended_region

    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite('outputs/final_result.png', result)

    inpainting_mask_path = generate_inpainting_mask(result)

    return 'outputs/final_result.png', inpainting_mask_path

def auto_combine_after_upload():
    try:
        return combine_images()
    except Exception as e:
        print(f"Error during image combination: {str(e)}")
        return None
