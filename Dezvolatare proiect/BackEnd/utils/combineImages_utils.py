import cv2
import numpy as np
import os
import time
from pathlib import Path
import shutil

def wait_for_gen_patch(timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists('outputs/gen_patch.png'):
            return True
        time.sleep(1)
    return False

def generate_inpainting_mask(final_image, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Creates a mask from the final image. Pixels that are very light/white 
    (close to white) will be white in the mask. All other pixels will be black.
    """
    # Convert to HSV for better color detection
    hsv_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for very light/white areas
    # Lower bound for white/light colors in HSV
    lower_white = np.array([0, 0, 200])  # High value (V), low saturation (S)
    upper_white = np.array([180, 30, 255])  # Allow some hue variation
    
    # Create mask for light areas
    light_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    
    # Also detect pure white pixels as backup
    pure_white_mask = cv2.inRange(final_image, (255, 255, 255), (255, 255, 255))
    
    # Combine both masks
    combined_mask = cv2.bitwise_or(light_mask, pure_white_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Save debug information
    print(f"[DEBUG] Light mask pixels: {np.sum(light_mask > 0)}")
    print(f"[DEBUG] Pure white mask pixels: {np.sum(pure_white_mask > 0)}")
    print(f"[DEBUG] Combined mask pixels: {np.sum(combined_mask > 0)}")
    
    cv2.imwrite(output_path, combined_mask)
    return output_path

def generate_inpainting_mask_alternative(final_image, original_mask_path='uploads/mask.png', output_path='outputs/new_mask_for_inpainting.png'):
    """
    Alternative method to create inpainting mask using multiple detection strategies.
    This function tries to detect areas that need inpainting using both color-based
    detection and the original mask as reference.
    """
    # Method 1: Color-based detection (HSV)
    hsv_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)
    
    # Detect very light areas
    lower_light = np.array([0, 0, 180])  # Lower threshold for light detection
    upper_light = np.array([180, 50, 255])
    light_mask = cv2.inRange(hsv_image, lower_light, upper_light)
    
    # Method 2: RGB-based detection for light colors
    # Detect pixels where all channels are above a threshold
    rgb_light_mask = np.all(final_image > 180, axis=2).astype(np.uint8) * 255
    
    # Method 3: Use original mask as reference (if available)
    original_mask = None
    try:
        original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        if original_mask is not None:
            original_mask = cv2.resize(original_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    except:
        print("[WARNING] Could not load original mask for reference")
    
    # Combine all masks
    combined_mask = cv2.bitwise_or(light_mask, rgb_light_mask)
    
    if original_mask is not None:
        # Use original mask as a guide - expand it slightly
        kernel = np.ones((5, 5), np.uint8)
        expanded_original = cv2.dilate(original_mask, kernel, iterations=1)
        combined_mask = cv2.bitwise_or(combined_mask, expanded_original)
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Ensure we have some white pixels
    if np.sum(combined_mask > 0) == 0:
        print("[WARNING] No pixels detected in inpainting mask, using original mask")
        if original_mask is not None:
            combined_mask = original_mask
        else:
            # Fallback: use a simple threshold
            gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
            _, combined_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    print(f"[DEBUG] Alternative mask pixels: {np.sum(combined_mask > 0)}")
    
    cv2.imwrite(output_path, combined_mask)
    return output_path

def generate_inpainting_mask_improved(final_image, original_mask_path='uploads/mask.png', output_path='outputs/new_mask_for_inpainting.png'):
    """
    Improved inpainting mask generation that prioritizes RGB light detection
    since this method was found to be the most effective.
    """
    # Load and resize original mask
    original_mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is not None:
        original_mask = cv2.resize(original_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        print(f"[DEBUG] Original mask loaded, white pixels: {np.sum(original_mask == 255)}")
    else:
        print("[WARNING] Could not load original mask")
        original_mask = np.zeros((512, 512), dtype=np.uint8)
    
    # PRIMARY METHOD: RGB-based detection for light colors (found to be the best)
    # Detect pixels where all channels are above a threshold
    rgb_light_mask = np.all(final_image > 180, axis=2).astype(np.uint8) * 255
    
    # Use original mask as a guide to focus the detection
    if np.sum(original_mask > 0) > 0:
        # Dilate original mask slightly to capture nearby areas
        kernel = np.ones((5, 5), np.uint8)
        dilated_original = cv2.dilate(original_mask, kernel, iterations=1)
        
        # Combine RGB light detection with dilated original mask
        result_mask = cv2.bitwise_and(rgb_light_mask, dilated_original)
        
        # Also include the original mask itself
        result_mask = cv2.bitwise_or(result_mask, original_mask)
    else:
        # If no original mask, use RGB light detection directly
        result_mask = rgb_light_mask
    
    # Clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel)
    
    # Ensure we have some white pixels
    if np.sum(result_mask > 0) == 0:
        print("[WARNING] No pixels detected in inpainting mask, using original mask")
        result_mask = original_mask
    
    print(f"[DEBUG] RGB light mask pixels: {np.sum(rgb_light_mask > 0)}")
    print(f"[DEBUG] Original mask pixels: {np.sum(original_mask > 0)}")
    print(f"[DEBUG] Final result mask pixels: {np.sum(result_mask > 0)}")
    
    cv2.imwrite(output_path, result_mask)
    return output_path

def generate_inpainting_mask_rgb_adjustable(final_image, threshold=180, output_path='outputs/new_mask_for_inpainting.png'):
    """
    RGB-based inpainting mask generation with adjustable threshold.
    Also covers transparent pixels (pixels with value 0 in all channels).
    Extends mask borders slightly while preserving internal structure.
    """
    # RGB-based detection for light colors with adjustable threshold
    rgb_light_mask = np.all(final_image > threshold, axis=2).astype(np.uint8) * 255
    
    # Detect transparent pixels (pixels with value 0 in all channels)
    transparent_mask = np.all(final_image == 0, axis=2).astype(np.uint8) * 255
    
    # Combine light pixels and transparent pixels
    combined_mask = cv2.bitwise_or(rgb_light_mask, transparent_mask)
    
    # Extend mask borders slightly while preserving internal structure
    result_mask = extend_mask_borders_preserve_structure(combined_mask, border_extension=2, preserve_internal=True)
    
    print(f"[DEBUG] RGB light mask pixels (threshold={threshold}): {np.sum(rgb_light_mask > 0)}")
    print(f"[DEBUG] Transparent pixels: {np.sum(transparent_mask > 0)}")
    print(f"[DEBUG] Combined mask pixels: {np.sum(combined_mask > 0)}")
    print(f"[DEBUG] Final extended mask pixels: {np.sum(result_mask > 0)}")
    
    cv2.imwrite(output_path, result_mask)
    return output_path

def generate_inpainting_mask_rgb_only(final_image, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Simplified inpainting mask generation using only RGB light detection
    since this method was found to be the most effective.
    """
    return generate_inpainting_mask_rgb_adjustable(final_image, threshold=180, output_path=output_path)

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

def combine_images(gen_patch_path='outputs/gen_patch_NObg_onWhite.png'):
    if not wait_for_gen_patch():
        raise TimeoutError("Timeout waiting for gen_patch.png to be generated")

    original = cv2.imread('uploads/original.jpg')
    mask = cv2.imread('uploads/mask.png', cv2.IMREAD_GRAYSCALE)
    gen_patch = cv2.imread(gen_patch_path)

    if original is None or mask is None or gen_patch is None:
        raise FileNotFoundError("One or more input images could not be loaded")

    # Analyze original mask first
    print("[DEBUG] Analyzing original mask...")
    mask_analysis = analyze_original_mask('uploads/mask.png')
    print(f"[DEBUG] Mask analysis: {mask_analysis}")

    # Resize both original and mask to 512x512
    original = cv2.resize(original, (512, 512))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    gen_patch = cv2.resize(gen_patch, (512, 512))

    print(f"[DEBUG] original shape: {original.shape}")
    print(f"[DEBUG] mask shape: {mask.shape}")
    print(f"[DEBUG] gen_patch shape: {gen_patch.shape}")
    print(f"[DEBUG] Using gen_patch from: {gen_patch_path}")

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
    
    # Aplică blending cu tranziții naturale (patch-ul are deja culorile transferate)
    # Folosește o combinație de blending liniar și seamless cloning
    blended_region = (resized_patch * blend_mask_3ch + 
                     original_region * (1 - blend_mask_3ch)).astype(np.uint8)
    
    # Update the result with blended region
    result[ymin:ymax, xmin:xmax] = blended_region

    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite('outputs/final_result.png', result)

    # Înlocuiește toți pixelii (255,255,255) cu cei din original și salvează ca final_result_adjusted.png
    print("[🔄 REPLACING PURE WHITE PIXELS WITH ORIGINAL]")
    white_pixel_mask = np.all(result == 255, axis=2)
    adjusted_result = result.copy()
    if np.sum(white_pixel_mask) > 0:
        adjusted_result[white_pixel_mask] = original[white_pixel_mask]
        print(f"[DEBUG] Replaced {np.sum(white_pixel_mask)} pure white pixels with original pixels")
    else:
        print("[DEBUG] No pure white pixels to replace detected")
    cv2.imwrite('outputs/final_result_adjusted.png', adjusted_result)

    # Use the enhanced inpainting mask that combines border and nearby transparent pixels
    print("[🔄 CREATING ENHANCED INPAINTING MASK]")
    inpainting_mask_path = create_enhanced_inpainting_mask(
        uploaded_mask_path='uploads/mask.png',
        gen_patch_nobg_path=gen_patch_path,  # Use the no-background version
        border_width=8,
        transparent_distance=1,
        output_path='outputs/new_mask_for_inpainting.png'
    )
    
    if inpainting_mask_path is None:
        # Fallback to border mask if enhanced mask creation fails
        print("[⚠️ ENHANCED MASK CREATION FAILED] Using border mask as fallback")
        inpainting_mask_path = create_border_mask_from_uploaded_mask(border_width=8)
        if inpainting_mask_path is None:
            # Final fallback to generated mask
            inpainting_mask_path = generate_inpainting_mask_rgb_adjustable(adjusted_result)
            print(f"[DEBUG] Border mask creation also failed, using generated mask")
    else:
        print(f"[DEBUG] Successfully created enhanced inpainting mask")

    return 'outputs/final_result_adjusted.png', inpainting_mask_path

def auto_combine_after_upload(gen_patch_path='outputs/gen_patch.png'):
    try:
        return combine_images(gen_patch_path=gen_patch_path)
    except Exception as e:
        print(f"Error during image combination: {str(e)}")
        return None

def analyze_original_mask(mask_path='uploads/mask.png'):
    """
    Analyze the original mask to understand what areas should be inpainted.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[ERROR] Could not read mask from {mask_path}")
        return None
    
    # Resize to 512x512 for consistency
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    print(f"[DEBUG] Original mask shape: {mask.shape}")
    print(f"[DEBUG] Original mask min/max: {np.min(mask)}/{np.max(mask)}")
    print(f"[DEBUG] Original mask mean: {np.mean(mask)}")
    print(f"[DEBUG] White pixels in original mask: {np.sum(mask == 255)}")
    print(f"[DEBUG] Non-zero pixels in original mask: {np.sum(mask > 0)}")
    
    # Find bounding box of white pixels
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) > 0:
        ymin, ymax = np.min(white_pixels[0]), np.max(white_pixels[0])
        xmin, xmax = np.min(white_pixels[1]), np.max(white_pixels[1])
        print(f"[DEBUG] White area bbox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
        print(f"[DEBUG] White area size: {xmax-xmin} x {ymax-ymin}")
    
    return {
        'shape': mask.shape,
        'white_pixels': np.sum(mask == 255),
        'non_zero_pixels': np.sum(mask > 0),
        'bbox': (xmin, xmax, ymin, ymax) if len(white_pixels[0]) > 0 else None
    }

def use_debug_mask_as_inpainting_mask(debug_mask_path='outputs/debug_replacement_mask.png', output_path='outputs/new_mask_for_inpainting.png'):
    """
    Use the debug replacement mask as the inpainting mask.
    This allows manual control over which areas should be inpainted.
    """
    debug_path = Path(debug_mask_path)
    output_path_obj = Path(output_path)
    
    if debug_path.exists():
        # Copy the debug mask as the inpainting mask
        shutil.copy2(debug_path, output_path_obj)
        print(f"[DEBUG] Using {debug_mask_path} as inpainting mask")
        print(f"[DEBUG] Copied to {output_path}")
        return output_path
    else:
        print(f"[WARNING] Debug mask {debug_mask_path} not found")
        return None

def force_use_debug_mask_as_inpainting_mask(debug_mask_path='outputs/debug_replacement_mask.png', output_path='outputs/new_mask_for_inpainting.png'):
    """
    Force using the debug replacement mask as the inpainting mask.
    This will use the debug mask even if no replacement pixels were found.
    """
    debug_path = Path(debug_mask_path)
    output_path_obj = Path(output_path)
    
    if debug_path.exists():
        # Copy the debug mask as the inpainting mask
        shutil.copy2(debug_path, output_path_obj)
        print(f"[DEBUG] FORCED: Using {debug_mask_path} as inpainting mask")
        print(f"[DEBUG] Copied to {output_path}")
        return output_path
    else:
        print(f"[WARNING] Debug mask {debug_mask_path} not found")
        return None

def extend_mask_borders_preserve_structure(mask, border_extension=2, preserve_internal=True):
    """
    Extend mask borders slightly while preserving internal structure.
    
    Args:
        mask: Input binary mask
        border_extension: Number of pixels to extend borders (default: 2)
        preserve_internal: Whether to preserve internal black areas (default: True)
    
    Returns:
        Extended mask with preserved internal structure
    """
    # Create a copy of the original mask
    original_mask = mask.copy()
    
    # Dilate to extend borders
    kernel_dilate = np.ones((border_extension * 2 + 1, border_extension * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    if preserve_internal:
        # Find internal black areas (holes) in the original mask
        # These are areas that are black in the original mask but surrounded by white
        kernel_fill = np.ones((3, 3), np.uint8)
        filled_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel_fill)
        
        # The holes are the difference between filled and original mask
        holes = cv2.subtract(filled_mask, original_mask)
        
        # Remove holes from the dilated mask to preserve internal structure
        result_mask = cv2.subtract(dilated_mask, holes)
    else:
        result_mask = dilated_mask
    
    # Clean up the result
    kernel_clean = np.ones((3, 3), np.uint8)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_CLOSE, kernel_clean)
    result_mask = cv2.morphologyEx(result_mask, cv2.MORPH_OPEN, kernel_clean)
    
    print(f"[DEBUG] Original mask pixels: {np.sum(original_mask > 0)}")
    print(f"[DEBUG] Dilated mask pixels: {np.sum(dilated_mask > 0)}")
    if preserve_internal:
        print(f"[DEBUG] Holes detected: {np.sum(holes > 0)}")
    print(f"[DEBUG] Final extended mask pixels: {np.sum(result_mask > 0)}")
    
    return result_mask

def use_debug_mask_as_inpainting_mask_with_extension(debug_mask_path='outputs/debug_replacement_mask.png', output_path='outputs/new_mask_for_inpainting.png', border_extension=2):
    """
    Use the debug replacement mask as the inpainting mask with border extension.
    This allows manual control over which areas should be inpainted and extends borders slightly.
    """
    debug_path = Path(debug_mask_path)
    output_path_obj = Path(output_path)
    
    if debug_path.exists():
        # Read the debug mask
        debug_mask = cv2.imread(str(debug_path), cv2.IMREAD_GRAYSCALE)
        if debug_mask is None:
            print(f"[ERROR] Could not read debug mask from {debug_mask_path}")
            return None
        
        # Extend mask borders while preserving internal structure
        extended_mask = extend_mask_borders_preserve_structure(debug_mask, border_extension=border_extension, preserve_internal=True)
        
        # Save the extended mask as the inpainting mask
        cv2.imwrite(str(output_path_obj), extended_mask)
        
        print(f"[DEBUG] Using {debug_mask_path} as inpainting mask with border extension")
        print(f"[DEBUG] Extended and saved to {output_path}")
        print(f"[DEBUG] Original mask pixels: {np.sum(debug_mask > 0)}")
        print(f"[DEBUG] Extended mask pixels: {np.sum(extended_mask > 0)}")
        
        return output_path
    else:
        print(f"[WARNING] Debug mask {debug_mask_path} not found")
        return None

def create_border_mask_from_uploaded_mask(uploaded_mask_path='uploads/mask.png', border_width=8, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Create a mask where only the borders of the uploaded mask are white.
    This allows inpainting only on the edges of the original mask area.
    
    Args:
        uploaded_mask_path: Path to the uploaded mask
        border_width: Width of the border to extract (default: 8)
        output_path: Output path for the border mask
    
    Returns:
        Path to the created border mask
    """
    # Load the uploaded mask
    uploaded_mask = cv2.imread(uploaded_mask_path, cv2.IMREAD_GRAYSCALE)
    if uploaded_mask is None:
        print(f"[ERROR] Could not read uploaded mask from {uploaded_mask_path}")
        return None
    
    # Resize to 512x512 for consistency
    uploaded_mask = cv2.resize(uploaded_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Create a dilated version of the mask
    kernel_dilate = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(uploaded_mask, kernel_dilate, iterations=1)
    
    # Create an eroded version of the mask
    kernel_erode = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(uploaded_mask, kernel_erode, iterations=1)
    
    # The border is the difference between dilated and eroded masks
    border_mask = cv2.subtract(dilated_mask, eroded_mask)
    
    # Clean up the border mask
    kernel_clean = np.ones((3, 3), np.uint8)
    border_mask = cv2.morphologyEx(border_mask, cv2.MORPH_CLOSE, kernel_clean)
    
    print(f"[DEBUG] Uploaded mask pixels: {np.sum(uploaded_mask > 0)}")
    print(f"[DEBUG] Border mask pixels: {np.sum(border_mask > 0)}")
    
    # Save the border mask as the inpainting mask
    cv2.imwrite(output_path, border_mask)
    
    return output_path

def create_border_mask_adjustable(uploaded_mask_path='uploads/mask.png', border_width=3, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Create a border mask with adjustable width from the uploaded mask.
    
    Args:
        uploaded_mask_path: Path to the uploaded mask
        border_width: Width of the border (default: 3)
        output_path: Output path for the border mask
    
    Returns:
        Path to the created border mask
    """
    # Load the uploaded mask
    uploaded_mask = cv2.imread(uploaded_mask_path, cv2.IMREAD_GRAYSCALE)
    if uploaded_mask is None:
        print(f"[ERROR] Could not read uploaded mask from {uploaded_mask_path}")
        return None
    
    # Resize to 512x512 for consistency
    uploaded_mask = cv2.resize(uploaded_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    
    # Method 1: Using dilation and erosion difference
    kernel_dilate = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(uploaded_mask, kernel_dilate, iterations=1)
    
    kernel_erode = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(uploaded_mask, kernel_erode, iterations=1)
    
    border_mask = cv2.subtract(dilated_mask, eroded_mask)
    
    # Method 2: Using edge detection as alternative
    # Find edges in the uploaded mask
    edges = cv2.Canny(uploaded_mask, 50, 150)
    edges = cv2.dilate(edges, np.ones((border_width, border_width), np.uint8), iterations=1)
    
    # Combine both methods for better border detection
    combined_border = cv2.bitwise_or(border_mask, edges)
    
    # Clean up the border mask
    kernel_clean = np.ones((3, 3), np.uint8)
    final_border_mask = cv2.morphologyEx(combined_border, cv2.MORPH_CLOSE, kernel_clean)
    
    print(f"[DEBUG] Uploaded mask pixels: {np.sum(uploaded_mask > 0)}")
    print(f"[DEBUG] Border width: {border_width}")
    print(f"[DEBUG] Final border mask pixels: {np.sum(final_border_mask > 0)}")
    
    # Save the final border mask as the inpainting mask
    cv2.imwrite(output_path, final_border_mask)
    
    return output_path

def create_enhanced_inpainting_mask(uploaded_mask_path='uploads/mask.png', gen_patch_nobg_path='outputs/gen_patch_NObg.png', 
                                   border_width=8, transparent_distance=1, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Create an enhanced inpainting mask that combines:
    1. The border of the original uploaded mask
    2. Transparent pixels from gen_patch_NObg.png that are close to the object (1-10 pixels away)
    
    Args:
        uploaded_mask_path: Path to the uploaded mask
        gen_patch_nobg_path: Path to the no-background generated patch
        border_width: Width of the border from original mask (default: 8)
        transparent_distance: Distance to look for transparent pixels around object (default: 1)
        output_path: Output path for the enhanced mask
    
    Returns:
        Path to the created enhanced mask
    """
    # Load the uploaded mask
    uploaded_mask = cv2.imread(uploaded_mask_path, cv2.IMREAD_GRAYSCALE)
    if uploaded_mask is None:
        print(f"[ERROR] Could not read uploaded mask from {uploaded_mask_path}")
        return None
    
    # Load the no-background patch
    gen_patch_nobg = cv2.imread(gen_patch_nobg_path)
    if gen_patch_nobg is None:
        print(f"[ERROR] Could not read no-background patch from {gen_patch_nobg_path}")
        return None
    
    # Resize both to 512x512 for consistency
    uploaded_mask = cv2.resize(uploaded_mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    gen_patch_nobg = cv2.resize(gen_patch_nobg, (512, 512))
    
    print(f"[DEBUG] Processing enhanced inpainting mask")
    print(f"[DEBUG] Border width: {border_width}")
    print(f"[DEBUG] Transparent pixel search distance: {transparent_distance}")
    
    # Step 1: Create border mask from uploaded mask
    kernel_dilate = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    dilated_mask = cv2.dilate(uploaded_mask, kernel_dilate, iterations=1)
    
    kernel_erode = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    eroded_mask = cv2.erode(uploaded_mask, kernel_erode, iterations=1)
    
    border_mask = cv2.subtract(dilated_mask, eroded_mask)
    
    # Step 2: Detect transparent pixels in the no-background patch
    transparent_mask = np.all(gen_patch_nobg == 0, axis=2).astype(np.uint8) * 255
    
    # Step 3: Create a mask of the object (non-transparent pixels) from the no-background patch
    object_mask = np.any(gen_patch_nobg > 0, axis=2).astype(np.uint8) * 255
    
    # Step 4: Dilate the object mask to find transparent pixels that are close to the object
    kernel_transparent = np.ones((transparent_distance * 2 + 1, transparent_distance * 2 + 1), np.uint8)
    dilated_object = cv2.dilate(object_mask, kernel_transparent, iterations=1)
    
    # Step 5: Find transparent pixels that are within the dilated object area
    nearby_transparent_mask = cv2.bitwise_and(transparent_mask, dilated_object)
    
    # Step 6: Combine border mask with nearby transparent pixels
    enhanced_mask = cv2.bitwise_or(border_mask, nearby_transparent_mask)
    
    # Step 7: Clean up the final mask
    kernel_clean = np.ones((3, 3), np.uint8)
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_CLOSE, kernel_clean)
    enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, kernel_clean)
    
    # Debug information
    print(f"[DEBUG] Uploaded mask pixels: {np.sum(uploaded_mask > 0)}")
    print(f"[DEBUG] Border mask pixels: {np.sum(border_mask > 0)}")
    print(f"[DEBUG] Transparent pixels in no-bg patch: {np.sum(transparent_mask > 0)}")
    print(f"[DEBUG] Object pixels in no-bg patch: {np.sum(object_mask > 0)}")
    print(f"[DEBUG] Nearby transparent pixels: {np.sum(nearby_transparent_mask > 0)}")
    print(f"[DEBUG] Final enhanced mask pixels: {np.sum(enhanced_mask > 0)}")
    
    # Save the enhanced mask
    cv2.imwrite(output_path, enhanced_mask)
    
    return output_path

def color_transfer(source, target):
    """
    Transfer color palette from source image to target image using LAB color space.
    
    Args:
        source: Source image (the one whose colors we want to transfer)
        target: Target image (the one that will receive the color palette)
    
    Returns:
        Target image with source color palette applied
    """
    try:
        # Check if images are valid
        if source is None or target is None:
            print("[ERROR] One or both images are None in color_transfer")
            return target
        
        if source.size == 0 or target.size == 0:
            print("[ERROR] One or both images are empty in color_transfer")
            return target
        
        print(f"[DEBUG] Color transfer - Source shape: {source.shape}, Target shape: {target.shape}")
        
        # Convert to LAB color space
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        # Compute mean and std for each channel
        (l_mean_src, a_mean_src, b_mean_src), (l_std_src, a_std_src, b_std_src) = cv2.meanStdDev(source)
        (l_mean_tar, a_mean_tar, b_mean_tar), (l_std_tar, a_std_tar, b_std_tar) = cv2.meanStdDev(target)

        # Check for zero standard deviations to avoid division by zero
        if l_std_src[0][0] == 0 or a_std_src[0][0] == 0 or b_std_src[0][0] == 0:
            print("[WARNING] Zero standard deviation detected, returning target as is")
            return cv2.cvtColor(target.astype("uint8"), cv2.COLOR_LAB2BGR)

        # Subtract the mean from source
        l, a, b = cv2.split(source)
        
        # Fix array indexing - cv2.meanStdDev returns 2D arrays, so we need [0][0]
        l = ((l - l_mean_src[0][0]) * (l_std_tar[0][0] / l_std_src[0][0])) + l_mean_tar[0][0]
        a = ((a - a_mean_src[0][0]) * (a_std_tar[0][0] / a_std_src[0][0])) + a_mean_tar[0][0]
        b = ((b - b_mean_src[0][0]) * (b_std_tar[0][0] / b_std_src[0][0])) + b_mean_tar[0][0]

        # Merge and convert back
        transfer = cv2.merge([l, a, b])
        transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
        
        print("[DEBUG] Color transfer completed successfully")
        return transfer
        
    except Exception as e:
        print(f"[ERROR] Color transfer failed: {e}")
        # Return target as fallback
        if target is not None:
            return cv2.cvtColor(target.astype("uint8"), cv2.COLOR_LAB2BGR) if len(target.shape) == 3 else target
        return None

def place_patch_on_white_background(patch_path='outputs/gen_patch_NObg.png', output_path='outputs/gen_patch_NObg_onWhite.png'):
    """
    Plasează patch-ul fără background pe un fundal alb de aceleași dimensiuni și salvează rezultatul.
    Asigură că toți pixelii (0,255,0) din patch rămân (0,255,0) în rezultatul final.
    """
    if not os.path.exists(patch_path):
        print(f"[ERROR] Patch file {patch_path} does not exist.")
        return None
    patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    if patch is None:
        print(f"[ERROR] Could not read patch from {patch_path}")
        return None
    h, w = patch.shape[:2]
    # Creează imagine albă
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    # Dacă patch-ul are canal alpha, folosește-l pentru compunere
    if patch.shape[2] == 4:
        alpha = patch[:, :, 3] / 255.0
        for c in range(3):
            white_bg[:, :, c] = (patch[:, :, c] * alpha + white_bg[:, :, c] * (1 - alpha)).astype(np.uint8)
    else:
        # Pentru patch RGB, suprascrie doar pixelii nenuli
        mask = np.any(patch > 0, axis=2)
        white_bg[mask] = patch[mask]
    # Asigură-te că toți pixelii (0,255,0) din patch rămân (0,255,0) în rezultat
    green_mask = (patch[:,:,0] == 0) & (patch[:,:,1] == 255) & (patch[:,:,2] == 0)
    white_bg[green_mask] = [0,255,0]
    cv2.imwrite(output_path, white_bg)
    print(f"[INFO] Saved patch on white background to {output_path}")
    return output_path

def create_custom_inpainting_mask(
    result_path='outputs/final_result_adjusted.png',
    mask_path='uploads/mask.png',
    output_path='outputs/new_mask_for_inpainting.png',
    border_width=8
):
    """
    Creează o mască de inpainting cu marginile măștii originale și pixelii aproape verde-neon (aproape de (0,255,0), toleranță 100) din final_result_adjusted.png transformați în alb (255) în mască.
    Extinde zonele albe cu un border de 2 pixeli în jurul lor.
    """
    # Încarcă imaginea rezultată și masca originală
    result = cv2.imread(result_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if result is None or mask is None:
        print(f"[ERROR] Could not read {result_path} or {mask_path}")
        return None
    # Redimensionează la 512x512
    result = cv2.resize(result, (512, 512))
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    # 1. Marginea măștii originale
    kernel = np.ones((border_width * 2 + 1, border_width * 2 + 1), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    border = cv2.subtract(dilated, eroded)
    # 2. Pixelii aproape verde-neon din result (aproape de (0,255,0), toleranță 100) -> alb (255)
    lower = np.array([0, 155, 0], dtype=np.uint8)  # BGR: (0, 255-100, 0)
    upper = np.array([100, 255, 100], dtype=np.uint8)  # BGR: (0+100, 255, 0+100)
    green_mask = cv2.inRange(result, lower, upper)
    # Combinăm marginile și pixelii verde-neon (toți ca alb în mască)
    final_mask = cv2.bitwise_or(border, green_mask)
    # Curățăm masca
    kernel_clean = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_clean)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)
    # Extinde zonele albe cu un border de 2 pixeli
    kernel_dilate = np.ones((5, 5), np.uint8)  # 2 pixeli în jur (5x5)
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=1)
    # Asigură-te că masca e uint8 și doar 0 sau 255
    final_mask = (final_mask > 0).astype(np.uint8) * 255
    cv2.imwrite(output_path, final_mask)
    print(f"[INFO] Saved custom inpainting mask to {output_path}")
    return output_path

def add_neon_green_contour(patch_path='outputs/gen_patch_NObg.png', contour_thickness=6):
    """
    Adaugă un contur verde neon (0,255,0) pe obiectul din patch-ul fără background (gen_patch_NObg.png).
    Suprascrie imaginea cu conturul adăugat.
    """
    if not os.path.exists(patch_path):
        print(f"[ERROR] Patch file {patch_path} does not exist.")
        return None
    patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
    if patch is None:
        print(f"[ERROR] Could not read patch from {patch_path}")
        return None
    # Creează mască binară pentru obiect (zona nenulă)
    if patch.shape[2] == 4:
        # Folosește canalul alpha
        object_mask = patch[:, :, 3] > 0
    else:
        object_mask = np.any(patch > 0, axis=2)
    object_mask = object_mask.astype(np.uint8) * 255
    # Găsește conturul
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Desenează conturul pe imaginea patch (doar pe canalele de culoare)
    patch_rgb = patch[:, :, :3].copy()
    neon_green = (0, 255, 0)  # BGR pentru OpenCV
    cv2.drawContours(patch_rgb, contours, -1, neon_green, contour_thickness)
    # Dacă există canal alpha, reconstruiește imaginea cu alpha
    if patch.shape[2] == 4:
        patch_out = np.dstack([patch_rgb, patch[:, :, 3]])
    else:
        patch_out = patch_rgb
    cv2.imwrite(patch_path, patch_out)
    print(f"[INFO] Added neon green contour (0,255,0) to {patch_path}")
    return patch_path

def replace_green_neon_with_original(
    result_path='outputs/final_result_adjusted.png',
    original_path='uploads/original.jpg',
    output_path='outputs/final_result_adjusted.png',
    tolerance=100
):
    """
    Înlocuiește toți pixelii aproape verde-neon (aproape de (0,255,0), toleranță 100) din result cu pixelul din original.
    Suprascrie sau salvează rezultatul la output_path.
    """
    result = cv2.imread(result_path)
    original = cv2.imread(original_path)
    if result is None or original is None:
        print(f"[ERROR] Could not read {result_path} or {original_path}")
        return None
    result = cv2.resize(result, (512, 512))
    original = cv2.resize(original, (512, 512))
    lower = np.array([0, 155, 0], dtype=np.uint8)
    upper = np.array([100, 255, 100], dtype=np.uint8)
    green_mask = cv2.inRange(result, lower, upper)
    green_mask_bool = green_mask > 0
    replaced = result.copy()
    replaced[green_mask_bool] = original[green_mask_bool]
    cv2.imwrite(output_path, replaced)
    print(f"[INFO] Replaced green-neon pixels with original in {output_path}")
    return output_path
