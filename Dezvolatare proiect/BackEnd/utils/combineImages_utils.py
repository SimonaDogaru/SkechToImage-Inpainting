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

def combine_images():
    if not wait_for_gen_patch():
        raise TimeoutError("Timeout waiting for gen_patch.png to be generated")

    original = cv2.imread('uploads/original.jpg')
    mask = cv2.imread('uploads/mask.png', cv2.IMREAD_GRAYSCALE)
    gen_patch = cv2.imread('outputs/gen_patch.png')

    if original is None or mask is None or gen_patch is None:
        raise FileNotFoundError("One or more input images could not be loaded")

    # Analyze original mask first
    print("[DEBUG] Analyzing original mask...")
    mask_analysis = analyze_original_mask('uploads/mask.png')
    print(f"[DEBUG] Mask analysis: {mask_analysis}")

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
    
    # Debug: Check pixel values in the blended region
    print(f"[DEBUG] Blended region min/max values: {np.min(blended_region)}/{np.max(blended_region)}")
    print(f"[DEBUG] Blended region mean values: {np.mean(blended_region, axis=(0,1))}")
    
    # Check if there are any very light pixels in the blended region
    light_pixels = np.sum(blended_region > 200, axis=2) >= 2  # At least 2 channels > 200
    print(f"[DEBUG] Light pixels in blended region: {np.sum(light_pixels)}")

    # STEP: Replace white/very light pixels with original image pixels
    print("[🔄 REPLACING LIGHT PIXELS WITH ORIGINAL]")
    
    # Create a mask for very light pixels in the blended region
    # Detect pixels where all channels are above a threshold (very light/white pixels)
    light_threshold = 180
    light_pixel_mask = np.all(blended_region > light_threshold, axis=2)
    
    # Create a mask for transparent pixels in the blended region
    transparent_pixel_mask = np.all(blended_region == 0, axis=2)
    
    # Combine both masks
    replacement_mask = np.logical_or(light_pixel_mask, transparent_pixel_mask)
    
    print(f"[DEBUG] Light pixels detected (threshold {light_threshold}): {np.sum(light_pixel_mask)}")
    print(f"[DEBUG] Transparent pixels detected: {np.sum(transparent_pixel_mask)}")
    print(f"[DEBUG] Total pixels to replace: {np.sum(replacement_mask)}")
    
    # Replace light and transparent pixels with original image pixels
    if np.sum(replacement_mask) > 0:
        # Extract the region from original image
        original_region = original[ymin:ymax, xmin:xmax]
        
        # Create a copy of the blended region
        corrected_region = blended_region.copy()
        
        # Replace light and transparent pixels with original pixels
        corrected_region[replacement_mask] = original_region[replacement_mask]
        
        # Update the result with corrected region
        result[ymin:ymax, xmin:xmax] = corrected_region
        
        print(f"[DEBUG] Replaced {np.sum(replacement_mask)} pixels with original pixels")
    else:
        print("[DEBUG] No pixels to replace detected")

    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite('outputs/final_result.png', result)

    # Apply global light pixel replacement to the entire result image
    print("[🔄 APPLYING GLOBAL LIGHT PIXEL REPLACEMENT]")
    result = replace_light_pixels_with_original_adjustable(result, original, light_threshold=180)
    
    # Save the corrected final result
    cv2.imwrite('outputs/final_result_corrected.png', result)

    # Use the border mask from uploaded mask for inpainting
    # Create a mask where only the borders of the uploaded mask are white
    inpainting_mask_path = create_border_mask_from_uploaded_mask(border_width=3)
    if inpainting_mask_path is None:
        # Fallback to generated mask if border mask creation fails
        inpainting_mask_path = generate_inpainting_mask_rgb_adjustable(result)
        print(f"[DEBUG] Border mask creation failed, using generated mask")
    else:
        print(f"[DEBUG] Successfully created border mask from uploaded mask for inpainting")

    return 'outputs/final_result_corrected.png', inpainting_mask_path

def auto_combine_after_upload():
    try:
        return combine_images()
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

def replace_light_pixels_with_original_adjustable(result_image, original_image, light_threshold=180, output_debug=True):
    """
    Replace white/very light pixels and transparent pixels in the result image with pixels from the original image.
    This helps create a cleaner image for inpainting by removing artificial light areas and transparent pixels.
    
    Args:
        result_image: The image to process
        original_image: The original image to get replacement pixels from
        light_threshold: Threshold for detecting light pixels (default: 180)
        output_debug: Whether to save debug images (default: True)
    """
    # Create a mask for very light pixels
    light_pixel_mask = np.all(result_image > light_threshold, axis=2)
    
    # Create a mask for transparent pixels (pixels with value 0 in all channels)
    transparent_pixel_mask = np.all(result_image == 0, axis=2)
    
    # Combine both masks
    replacement_mask = np.logical_or(light_pixel_mask, transparent_pixel_mask)
    
    print(f"[DEBUG] Light pixels detected (threshold {light_threshold}): {np.sum(light_pixel_mask)}")
    print(f"[DEBUG] Transparent pixels detected: {np.sum(transparent_pixel_mask)}")
    print(f"[DEBUG] Total pixels to replace: {np.sum(replacement_mask)}")
    
    # Create a copy of the result image
    corrected_image = result_image.copy()
    
    # Replace light and transparent pixels with original pixels
    if np.sum(replacement_mask) > 0:
        corrected_image[replacement_mask] = original_image[replacement_mask]
        print(f"[DEBUG] Replaced {np.sum(replacement_mask)} pixels with original pixels")
    else:
        print("[DEBUG] No pixels to replace detected")
    
    return corrected_image

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

def create_border_mask_from_uploaded_mask(uploaded_mask_path='uploads/mask.png', border_width=3, output_path='outputs/new_mask_for_inpainting.png'):
    """
    Create a mask where only the borders of the uploaded mask are white.
    This allows inpainting only on the edges of the original mask area.
    
    Args:
        uploaded_mask_path: Path to the uploaded mask
        border_width: Width of the border to extract (default: 3)
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
