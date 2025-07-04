import cv2
import numpy as np
import os
import time
from pathlib import Path
import shutil
#final version of the wait for gen patch
def wait_for_gen_patch(timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists('outputs/gen_patch.png'):
            return True
        time.sleep(1)
    return False


#final version of the combine images
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

    # find bounding box for optimization
    white_pixels = np.where(mask == 255)
    if len(white_pixels[0]) == 0:
        raise ValueError("No white pixels found in the mask")

    ymin, ymax = np.min(white_pixels[0]), np.max(white_pixels[0])
    xmin, xmax = np.min(white_pixels[1]), np.max(white_pixels[1])

    print(f"[DEBUG] bbox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    patch_width = xmax - xmin
    patch_height = ymax - ymin

    # resize the generated patch to the bounding box size
    resized_patch = cv2.resize(gen_patch, (patch_width, patch_height))
    
    # create a mask for the blending area (only the white area from the original mask)
    blend_mask = mask[ymin:ymax, xmin:xmax]
    
    # create a mask with gradient for smoother transitions
    # apply blur to create smooth transitions
    smooth_mask = cv2.GaussianBlur(blend_mask, (15, 15), 0)
    
    # normalize the mask to 0-1 for blending
    blend_mask_normalized = smooth_mask.astype(np.float32) / 255.0
    
    # extend the mask to 3 channels for blending
    blend_mask_3ch = np.stack([blend_mask_normalized] * 3, axis=2)
    
    # create a copy of the original image
    result = original.copy()
    
    # extract the region from the original image
    original_region = original[ymin:ymax, xmin:xmax]
    
    # apply blending with natural transitions (the patch already has the colors transferred)
    # use a combination of linear blending and seamless cloning
    blended_region = (resized_patch * blend_mask_3ch + 
                     original_region * (1 - blend_mask_3ch)).astype(np.uint8)
    
    # update the result with the blended region
    result[ymin:ymax, xmin:xmax] = blended_region

    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite('outputs/final_result.png', result)

    # replace all white pixels with original pixels and save as final_result_adjusted.png
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
#final version of the auto combine after upload
def auto_combine_after_upload(gen_patch_path='outputs/gen_patch.png'):
    try:
        return combine_images(gen_patch_path=gen_patch_path)
    except Exception as e:
        print(f"Error during image combination: {str(e)}")
        return None
#final version of the analyze original mask
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
#final version of the color transfer
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
#final version of the place patch on white background
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
    # create a white background
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    # if the patch has an alpha channel, use it for composition
    if patch.shape[2] == 4:
        alpha = patch[:, :, 3] / 255.0
        for c in range(3):
            white_bg[:, :, c] = (patch[:, :, c] * alpha + white_bg[:, :, c] * (1 - alpha)).astype(np.uint8)
    else:
        # for RGB patch, overwrite only non-zero pixels
        mask = np.any(patch > 0, axis=2)
        white_bg[mask] = patch[mask]
    # ensure that all green-neon pixels in the patch remain green-neon in the result
    green_mask = (patch[:,:,0] == 0) & (patch[:,:,1] == 255) & (patch[:,:,2] == 0)
    white_bg[green_mask] = [0,255,0]
    cv2.imwrite(output_path, white_bg)
    print(f"[INFO] Saved patch on white background to {output_path}")
    return output_path
#final version of the custom inpainting mask
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
    # load the result image and the original mask
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
#final version of the add neon green contour
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
    # create binary mask for the object (non-zero area)
    if patch.shape[2] == 4:
        # use the alpha channel
        object_mask = patch[:, :, 3] > 0
    else:
        object_mask = np.any(patch > 0, axis=2)
    object_mask = object_mask.astype(np.uint8) * 255
    # find the contour
    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw the contour on the patch (only on the color channels)
    patch_rgb = patch[:, :, :3].copy()
    neon_green = (0, 255, 0)  # BGR for OpenCV
    cv2.drawContours(patch_rgb, contours, -1, neon_green, contour_thickness)
    # if there is an alpha channel, reconstruct the image with alpha
    if patch.shape[2] == 4:
        patch_out = np.dstack([patch_rgb, patch[:, :, 3]])
    else:
        patch_out = patch_rgb
    cv2.imwrite(patch_path, patch_out)
    print(f"[INFO] Added neon green contour (0,255,0) to {patch_path}")
    return patch_path
#final version of the replace green-neon with original
def replace_green_neon_with_original(
    result_path='outputs/final_result_adjusted.png',
    original_path='uploads/original.jpg',
    output_path='outputs/final_result_adjusted.png',
    tolerance=100
):
    """
    Replace all green-neon pixels (close to (0,255,0), tolerance 100) in result with the pixel from original.
    Overwrite or save the result at output_path.
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
