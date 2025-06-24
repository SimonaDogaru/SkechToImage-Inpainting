from fastapi import APIRouter, HTTPException
from pathlib import Path
import cv2
import numpy as np
import requests
from utils.image_utils import call_colab_generate, call_colab_inpaint
from utils.combineImages_utils import auto_combine_after_upload, color_transfer, place_patch_on_white_background, add_neon_green_contour, create_custom_inpainting_mask, replace_green_neon_with_original

router = APIRouter()

COLAB_GENERATE_URL = "https://484e-35-232-73-34.ngrok-free.app/generate"
COLAB_INPAINT_URL = "https://484e-35-232-73-34.ngrok-free.app" 
REMOVE_BG_API_KEY = "EZCUAozb27KF7FEi4gTvGB6F"

def remove_background_with_removebg(image_path, output_path):
    """
    Remove background from image using remove.bg API.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image without background
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the image file
        with open(image_path, 'rb') as image_file:
            files = {'image_file': image_file}
            headers = {'X-Api-Key': REMOVE_BG_API_KEY}
            
            # Send request to remove.bg API
            response = requests.post(
                'https://api.remove.bg/v1.0/removebg',
                files=files,
                headers=headers
            )
            
            if response.status_code == 200:
                # Save the result
                with open(output_path, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"[✅ BACKGROUND REMOVED] Saved to {output_path}")
                return True
            else:
                print(f"[❌ REMOVE.BG FAILED] Status: {response.status_code}, Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"[❌ REMOVE.BG EXCEPTION] {e}")
        return False

@router.post("/inpaint")
async def inpaint_after_upload():
    try:
        # Step 1: Generate patch from sketch
        sketch_path = Path("uploads/sketch.png")
        gen_patch_path = Path("outputs/gen_patch.png")
        gen_patch_color_transferred_path = Path("outputs/gen_patch_color_transferred.png")
        gen_patch_nobg_path = Path("outputs/gen_patch_NObg.png")
        
        if not sketch_path.exists():
            raise HTTPException(status_code=404, detail="Sketch file not found. Please upload files first.")
        
        print("[🔄 GENERATING PATCH]")
        success = call_colab_generate(
            colab_url=COLAB_GENERATE_URL,
            sketch_path=str(sketch_path),
            output_path=str(gen_patch_path)
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate patch with Colab")
        
        # Step 2: Apply color transfer from original to generated patch
        print("[🔄 APPLYING COLOR TRANSFER TO GENERATED PATCH]")
        original = cv2.imread('uploads/original.jpg')
        gen_patch = cv2.imread(str(gen_patch_path))
        
        if original is None or gen_patch is None:
            raise HTTPException(status_code=500, detail="Could not load original image or generated patch")
        
        # Resize both to same size for color transfer
        original_resized = cv2.resize(original, (512, 512))
        gen_patch_resized = cv2.resize(gen_patch, (512, 512))
        
        # Apply color transfer
        color_transferred_patch = color_transfer(original_resized, gen_patch_resized)
        
        if color_transferred_patch is None:
            print("[⚠️ COLOR TRANSFER FAILED] Using original patch")
            color_transferred_patch = gen_patch_resized
        
        # Save the color transferred patch
        cv2.imwrite(str(gen_patch_color_transferred_path), color_transferred_patch)
        print("[✅ COLOR TRANSFER COMPLETED]")
        
        # Step 3: Remove background from color transferred patch
        print("[🔄 REMOVING BACKGROUND]")
        bg_removal_success = remove_background_with_removebg(
            image_path=str(gen_patch_color_transferred_path),
            output_path=str(gen_patch_nobg_path)
        )
        
        if not bg_removal_success:
            print("[⚠️ BACKGROUND REMOVAL FAILED] Using color transferred patch")
            # If background removal fails, use the color transferred patch
            gen_patch_nobg_path = gen_patch_color_transferred_path
        else:
            print("[✅ BACKGROUND REMOVED SUCCESSFULLY]")

        # Step 3.1: Adaugă contur verde neon pe patch-ul fără background (obligatoriu înainte de fundal alb)
        print("[🔄 ADDING NEON GREEN CONTOUR TO PATCH]")
        add_neon_green_contour(patch_path=str(gen_patch_nobg_path), contour_thickness=6)

        # Step 3.2: Place patch on white background (folosește patch-ul cu contur deja adăugat)
        print("[🔄 PLACING PATCH ON WHITE BACKGROUND]")
        gen_patch_nobg_onwhite_path = 'outputs/gen_patch_NObg_onWhite.png'
        place_patch_on_white_background(patch_path=str(gen_patch_nobg_path), output_path=gen_patch_nobg_onwhite_path)

        # Step 4: Combine images (using the patch on white background)
        print("[🔄 COMBINING IMAGES]")
        result = auto_combine_after_upload(gen_patch_path=gen_patch_nobg_onwhite_path)
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to combine images")
        final_result_path, _ = result

        # Step 4.1: Creează automat noua mască pentru inpainting
        print("[🔄 GENERATING CUSTOM INPAINTING MASK]")
        create_custom_inpainting_mask(
            result_path='outputs/final_result_adjusted.png',
            mask_path='uploads/mask.png',
            output_path='outputs/new_mask_for_inpainting.png',
            border_width=8
        )
        inpainting_mask_path = 'outputs/new_mask_for_inpainting.png'

        # Înlocuiește pixelii verde-neon cu cei din original
        print("[🔄 REPLACING GREEN-NEON PIXELS WITH ORIGINAL]")
        replace_green_neon_with_original(
            result_path='outputs/final_result_adjusted.png',
            original_path='uploads/original.jpg',
            output_path='outputs/final_result_adjusted.png',
            tolerance=100
        )

        # Step 5: Perform inpainting
        print("[🔄 PERFORMING INPAINTING]")
        image_path = Path("outputs/final_result_adjusted.png")
        mask_path = Path(inpainting_mask_path)
        output_path = Path("outputs/inpainted_result.jpg")

        if not image_path.exists() or not mask_path.exists():
            raise HTTPException(status_code=404, detail="Required files not found for inpainting.")

        success = call_colab_inpaint(
            colab_url=COLAB_INPAINT_URL,
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path
        )

        if not success:
            raise HTTPException(status_code=500, detail="Inpainting failed.")

        return {
            "message": "Complete processing pipeline completed successfully",
            "generated_patch": str(gen_patch_path),
            "color_transferred_patch": str(gen_patch_color_transferred_path),
            "generated_patch_no_bg": str(gen_patch_nobg_path),
            "final_result": str(final_result_path),
            "inpainting_mask": str(inpainting_mask_path),
            "inpainted_image": str(output_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[❌ INPAINTING EXCEPTION] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during processing pipeline")
