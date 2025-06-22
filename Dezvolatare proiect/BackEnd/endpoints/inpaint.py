from fastapi import APIRouter, HTTPException
from pathlib import Path
from utils.image_utils import call_colab_generate, call_colab_inpaint
from utils.combineImages_utils import auto_combine_after_upload

router = APIRouter()

COLAB_GENERATE_URL = "https://14cc-34-87-187-95.ngrok-free.app/generate"
COLAB_INPAINT_URL = "https://14cc-34-87-187-95.ngrok-free.app"

@router.post("/inpaint")
async def inpaint_after_upload():
    try:
        # Step 1: Generate patch using Colab
        sketch_path = Path("uploads/sketch.png")
        gen_patch_path = Path("outputs/gen_patch.png")
        
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
        
        # Step 2: Combine images
        print("[🔄 COMBINING IMAGES]")
        result = auto_combine_after_upload()
        if result is None:
            raise HTTPException(status_code=500, detail="Failed to combine images")
        
        final_result_path, inpainting_mask_path = result
        
        # Step 3: Perform inpainting
        print("[🔄 PERFORMING INPAINTING]")
        image_path = Path("outputs/final_result_corrected.png")
        mask_path = Path("outputs/new_mask_for_inpainting.png")
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
            "final_result": str(final_result_path),
            "inpainting_mask": str(inpainting_mask_path),
            "inpainted_image": str(output_path)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[❌ INPAINTING EXCEPTION] {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during processing pipeline")
