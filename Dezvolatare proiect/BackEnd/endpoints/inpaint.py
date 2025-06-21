from fastapi import APIRouter, HTTPException
from pathlib import Path
from utils.image_utils import call_colab_inpaint

router = APIRouter()

COLAB_URL = "https://ed4d-34-126-86-58.ngrok-free.app"  # înlocuiește cu linkul tău actual

@router.post("/inpaint")
async def inpaint_after_upload():
    image_path = Path("outputs/final_result.png")
    mask_path = Path("outputs/new_mask_for_inpainting.png")
    output_path = Path("outputs/inpainted_result.jpg")

    if not image_path.exists() or not mask_path.exists():
        raise HTTPException(status_code=404, detail="Required files not found.")

    success = call_colab_inpaint(
        colab_url=COLAB_URL,
        image_path=image_path,
        mask_path=mask_path,
        output_path=output_path
    )

    if not success:
        raise HTTPException(status_code=500, detail="Inpainting failed.")

    return {
        "message": "Inpainting completed successfully.",
        "inpainted_image": str(output_path)
    }
