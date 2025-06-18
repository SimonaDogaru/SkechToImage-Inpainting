from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import List
import shutil
from pathlib import Path
from utils.image_utils import call_colab_generate
from utils.combineImages_utils import auto_combine_after_upload
import traceback

router = APIRouter()

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
COLAB_URL = "https://cfaa-34-124-184-64.ngrok-free.app/generate"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {
    "original": [".jpg", ".jpeg"],
    "mask": [".png"],
    "sketch": [".png"]
}

def validate_file_extension(filename: str, file_type: str) -> bool:
    """Validate if the file has the correct extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS[file_type]

@router.post("/upload")
async def upload_files(
    original: UploadFile = File(...),
    mask: UploadFile = File(...),
    sketch: UploadFile = File(...)
):
    try:
        # Validate file extensions
        if not validate_file_extension(original.filename, "original"):
            raise HTTPException(status_code=400, detail="Original file must be a JPG")
        if not validate_file_extension(mask.filename, "mask"):
            raise HTTPException(status_code=400, detail="Mask file must be a PNG")
        if not validate_file_extension(sketch.filename, "sketch"):
            raise HTTPException(status_code=400, detail="Sketch file must be a PNG")

        # Define file paths
        file_paths = {
            "original": UPLOAD_DIR / "original.jpg",
            "mask": UPLOAD_DIR / "mask.png",
            "sketch": UPLOAD_DIR / "sketch.png"
        }

        # Save files
        for file, path in zip([original, mask, sketch], file_paths.values()):
            with open(path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Call Colab to generate patch
        output_path = OUTPUT_DIR / "gen_patch.png"
        success = call_colab_generate(
            colab_url=COLAB_URL,
            sketch_path=file_paths["sketch"],
            output_path=output_path
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate patch with Colab")

        # Automatically combine images after successful patch generation
        final_result = auto_combine_after_upload()
        if final_result is None:
            raise HTTPException(status_code=500, detail="Failed to combine images")

        return JSONResponse(
            content={
                "message": "Files uploaded, patch generated and images combined successfully",
                "uploaded_files": [str(path) for path in file_paths.values()],
                "generated_patch": str(output_path),
                "final_result": str(final_result)
            },
            status_code=200
        )

    except Exception as e:
        # Clean up any partially saved files in case of error
        for path in file_paths.values():
            try:
                if path.exists():
                    path.unlink()
            except:
                pass
        #raise HTTPException(status_code=500, detail=str(e))
        print("[❌ EXCEPTION]", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error – see terminal logs")

