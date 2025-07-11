from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from typing import List
import shutil
from pathlib import Path
import traceback

router = APIRouter()

# Ensure directories exist
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

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

        return JSONResponse(
            content={
                "message": "Files uploaded successfully",
                "uploaded_files": [str(path) for path in file_paths.values()]
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
        print("[❌ EXCEPTION]", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal Server Error – see terminal logs")

