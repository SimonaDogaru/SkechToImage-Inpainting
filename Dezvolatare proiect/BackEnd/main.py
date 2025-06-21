from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from pathlib import Path
import uvicorn

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Image Editing API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from endpoints import upload, inpaint, image_processing

# Include routers
#app.include_router(image_processing.router, prefix="/api/v1", tags=["image-processing"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(inpaint.router, prefix="/api/v1", tags=["inpaint"])

@app.get("/")
async def root():
    return {"message": "Image Editing API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 