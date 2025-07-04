from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
import uvicorn
import numpy as np

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Image Editing API")

# Mount the outputs directory as static
app.mount("/output", StaticFiles(directory="outputs"), name="output")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from endpoints import upload, inpaint

# Include routers
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(inpaint.router, prefix="/api/v1", tags=["inpaint"])

@app.get("/")
async def root():
    return {"message": "Image Editing API is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 