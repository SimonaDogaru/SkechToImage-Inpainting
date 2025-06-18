# Image Editing API

A FastAPI backend for image editing that integrates with ControlNet and LaMa models running in Google Colab.

## Setup

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Colab endpoints:
```env
COLAB_CONTROLNET_URL=http://your-colab-url/generate
COLAB_LAMA_URL=http://your-colab-url/inpaint
```

## Running the API

Start the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /api/v1/process-image

Process an image with mask and sketch inputs.

**Request:**
- `image`: Original image file (PNG/JPG)
- `mask`: Mask image file (PNG)
- `sketch`: Sketch image file (PNG/JPG)
- `do_inpainting`: Boolean flag for optional LaMa inpainting (default: false)

**Response:**
- Returns the processed image file (JPG)

## Directory Structure

```
.
├── main.py              # Main FastAPI application
├── requirements.txt     # Python dependencies
├── uploads/            # Temporary storage for uploaded files
├── outputs/            # Storage for processed images
├── endpoints/          # API route handlers
│   └── image_processing.py
└── utils/             # Utility functions
    └── image_utils.py
```

## Notes

- The API creates unique session directories for each request to prevent file conflicts
- All temporary files are automatically cleaned up after processing
- The API uses OpenCV's seamless cloning for blending the generated patch
- Make sure your Colab endpoints are properly configured and running before using the API 