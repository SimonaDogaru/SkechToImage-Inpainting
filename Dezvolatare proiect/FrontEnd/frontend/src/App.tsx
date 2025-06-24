import React, { useRef, useState } from "react";
import { Image as KonvaImage } from "react-konva";
import Toolbar from "./components/Toolbar";
import Controls from "./components/Controls";
import CanvasEditor from "./components/CanvasEditor";
import ImagePreview from "./components/ImagePreview";
import { uploadImages, generateInpaint } from "./utils/api";

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 512;
type Tool = "none" | "mask" | "sketch";

const App = () => {
  // Tool state
  const [tool, setTool] = useState<Tool>("none");
  // Referințe la layerele din CanvasEditor
  const stageRef = useRef<any>(null);
  const maskLayerRef = useRef<any>(null);
  const sketchLayerRef = useRef<any>(null);
  const imageLayerRef = useRef<any>(null);
  // Preview image state
  const [previewUrl, setPreviewUrl] = useState<string>("");
  // Undo/Clear handlers
  const canvasEditorUndoRef = useRef<() => void>(() => {});
  const canvasEditorClearRef = useRef<() => void>(() => {});
  // Clear only sketch lines handler
  const clearSketchLinesRef = useRef<() => void>(() => {});
  // File input ref for upload button
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Image state for canvas
  const [canvasImage, setCanvasImage] = useState<HTMLImageElement | null>(null);
  // Image state for sketch layer
  const [sketchImage, setSketchImage] = useState<HTMLImageElement | null>(null);
  const [sketchUploaded, setSketchUploaded] = useState(false);
  // File input ref for sketch upload
  const sketchFileInputRef = useRef<HTMLInputElement>(null);

  // Funcție pentru exportul imaginilor
  const handleUpload = async () => {
    if (!stageRef.current || !imageLayerRef.current || !maskLayerRef.current || !sketchLayerRef.current) return;
    // 1. Export original
    const originalDataUrl = imageLayerRef.current.toDataURL({ mimeType: "image/jpeg", quality: 1 });
    // 2. Export mask (alb pe negru)
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = CANVAS_WIDTH;
    maskCanvas.height = CANVAS_HEIGHT;
    const ctx = maskCanvas.getContext("2d");
    let maskDataUrl = "";
    if (ctx) {
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      const maskImg = new window.Image();
      maskImg.src = maskLayerRef.current.toDataURL({ mimeType: "image/png" });
      await new Promise((resolve) => {
        maskImg.onload = () => {
          ctx.drawImage(maskImg, 0, 0);
          maskDataUrl = maskCanvas.toDataURL("image/png");
          resolve(void 0);
        };
      });
    }
    // 3. Export sketch (negru pe alb) sau folosește sketchImage dacă există
    let sketchDataUrl = "";
    let sketchFile;
    if (sketchImage) {
      // Use uploaded sketch image
      // Convert to PNG file
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = CANVAS_WIDTH;
      tempCanvas.height = CANVAS_HEIGHT;
      const tempCtx = tempCanvas.getContext("2d");
      if (tempCtx) {
        tempCtx.fillStyle = "#fff";
        tempCtx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        tempCtx.drawImage(sketchImage, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        sketchDataUrl = tempCanvas.toDataURL("image/png");
      }
    } else {
      // Export drawn sketch
      const sketchCanvas = document.createElement("canvas");
      sketchCanvas.width = CANVAS_WIDTH;
      sketchCanvas.height = CANVAS_HEIGHT;
      const ctx2 = sketchCanvas.getContext("2d");
      if (ctx2) {
        ctx2.fillStyle = "#fff";
        ctx2.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        const sketchImg = new window.Image();
        sketchImg.src = sketchLayerRef.current.toDataURL({ mimeType: "image/png" });
        await new Promise((resolve) => {
          sketchImg.onload = () => {
            ctx2.drawImage(sketchImg, 0, 0);
            sketchDataUrl = sketchCanvas.toDataURL("image/png");
            resolve(void 0);
          };
        });
      }
    }
    // Convert dataURL to File
    function dataURLtoFile(dataurl, filename) {
      const arr = dataurl.split(",");
      const mime = arr[0].match(/:(.*?);/)[1];
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8arr = new Uint8Array(n);
      while (n--) {
        u8arr[n] = bstr.charCodeAt(n);
      }
      return new File([u8arr], filename, { type: mime });
    }
    const originalFile = dataURLtoFile(originalDataUrl, "original_photo.jpg");
    const maskFile = dataURLtoFile(maskDataUrl, "mask_photo.png");
    sketchFile = dataURLtoFile(sketchDataUrl, "sketch_photo.png");
    // Trimite la backend
    try {
      await uploadImages({ original: originalFile, mask: maskFile, sketch: sketchFile });
      alert("Upload successful!");
    } catch (e) {
      alert("Upload failed!");
    }
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (sketchFileInputRef.current) sketchFileInputRef.current.value = '';
  };

  // Funcție pentru generare imagine inpaint
  const handleGenerate = async () => {
    try {
      await generateInpaint();
      setPreviewUrl("http://127.0.0.1:8000/output/inpainted_result.jpg?ts=" + Date.now());
    } catch (e) {
      alert("Generate failed!");
    }
  };

  // Buton de download pentru preview
  const handleDownloadPreview = async () => {
    if (!previewUrl) return;
    try {
      const response = await fetch(previewUrl, { mode: 'cors' });
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'inpaint_result.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      alert('Download failed!');
    }
  };

  // Undo handler
  const handleUndo = () => {
    if (canvasEditorUndoRef.current) canvasEditorUndoRef.current();
  };
  // Clear handler
  const handleClear = () => {
    if (canvasEditorClearRef.current) canvasEditorClearRef.current();
  };

  return (
    <div className="min-h-screen bg-white flex flex-col items-center py-8">
      <div className="flex items-center gap-2 mb-8">
        <span className="bg-green-500 text-white rounded-lg p-2 text-2xl">✅</span>
        <h1 className="text-4xl font-bold">Smart Image Editor</h1>
      </div>
      
      {/* Main content area - centered */}
      <div className="flex flex-row items-center justify-center gap-8 w-full max-w-7xl">
        {/* Toolbar */}
        <Toolbar tool={tool} onSelectTool={setTool} onUndo={handleUndo} onClear={handleClear} />
        
        {/* Canvas area with upload button */}
        <div className="flex flex-col items-center relative">
          {/* Upload button - round with + sign, centered above canvas */}
          <div className="flex flex-row items-center justify-start mb-4 w-full gap-3">
            {/* Upload photo button and input */}
            <div className="relative flex items-center">
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    const img = new window.Image();
                    img.src = reader.result as string;
                    img.onload = () => {
                      setCanvasImage(img);
                      setSketchImage(null);
                      setSketchUploaded(false);
                      // Clear mask and sketch lines
                      if (canvasEditorClearRef.current) canvasEditorClearRef.current();
                      if (clearSketchLinesRef.current) clearSketchLinesRef.current();
                    };
                  };
                  reader.readAsDataURL(file);
                }}
              />
              <button
                className="w-12 h-12 bg-[#4CAF50] hover:bg-[#388E3C] text-white rounded-full flex items-center justify-center text-2xl font-bold shadow-lg border-2 border-white transition-colors"
                onClick={() => fileInputRef.current?.click()}
                title="Upload Image"
              >
                +
              </button>
            </div>
            {/* Upload sketch button and input */}
            <div className="relative flex items-center ml-3">
              <input
                type="file"
                accept="image/*"
                ref={sketchFileInputRef}
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  const reader = new FileReader();
                  reader.onload = () => {
                    const img = new window.Image();
                    img.src = reader.result as string;
                    img.onload = () => {
                      setSketchImage(img);
                      setSketchUploaded(true);
                      // Clear sketch lines
                      if (clearSketchLinesRef.current) clearSketchLinesRef.current();
                    };
                  };
                  reader.readAsDataURL(file);
                }}
              />
              <button
                className="flex flex-row items-center gap-2 px-4 h-12 bg-[#388E3C] hover:bg-[#4CAF50] text-white rounded-full font-semibold shadow-lg border-2 border-white transition-colors text-base"
                onClick={() => sketchFileInputRef.current?.click()}
                title="Upload Sketch"
              >
                <span className="text-2xl">✏️</span>
                Upload your sketch
              </button>
            </div>
            {sketchUploaded && <span className="text-green-600 font-semibold">sketch uploaded</span>}
          </div>
          
          {/* Canvas Editor */}
          <CanvasEditor
            stageRef={stageRef}
            imageLayerRef={imageLayerRef}
            maskLayerRef={maskLayerRef}
            sketchLayerRef={sketchLayerRef}
            tool={tool}
            setUndoHandler={fn => (canvasEditorUndoRef.current = fn)}
            setClearHandler={fn => (canvasEditorClearRef.current = fn)}
            image={canvasImage}
            setClearSketchLinesHandler={fn => (clearSketchLinesRef.current = fn)}
          />
          
          {/* Controls */}
          <div className="w-full mt-6">
            <Controls onUpload={handleUpload} onGenerate={handleGenerate} />
          </div>
        </div>
        {/* Spacer 10% of page width */}
        <div className="w-[10vw] min-w-[40px] max-w-[100px]" />
        {/* Preview */}
        <div className="flex flex-col items-center">
          <ImagePreview imageUrl={previewUrl} />
          <button
            className="mt-4 px-6 py-2 bg-[#4CAF50] hover:bg-[#388E3C] text-white rounded-full font-semibold border-2 border-white w-full transition-all duration-150 shadow-sm"
            onClick={handleDownloadPreview}
            disabled={!previewUrl}
          >
            <span className="text-white">Download</span>
        </button>
        </div>
      </div>
    </div>
  );
};

export default App;
