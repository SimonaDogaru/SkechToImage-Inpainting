import React, { useRef, useState, useEffect } from "react";
import { Stage, Layer, Image as KonvaImage, Line } from "react-konva";

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 512;
type Tool = "none" | "mask" | "sketch";

interface CanvasEditorProps {
  stageRef: React.RefObject<any>;
  imageLayerRef: React.RefObject<any>;
  maskLayerRef: React.RefObject<any>;
  sketchLayerRef: React.RefObject<any>;
  tool: Tool;
  setUndoHandler?: (fn: () => void) => void;
  setClearHandler?: (fn: () => void) => void;
  image?: HTMLImageElement | null;
  sketchImage?: HTMLImageElement | null;
  setClearSketchLinesHandler?: (fn: () => void) => void;
}

const CanvasEditor: React.FC<CanvasEditorProps> = ({ stageRef, imageLayerRef, maskLayerRef, sketchLayerRef, tool, setUndoHandler, setClearHandler, image, sketchImage, setClearSketchLinesHandler }) => {
  const [isDrawing, setIsDrawing] = useState(false);
  const [maskLines, setMaskLines] = useState<{ points: number[] }[]>([]);
  const [sketchLines, setSketchLines] = useState<{ points: number[] }[]>([]);

  // Undo function
  const handleUndo = () => {
    if (tool === "mask" && maskLines.length > 0) {
      setMaskLines((lines) => lines.slice(0, -1));
    } else if (tool === "sketch" && sketchLines.length > 0) {
      setSketchLines((lines) => lines.slice(0, -1));
    }
  };

  // Clear function
  const handleClear = () => {
    setMaskLines([]);
    setSketchLines([]);
  };

  useEffect(() => {
    if (setUndoHandler) setUndoHandler(handleUndo);
    if (setClearHandler) setClearHandler(handleClear);
    if (setClearSketchLinesHandler) setClearSketchLinesHandler(() => () => setSketchLines([]));
  }, [tool, maskLines, sketchLines]);

  // Mouse events pentru desenare mască/sketch
  const handleMouseDown = (e: any) => {
    if (tool === "mask") {
      setIsDrawing(true);
      const pos = e.target.getStage().getPointerPosition();
      setMaskLines([...maskLines, { points: [pos.x, pos.y] }]);
    } else if (tool === "sketch") {
      setIsDrawing(true);
      const pos = e.target.getStage().getPointerPosition();
      setSketchLines([...sketchLines, { points: [pos.x, pos.y] }]);
    }
  };

  const handleMouseMove = (e: any) => {
    if (!isDrawing) return;
    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    if (tool === "mask") {
      setMaskLines((prevLines) => {
        const lastLine = prevLines[prevLines.length - 1];
        const newLines = prevLines.slice(0, -1);
        return [
          ...newLines,
          { points: [...lastLine.points, point.x, point.y] },
        ];
      });
    } else if (tool === "sketch") {
      setSketchLines((prevLines) => {
        const lastLine = prevLines[prevLines.length - 1];
        const newLines = prevLines.slice(0, -1);
        return [
          ...newLines,
          { points: [...lastLine.points, point.x, point.y] },
        ];
      });
    }
  };

  const handleMouseUp = () => {
    setIsDrawing(false);
  };

  return (
    <div className="canvas-editor flex flex-col items-center gap-6 w-full">
      <div className="image-canvas bg-white rounded-lg overflow-hidden shadow-md w-[512px] h-[512px] flex items-center justify-center border border-[#DDDDDD] relative">
        <Stage
          width={CANVAS_WIDTH}
          height={CANVAS_HEIGHT}
          onMouseDown={handleMouseDown}
          onMousemove={handleMouseMove}
          onMouseup={handleMouseUp}
          ref={stageRef}
        >
          {/* Layer 1: Imagine */}
          <Layer ref={imageLayerRef}>
            {image && <KonvaImage image={image} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />}
          </Layer>
          {/* Layer 2: Mask (desen alb) */}
          <Layer ref={maskLayerRef}>
            {maskLines.map((line, idx) => (
              <Line
                key={"mask-" + idx}
                points={line.points}
                stroke="#fff"
                strokeWidth={16}
                lineCap="round"
                lineJoin="round"
                globalCompositeOperation="source-over"
              />
            ))}
          </Layer>
          {/* Layer 3: Sketch (desen negru) */}
          <Layer ref={sketchLayerRef}>
            {sketchImage && <KonvaImage image={sketchImage} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} />}
            {sketchLines.map((line, idx) => (
              <Line
                key={"sketch-" + idx}
                points={line.points}
                stroke="#000"
                strokeWidth={3}
                lineCap="round"
                lineJoin="round"
                globalCompositeOperation="source-over"
              />
            ))}
          </Layer>
        </Stage>
        {!image && <span className="text-gray-400 absolute">Image Canvas</span>}
      </div>
    </div>
  );
};

export default CanvasEditor; 