import React from "react";

type Tool = "none" | "mask" | "sketch";

interface ToolbarProps {
  tool: Tool;
  onSelectTool: (tool: Tool) => void;
  onUndo: () => void;
  onClear: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({ tool, onSelectTool, onUndo, onClear }) => {
  const btnBase =
    "w-full bg-white rounded-lg flex flex-col items-center justify-center transition-all duration-150 py-2 hover:bg-[#bbf7d0] border border-[#e5e7eb] focus:outline-none";

  return (
    <div className="toolbar flex flex-col items-center py-4 px-2 bg-white rounded-xl shadow-md gap-2 border border-[#DDDDDD] w-20">
      {/* Mask tool */}
      <button
        className={btnBase}
        title="Mask"
        onClick={() => onSelectTool(tool === "mask" ? "none" : "mask")}
      >
        <span>
          <svg width="28" height="28" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="9" fill="none" stroke={tool === "mask" ? '#22c55e' : 'black'} strokeWidth="3" />
            {tool === "mask" && <circle cx="12" cy="12" r="7" fill="#22c55e" />}
          </svg>
        </span>
        <span className="text-xs mt-1 text-black">Mask</span>
      </button>
      {/* Sketch tool */}
      <button
        className={btnBase}
        title="Sketch"
        onClick={() => onSelectTool(tool === "sketch" ? "none" : "sketch")}
      >
        <span>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={tool === "sketch" ? "#22c55e" : "black"} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M16.5 3.5L20.5 7.5" />
            <path d="M2 21L16.5 3.5" />
            <path d="M2 21H7L16.5 3.5" />
          </svg>
        </span>
        <span className="text-xs mt-1 text-black">Sketch</span>
      </button>
      {/* Undo */}
      <button className={btnBase} title="Undo" onClick={onUndo}>
        <span>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 19C4.58 19 1 15.42 1 11C1 6.58 4.58 3 9 3C13.42 3 17 6.58 17 11C17 13.21 15.84 15.21 14 16.32" />
            <polyline points="9 15 5 11 9 7" />
          </svg>
        </span>
        <span className="text-xs mt-1 text-black">Undo</span>
      </button>
      {/* Clear (trash) */}
      <button className={btnBase} title="Clear" onClick={onClear}>
        <span>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="6" width="18" height="14" rx="2" />
            <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
            <line x1="10" y1="11" x2="10" y2="17" />
            <line x1="14" y1="11" x2="14" y2="17" />
          </svg>
        </span>
        <span className="text-xs mt-1 text-black">Clear</span>
      </button>
    </div>
  );
};

export default Toolbar; 