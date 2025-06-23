import React from "react";

interface ControlsProps {
  onUpload: () => void;
  onGenerate: () => void;
}

const Controls: React.FC<ControlsProps> = ({ onUpload, onGenerate }) => {
  return (
    <div className="controls flex flex-col items-center gap-4 my-4 w-full">
      <button
        className="upload-btn w-full bg-[#4CAF50] hover:bg-[#388E3C] text-white px-6 py-3 rounded-full font-semibold text-lg shadow border-2 border-white transition-all duration-150"
        type="button"
        onClick={onUpload}
      >
        <span className="text-white">Upload</span>
      </button>
      <button
        className="generate-btn w-full bg-[#4CAF50] hover:bg-[#388E3C] text-white px-6 py-3 rounded-full font-semibold text-lg shadow border-2 border-white transition-all duration-150"
        type="button"
        onClick={onGenerate}
      >
        <span className="text-white">Generate</span>
      </button>
    </div>
  );
};

export default Controls; 