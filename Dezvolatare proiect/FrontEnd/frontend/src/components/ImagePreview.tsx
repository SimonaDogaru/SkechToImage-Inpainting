import React from "react";

interface ImagePreviewProps {
  imageUrl?: string;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({ imageUrl }) => {
  return (
    <div className="image-preview flex flex-col items-center gap-2 my-6 w-full">
      <div className="preview-img bg-white rounded-lg overflow-hidden shadow-md w-[512px] h-[512px] flex items-center justify-center border border-[#DDDDDD]">
        {imageUrl ? (
          <img src={imageUrl} alt="Preview" className="object-contain w-full h-full" />
        ) : (
          <span className="text-gray-300 text-2xl">Preview Result</span>
        )}
      </div>
    </div>
  );
};

export default ImagePreview; 