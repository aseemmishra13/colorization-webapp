"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [fileType, setFileType] = useState(null); // Track whether it's an image or video

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    if (!selectedFile) return;
    setFile(selectedFile);

    // Detect if file is an image or video
    if (selectedFile.type.startsWith("image/")) {
      setFileType("image");
      setPreview(URL.createObjectURL(selectedFile));
    } else if (selectedFile.type.startsWith("video/")) {
      setFileType("video");
      setPreview(URL.createObjectURL(selectedFile));
    } else {
      alert("Please upload an image or video.");
      setFile(null);
      setPreview(null);
      setFileType(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setIsProcessing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      let endpoint = "http://localhost:8000/colorize/image"; // Default for images

      if (fileType === "video") {
        endpoint = "http://localhost:8000/colorize/video";
      }

      const response = await axios.post(endpoint, formData, {
        responseType: "blob",
      });

      const url = URL.createObjectURL(response.data);
      setResult(url);
    } catch (error) {
      console.error("Error:", error);
      alert("Processing failed. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto text-center">
      <h1 className="text-2xl font-bold mb-4">Upload Image or Video for Colorization</h1>

      <input
        type="file"
        onChange={handleFileChange}
        className="mb-4 block mx-auto"
        accept="image/*,video/*"
      />

      {preview && (
        <div className="mb-4">
          {fileType === "image" ? (
            <img src={preview} alt="Preview" className="max-w-full max-h-80 mx-auto" />
          ) : (
            <video controls className="max-w-full max-h-80 mx-auto">
              <source src={preview} type={file.type} />
              Your browser does not support the video tag.
            </video>
          )}
        </div>
      )}

      <button
        onClick={handleUpload}
        className="bg-blue-500 text-white px-6 py-2 rounded-lg disabled:bg-gray-400"
        disabled={isProcessing || !file}
      >
        {isProcessing ? "Processing..." : "Upload & Colorize"}
      </button>

      {result && (
        <div className="mt-6">
          <h2 className="text-lg font-bold">Colorized Output:</h2>
          {fileType === "image" ? (
            <img src={result} alt="Colorized" className="max-w-full max-h-80 mx-auto mt-2" />
          ) : (
            <video controls className="max-w-full max-h-80 mx-auto mt-2">
              <source src={result} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          )}
        </div>
      )}
    </div>
  );
}
