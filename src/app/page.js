"use client";
import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { ArrowPathIcon } from "@heroicons/react/24/outline";



export default function Home() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    const socket = new WebSocket("ws://localhost:8000/ws/progress");

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress);
    };

    return () => socket.close();
  }, []);


  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setError(null);
    setFile(file);
    setPreviewUrl(URL.createObjectURL(file));

    if (file.type.startsWith("image/")) {
      setFileType("image");
    } else if (file.type.startsWith("video/")) {
      setFileType("video");
    } else {
      setError("Unsupported file type. Please upload an image or video.");
      setFile(null);
      setPreviewUrl(null);
      setFileType(null);
    }
  };

  const handleColorize = async () => {
    if (!file || !fileType) return;

    setIsProcessing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append("file", file);

      const endpoint = fileType === "image" 
        ? "http://localhost:8000/colorize/image" 
        : "http://localhost:8000/colorize/video";

      const response = await axios.post(endpoint, formData, {
        responseType: "blob",
        timeout: fileType === "image" ? 12000000 : 60000000
      });

      const mimeType = fileType === "image" ? "image/jpeg" : "video/mp4";
      const blob = new Blob([response.data], { type: mimeType });
      const resultUrl = URL.createObjectURL(blob);
      
      setResultUrl(resultUrl);
      setError(null);

      if (fileType === "video" && videoRef.current) {
        videoRef.current.load();
      }

    } catch (err) {
      console.error("Colorization failed:", err);
      setError(err.response?.data?.detail || "Processing failed. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-6">Image & Video Colorization</h1>

        {/* File Upload */}
        <div className="mb-6">
          <label className="block mb-2 text-sm font-medium text-gray-600">
            Upload an image or video:
          </label>
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*,video/*"
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 text-red-700 rounded-lg">
            {error}
          </div>
        )}

        {/* Preview Section */}
        {previewUrl && (
          <div className="mb-6">
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              Original {fileType?.toUpperCase()}
            </h2>
            {fileType === "image" ? (
              <img 
                src={previewUrl} 
                alt="Preview" 
                className="w-full rounded-lg border object-contain max-h-96"
              />
            ) : (
              <video 
                controls
                className="w-full rounded-lg border"
                style={{ maxHeight: "400px" }}
              >
                <source src={previewUrl} type={file?.type} />
                Your browser does not support the video tag.
              </video>
            )}
          </div>
        )}

        {/* Process Button */}
        <button
          onClick={handleColorize}
          disabled={!file || isProcessing}
          className={`w-full py-3 px-6 rounded-lg font-semibold text-white transition-colors flex items-center justify-center gap-2
             ${isProcessing ? "bg-blue-600 cursor-wait" : "bg-blue-600 hover:bg-blue-700"} 
            ${!file && "bg-gray-400 cursor-not-allowed"}`}
        >
          {isProcessing ? (
            <>
              <ArrowPathIcon className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : (
            `Colorize ${fileType?.toUpperCase()}`
          )}
        </button>
        <div className="progress-bar-container">
        <div className="progress-bar" style={{ width: `${progress}%` }}></div>
      </div>
      <p>{progress.toFixed(2)}%</p>

        {/* Processing Status Message */}
        {isProcessing && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <p className="text-blue-800 text-sm text-center">
              {fileType === "image" ? (
                "Colorizing image - this usually takes 15-30 seconds..."
              ) : (
                <>
                  Colorizing video - this may take 2-5 minutes depending on length.<br />
                  <span className="text-xs opacity-75">Please keep this tab open</span>
                </>
              )}
            </p>
          </div>
        )}

        {/* Result Section */}
        {resultUrl && (
          <div className="mt-8">
            <h2 className="text-lg font-semibold text-gray-700 mb-2">
              Colorized Result
            </h2>
            {fileType === "image" ? (
              <img 
                src={resultUrl} 
                alt="Colorized result" 
                className="w-full rounded-lg border object-contain max-h-96"
              />
            ) : (
              <video
                ref={videoRef}
                controls
                className="w-full rounded-lg border"
                style={{ maxHeight: "400px" }}
              >
                <source src={resultUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
            <div className="mt-4 flex justify-end">
              <a
                href={resultUrl}
                download={`colorized-${fileType}.${fileType === 'image' ? 'jpg' : 'mp4'}`}
                className="px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-800 transition-colors"
              >
                Download {fileType?.toUpperCase()}
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}