# backend/main.py
import os
import cv2
import torch
import uvicorn
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer, get_video_colorizer
# At the top of main.py
from fastapi import WebSocket
from websocket_manager import manager 
active_connections = set()
# --------------------------
# Initialize DeOldify
# --------------------------
device.set(device=DeviceId.GPU0)  # Use GPU0 if available
image_colorizer = get_image_colorizer(artistic=False)
video_colorizer = get_video_colorizer()

# --------------------------
# Configure FastAPI
# --------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add to main.py

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

# --------------------------
# Common Functions
# --------------------------
def validate_file_type(file: UploadFile, allowed_types: list):
    if not any(file.content_type.startswith(t) for t in allowed_types):
        raise HTTPException(400, f"Invalid file type. Allowed types: {', '.join(allowed_types)}")

# --------------------------
# Image Endpoint
# --------------------------
@app.post("/colorize/image")
async def colorize_image(file: UploadFile = File(...), render_factor: int = 35):
    validate_file_type(file, ["image/"])
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save uploaded image
            input_path = os.path.join(tmp_dir, "input_image.jpg")
            with open(input_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Colorize with DeOldify
            result_path = image_colorizer.plot_transformed_image(
                input_path,
                render_factor=render_factor,
                compare=False,
                watermarked=False)
            
            # Return processed image
            with open(result_path, "rb") as result_file:
                return Response(
                    content=result_file.read(),
                    media_type="image/jpeg",
                    headers={"Content-Disposition": "attachment; filename=colorized.jpg"}
                )

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(500, f"Image processing failed: {str(e)}")

# --------------------------
# Video Endpoint (Existing)
# --------------------------
@app.post("/colorize/video")
async def colorize_video(file: UploadFile = File(...), render_factor: int = 21):
    validate_file_type(file, ["video/"])
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save uploaded video
            input_path = os.path.join(tmp_dir, "input_video.mp4")
            with open(input_path, "wb") as buffer:
                buffer.write(await file.read())
            
            # Colorize with DeOldify
            output_path = video_colorizer.colorize_from_file_name(
                input_path,
                render_factor=render_factor,
                watermarked=False
            )
            
            # Return processed video
            with open(output_path, "rb") as result_file:
                return Response(
                    content=result_file.read(),
                    media_type="video/mp4",
                    headers={"Content-Disposition": "attachment; filename=colorized.mp4"}
                )

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise HTTPException(500, f"Video processing failed: {str(e)}")

# --------------------------
# Run Server
# --------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000,  ws='websockets')