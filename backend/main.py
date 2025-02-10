# backend/main.py
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from io import BytesIO
from skimage.color import rgb2lab, lab2rgb
from torchvision import transforms

# --------------------------
# Model Architecture (Must match Colab exactly)
# --------------------------
from torchvision.models import resnet18
from fastai.vision.learner import create_body
from fastai.vision.models.unet import DynamicUnet
from torch import nn

from model import build_res_unet, MainModel

# --------------------------
# Initialize FastAPI and Model
# --------------------------
app = FastAPI()

# Allow CORS for React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))

model = MainModel(net_G=net_G)
model.load_state_dict(torch.load("final_model_weights.pt", map_location=device))


# --------------------------
# Preprocessing/Postprocessing
# --------------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Replicate Colab's training preprocessing"""
    # Resize and convert to LAB
    image = image.resize((256, 256), Image.BICUBIC)
    img_np = np.array(image)  # Convert to NumPy array

    # Convert RGB to LAB
    img_lab = rgb2lab(img_np).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)  # Convert LAB to Tensor

    # Extract L and ab channels
    L = img_lab[[0], ...] / 50. - 1.  # Normalize L: [0, 100] → [-1, 1]
    ab = img_lab[[1, 2], ...] / 110.  # Normalize ab: [-110, 110] → [-1, 1]

    # Return as a dictionary (same as training)
    return {"L": L.unsqueeze(0), "ab": ab.unsqueeze(0)} 


def lab_to_rgb(L, ab):
    """
    Converts L and ab tensors to RGB format.
    If input is a batch (B, 1, H, W) or (B, 2, H, W), it processes batch-wise.
    If input is a single image (1, H, W), it converts one image.
    """
    L = (L + 1.) * 50.  # Convert L from [-1,1] to [0,100]
    ab = ab * 110.  # Convert ab from [-1,1] to [-110,110]

    if L.dim() == 3:  # If single image (1, H, W), add batch dimension
        L = L.unsqueeze(0)
        ab = ab.unsqueeze(0)

    # Convert from PyTorch tensor to NumPy array
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()

    rgb_imgs = [lab2rgb(img) for img in Lab]  # Convert each image in batch
    return np.stack(rgb_imgs, axis=0)  # Stack to return batch-like output


# --------------------------
# API Endpoints
# --------------------------
@app.post("/colorize/image")
async def colorize_image(file: UploadFile = File(...)):
    # Validate input
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload an image file (JPEG/PNG)")
    
    try:
        # Read and preprocess
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        print(f"Received file: {file.filename}, Content Type: {file.content_type}")

        original_size = image.size
        L_tensor = preprocess_image(image)
        
        # Predict
        model.net_G.eval()
        with torch.no_grad():
            model.setup_input(L_tensor)
            model.forward()
            print("hello")
            model.net_G.train()
            fake_color = model.fake_color.detach()
            L = model.L

           
        fake_imgs = lab_to_rgb(L, fake_color)  # Convert LAB to RGB
        fake_imgs = (fake_imgs[0] * 255).astype(np.uint8)  # Rescale to [0,255]
        
        # Resize back to original size
        fake_imgs = cv2.resize(fake_imgs, original_size)
        
        # Return as PNG
        _, buffer = cv2.imencode(".png", cv2.cvtColor(fake_imgs, cv2.COLOR_RGB2BGR))
        return Response(content=buffer.tobytes(), media_type="image/png")
    
    except Exception as e:
       print(f"Error during model inference: {e}")

import cv2
import tempfile
import shutil
import os

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles

os.makedirs("static/videos", exist_ok=True)


app.mount("/static", StaticFiles(directory="static"), name="static")



@app.post("/colorize/video")
async def colorize_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Upload a video file (MP4, AVI, etc.)")

    try:
        temp_dir = tempfile.mkdtemp()
        temp_video_path = os.path.join(temp_dir, file.filename)

        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise HTTPException(500, "Could not open video file")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video_path = os.path.join(temp_dir, "colorized_" + file.filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data = preprocess_image(image)
            L_tensor = data["L"]

            with torch.no_grad():
                fake_color = model.net_G(L_tensor)

            fake_imgs = lab_to_rgb(L_tensor, fake_color)
            fake_imgs = (fake_imgs[0] * 255).astype(np.uint8)
            fake_imgs = cv2.resize(fake_imgs, (frame_width, frame_height))

            colorized_frame = cv2.cvtColor(fake_imgs, cv2.COLOR_RGB2BGR)
            out.write(colorized_frame)

        cap.release()
        out.release()

        return FileResponse(
            output_video_path, 
            media_type="video/mp4",
            filename="colorized_output.mp4",  # Ensures browser treats it as a file
        )

    except Exception as e:
        print(f"Error during video processing: {e}")
        raise HTTPException(500, f"Internal Server Error: {str(e)}")



# --------------------------
# Run Server
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)