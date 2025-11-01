# uvicorn main:app --reload --host 0.0.0.0 --port 8000
import os
import shutil
from typing import Optional
from pathlib import Path
import tempfile
import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Setup ---
app = FastAPI(
    title="VidSpot YOLOv8 Object Detection API",
    description="API for detecting the first and last frame of a specific object in a video using YOLOv8."
)

# Define origins that are allowed to make requests
# This is crucial for connecting the React frontend (running on 3000) to the backend (running on 8000)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Model Loading ---
# The model file is expected to be present in the root directory or correctly specified path
# Assuming the user has 'yolov8n.pt' in the working directory
MODEL_PATH = "yolov8n.pt"

# Initialize model to be loaded during startup. 
# We'll rely on the startup process to halt if the model can't be found.
# If the file is missing, the application startup will fail, which is better
# than running the API without a core dependency.
try:
    # Check if the model file exists and load it
    if not Path(MODEL_PATH).exists():
        # Raise FileNotFoundError to halt startup
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model loaded successfully from {MODEL_PATH}")
    
except Exception as e:
    print(f"FATAL: Error loading YOLO model. Application will not function: {e}")
    # Reraise the exception to ensure FastAPI startup fails if the model is essential
    raise SystemExit(f"Model loading failed: {e}")


# --- Utility Functions ---

def get_yolo_class_id(target_class: str) -> Optional[int]:
    """Tries to find the COCO class ID for a given class name."""
    # Since the model loading now causes startup to fail if the model is missing, 
    # we can trust that 'model' is an instance of YOLO here.
    
    # We search the values (names) and return the key (ID)
    for class_id, class_name in model.names.items():
        if class_name.lower() == target_class.lower():
            return class_id
            
    return None

def process_video_for_object(video_path: str, target_class_id: int):
    """
    Scans a video for a specific object class and returns the first and last
    frame numbers where it is detected, along with the video's FPS.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video file.")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)

    first_frame = -1
    last_frame = -1
    frame_count = 0

    while cap.isOpened():
        # FIX: Correctly read the frame without re-assigning 'cap'
        ret, frame = cap.read() 
        if not ret:
            break
        
        # Run YOLO inference
        results = model(frame, verbose=False) 
        
        detected = False
        if results and results[0].boxes:
            # Check if the target class ID is in the detected boxes
            detected_classes = results[0].boxes.cls.tolist()
            if target_class_id in detected_classes:
                detected = True

        if detected:
            if first_frame == -1:
                first_frame = frame_count
            last_frame = frame_count
        
        frame_count += 1

    cap.release()
    
    # Return FPS along with frames and total count
    return first_frame, last_frame, frame_count, fps

# --- API Endpoints ---

@app.post("/search")
async def search_video(
    video: UploadFile = File(...),
    target_class: str = Form(...),
):
    # 1. Target Class validation is now handled in get_yolo_class_id, 
    # which is safe because the model MUST have been loaded for the app to start.

    # 2. Validate Target Class
    target_class_id = get_yolo_class_id(target_class)
    if target_class_id is None:
        return JSONResponse(
            status_code=200, # Return 200 but indicate not found/invalid class
            content={
                "message": f"Object '{target_class}' is not a recognized YOLO class. Please try 'person', 'car', 'dog', etc.",
                "first_frame": -1,
                "last_frame": -1,
                "fps": 0.0 # Include FPS in error response too, set to 0
            }
        )

    # 3. Save Uploaded Video to a Temporary File
    temp_dir = tempfile.mkdtemp()
    # Pylance error fixed: temp_dir is guaranteed to be a str here.
    temp_video_path = os.path.join(temp_dir, video.filename)
    
    try:
        # Write the uploaded file content to the temp path
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 4. Process the Video
        # Updated to receive FPS
        first_frame, last_frame, total_frames, fps = process_video_for_object(
            temp_video_path, target_class_id
        )

        # 5. Return Results
        if first_frame == -1:
             response_content = {
                "message": f"Object '{target_class}' was not found in the video over {total_frames} frames.",
                "first_frame": -1,
                "last_frame": -1,
                "fps": fps
            }
        else:
             response_content = {
                "message": f"Object '{target_class}' detected from frame {first_frame} to {last_frame} (Total frames: {total_frames}).",
                "first_frame": first_frame,
                "last_frame": last_frame,
                "fps": fps # Include FPS in successful response
            }
            
        return JSONResponse(content=response_content)

    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Video processing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during processing: {e}")
    finally:
        # 6. Cleanup Temporary File/Directory
        # This cleanup is now inside the 'finally' block of the primary search logic, 
        # ensuring the temporary directory is cleaned up safely.
        shutil.rmtree(temp_dir, ignore_errors=True)

# Basic root endpoint for health check (optional, but good practice)
@app.get("/")
def health_check():
    return {"status": "ok", "service": "VidSpot Backend API", "model_loaded": True} # Always True now if startup succeeded
