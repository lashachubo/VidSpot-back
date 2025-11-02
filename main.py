import os
import shutil
import tempfile
from typing import Optional
from pathlib import Path
from contextlib import contextmanager

import cv2
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# --- FastAPI App Setup ---
app = FastAPI(
    title="VidSpot YOLOv8 Object Detection API",
    description="API for detecting the first and last frame of a specific object in a video using YOLOv8."
)

# Define origins for CORS (React development server)
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Model Loading ---
MODEL_PATH = "yolov8n.pt"
model: YOLO # Explicit type hint for Pylance

try:
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    # Load the model once at startup
    model = YOLO(MODEL_PATH)
    print(f"YOLOv8 model loaded successfully from {MODEL_PATH}")
    
except Exception as e:
    print(f"FATAL: Error loading YOLO model. Application will not function: {e}")
    # Halt startup if the core model dependency fails
    raise SystemExit(f"Model loading failed: {e}")


# --- Utility Functions & Context Manager ---

@contextmanager
def video_capture_context(video_path: str):
    """
    Custom context manager for cv2.VideoCapture to ensure resources are released.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError("Could not open video file.")
    try:
        yield cap
    finally:
        cap.release()

def get_yolo_class_id(target_class: str, yolo_model: YOLO) -> Optional[int]:
    """Tries to find the COCO class ID for a given class name."""
    # Use the model's names dictionary to efficiently find the class ID
    for class_id, class_name in yolo_model.names.items():
        if class_name.lower() == target_class.lower():
            return class_id
            
    return None

def process_video_for_object(video_path: str, target_class_id: int, yolo_model: YOLO):
    """
    Scans a video for a specific object class and returns the first and last
    frame numbers where it is detected, along with the video's FPS.
    """
    first_frame = -1
    last_frame = -1
    frame_count = 0
    fps = 0.0

    # Use the context manager for safe resource handling
    with video_capture_context(video_path) as cap:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            # Read the frame
            ret, frame = cap.read() 
            if not ret:
                break
            
            # Run YOLO inference (only checking for the presence of the target class)
            # The 'verbose=False' is already good for keeping the console clean.
            results = yolo_model(frame, verbose=False, stream=False) 
            
            detected = False
            # Ensure results are valid and boxes exist
            if results and results[0].boxes is not None:
                # Check if the target class ID is in the detected classes
                # Check is efficient as it only needs to look at the class IDs
                if target_class_id in results[0].boxes.cls.tolist():
                    detected = True

            if detected:
                if first_frame == -1:
                    first_frame = frame_count
                last_frame = frame_count
            
            frame_count += 1
    
    # Return FPS along with frames and total count
    return first_frame, last_frame, frame_count, fps

# --- API Endpoints ---

@app.post("/search")
async def search_video(
    video: UploadFile = File(...),
    target_class: str = Form(...),
):
    # Trim the target class early for robust validation
    trimmed_class = target_class.strip()
    
    # 1. Validate Target Class
    target_class_id = get_yolo_class_id(trimmed_class, model)
    if target_class_id is None:
        return JSONResponse(
            status_code=200, 
            content={
                "message": f"Object '{trimmed_class}' is not a recognized YOLO class. Please try 'person', 'car', 'dog', etc.",
                "first_frame": -1,
                "last_frame": -1,
                "fps": 0.0 
            }
        )

    # 2. Save Uploaded Video to a Temporary File
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)
    
    try:
        # Write the uploaded file content to the temp path
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # 3. Process the Video
        first_frame, last_frame, total_frames, fps = process_video_for_object(
            temp_video_path, target_class_id, model
        )

        # 4. Return Results
        if first_frame == -1:
             response_content = {
                "message": f"Object '{trimmed_class}' was not found in the video over {total_frames} frames.",
                "first_frame": -1,
                "last_frame": -1,
                "fps": fps
            }
        else:
             response_content = {
                "message": f"Object '{trimmed_class}' detected from frame {first_frame} to {last_frame} (Total frames: {total_frames}).",
                "first_frame": first_frame,
                "last_frame": last_frame,
                "fps": fps
            }
            
        return JSONResponse(content=response_content)

    except IOError as e:
        # Catch video opening/reading errors
        raise HTTPException(status_code=500, detail=f"Video processing I/O error: {e}")
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during processing. Check server logs.")
    finally:
        # 5. Cleanup Temporary File/Directory
        shutil.rmtree(temp_dir, ignore_errors=True)

# Basic root endpoint for health check
@app.get("/")
def health_check():
    return {"status": "ok", "service": "VidSpot Backend API", "model_loaded": True}
