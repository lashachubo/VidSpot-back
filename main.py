# uvicorn main:app --reload --host 0.0.0.0 --port 8000
import os
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

import cv2
import torch
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging

# --- Configuration ---
class Config:
    MODEL_PATH = "yolov8n.pt"
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    MAX_VIDEO_SIZE_MB = 500
    MAX_VIDEO_SIZE_BYTES = MAX_VIDEO_SIZE_MB * 1024 * 1024
    FRAME_SKIP = 1  # Process every Nth frame (1 = all frames, 2 = every other frame)
    BATCH_SIZE = 32  # Batch size for YOLO inference
    ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
    LOG_LEVEL = "INFO"

config = Config()

# --- Logging Setup ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Response Models ---
class SearchResponse(BaseModel):
    message: str
    first_frame: int
    last_frame: int
    total_frames: int
    fps: float
    first_timestamp: float = Field(description="Time in seconds")
    last_timestamp: float = Field(description="Time in seconds")
    duration: float = Field(description="Detection duration in seconds")
    confidence: Optional[float] = Field(None, description="Average detection confidence")

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events"""
    # Startup
    logger.info("Starting VidSpot API...")
    
    if not Path(config.MODEL_PATH).exists():
        logger.error(f"Model file not found at: {config.MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at: {config.MODEL_PATH}")
    
    try:
        app.state.model = YOLO(config.MODEL_PATH)
        logger.info(f"YOLOv8 model loaded successfully from {config.MODEL_PATH}")
        
        # Create thread pool for async processing
        app.state.executor = ThreadPoolExecutor(max_workers=2)
        logger.info("Thread pool executor initialized")
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise SystemExit(f"Model loading failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down VidSpot API...")
    app.state.executor.shutdown(wait=True)
    logger.info("Cleanup complete")

# --- FastAPI App Setup ---
app = FastAPI(
    title="VidSpot YOLOv8 Object Detection API",
    description="API for detecting the first and last frame of a specific object in a video using YOLOv8.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    # Check filename exists
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )
    
    # Check extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(config.ALLOWED_VIDEO_EXTENSIONS)}"
        )
    
    # Check MIME type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid content type. Must be a video file."
        )

def get_yolo_class_id(model: YOLO, target_class: str) -> Optional[int]:
    """Find the COCO class ID for a given class name"""
    for class_id, class_name in model.names.items():
        if class_name.lower() == target_class.lower():
            return class_id
    return None

def get_available_classes(model: YOLO) -> list:
    """Get list of all available YOLO classes"""
    return sorted(model.names.values())

def process_video_for_object(
    video_path: str, 
    target_class_id: int,
    model: YOLO,
    frame_skip: int = 1
) -> Dict[str, Any]:
    """
    Scan video for a specific object class and return detection information.
    
    Args:
        video_path: Path to video file
        target_class_id: YOLO class ID to detect
        model: Loaded YOLO model
        frame_skip: Process every Nth frame (default: 1 = all frames)
    
    Returns:
        Dictionary with detection results
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not open video file. File may be corrupted.")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        first_frame = -1
        last_frame = -1
        frame_count = 0
        confidences = []
        
        frames_batch = []
        frame_indices = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if configured
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            frames_batch.append(frame)
            frame_indices.append(frame_count)
            
            # Process in batches
            if len(frames_batch) >= config.BATCH_SIZE:
                results = model(frames_batch, verbose=False)
                
                for i, result in enumerate(results):
                    if result.boxes:
                        detected_classes = result.boxes.cls.tolist()
                        if target_class_id in detected_classes:
                            current_frame = frame_indices[i]
                            if first_frame == -1:
                                first_frame = current_frame
                            last_frame = current_frame
                            
                            # Store confidence scores
                            mask = result.boxes.cls == target_class_id
                            conf_values = result.boxes.conf[mask].tolist()
                            confidences.extend(conf_values)
                
                frames_batch = []
                frame_indices = []
            
            frame_count += 1
        
        # Process remaining frames
        if frames_batch:
            results = model(frames_batch, verbose=False)
            for i, result in enumerate(results):
                if result.boxes:
                    detected_classes = result.boxes.cls.tolist()
                    if target_class_id in detected_classes:
                        current_frame = frame_indices[i]
                        if first_frame == -1:
                            first_frame = current_frame
                        last_frame = current_frame
                        
                        mask = result.boxes.cls == target_class_id
                        conf_values = result.boxes.conf[mask].tolist()
                        confidences.extend(conf_values)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        return {
            "first_frame": first_frame,
            "last_frame": last_frame,
            "total_frames": total_frames,
            "fps": fps,
            "confidence": avg_confidence
        }
    
    finally:
        cap.release()

# --- API Endpoints ---

@app.get("/")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "VidSpot Backend API",
        "version": "2.0.0",
        "model_loaded": hasattr(app.state, 'model')
    }

@app.get("/classes")
def list_classes():
    """Get list of all available YOLO classes"""
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    classes = get_available_classes(app.state.model)
    return {
        "classes": classes,
        "count": len(classes)
    }

@app.post("/search", response_model=SearchResponse)
async def search_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_class: str = Form(...),
):
    """
    Search for object in uploaded video
    
    Args:
        video: Video file to process
        target_class: Object class to detect (e.g., 'person', 'car', 'dog')
    
    Returns:
        Detection results with frame numbers and timestamps
    """
    
    # Validate filename exists
    if not video.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Validate file
    validate_video_file(video)
    
    # Validate target class
    target_class_id = get_yolo_class_id(app.state.model, target_class)
    if target_class_id is None:
        available_classes = get_available_classes(app.state.model)
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"Object '{target_class}' is not a recognized YOLO class.",
                "available_classes": available_classes[:20],  # Show first 20
                "suggestion": "Try 'person', 'car', 'dog', 'cat', 'bottle', etc."
            }
        )
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video.filename)
    
    try:
        # Save uploaded file
        file_size = 0
        with open(temp_video_path, "wb") as buffer:
            while chunk := await video.read(8192):  # Read in chunks
                file_size += len(chunk)
                if file_size > config.MAX_VIDEO_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {config.MAX_VIDEO_SIZE_MB}MB"
                    )
                buffer.write(chunk)
        
        logger.info(f"Processing video: {video.filename} ({file_size / 1024 / 1024:.2f}MB)")
        
        # Process video in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            app.state.executor,
            process_video_for_object,
            temp_video_path,
            target_class_id,
            app.state.model,
            config.FRAME_SKIP
        )
        
        # Calculate timestamps
        fps = result["fps"]
        first_frame = result["first_frame"]
        last_frame = result["last_frame"]
        
        if first_frame == -1:
            return SearchResponse(
                message=f"Object '{target_class}' was not found in the video.",
                first_frame=-1,
                last_frame=-1,
                total_frames=result["total_frames"],
                fps=fps,
                first_timestamp=0.0,
                last_timestamp=0.0,
                duration=0.0,
                confidence=None
            )
        
        first_timestamp = first_frame / fps if fps > 0 else 0
        last_timestamp = last_frame / fps if fps > 0 else 0
        duration = last_timestamp - first_timestamp
        
        logger.info(
            f"Detection complete: {target_class} found in frames {first_frame}-{last_frame} "
            f"({first_timestamp:.2f}s-{last_timestamp:.2f}s)"
        )
        
        return SearchResponse(
            message=f"Object '{target_class}' detected from frame {first_frame} to {last_frame}.",
            first_frame=first_frame,
            last_frame=last_frame,
            total_frames=result["total_frames"],
            fps=fps,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            duration=duration,
            confidence=result["confidence"]
        )
    
    except IOError as e:
        logger.error(f"Video processing error: {e}")
        raise HTTPException(status_code=422, detail=f"Video processing error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)
        await video.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)