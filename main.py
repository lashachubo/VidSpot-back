import cv2
from ultralytics import YOLO
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import os

app = FastAPI()
model = YOLO('yolov8n.pt')

def detect_object(frame, target_class="person"):
    results = model(frame, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label.lower() == target_class.lower():
            return True
    return False

def binary_search(video_path, target_class="person"):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    low, high = 0, total_frames - 1
    first_occurrence = -1

    while low <= high:
        mid = (low + high) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ret, frame = cap.read()
        if not ret:
            break

        if detect_object(frame, target_class):
            first_occurrence = mid
            high = mid - 1
        else:
            low = mid + 1

    cap.release()
    return first_occurrence

@app.post("/search")
async def search(video: UploadFile = File(...), target_class: str = Form(...)):
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    frame_number = binary_search(temp_path, target_class)
    os.remove(temp_path)

    if frame_number == -1:
        return JSONResponse(content={"message": f"No '{target_class}' detected."}, status_code=200)
    return JSONResponse(content={"frame_number": frame_number, "message": f"First '{target_class}' found at frame {frame_number}"}, status_code=200)
