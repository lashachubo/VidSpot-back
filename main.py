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

def frame_has_object(cap, frame_num, target_class):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return False
    return detect_object(frame, target_class)


def binary_search_first(cap, total_frames, target_class):
    left, right = 0, total_frames - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if frame_has_object(cap, mid, target_class):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result


def binary_search_last(cap, total_frames, target_class):
    left, right = 0, total_frames - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if frame_has_object(cap, mid, target_class):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    return result


def find_first_and_last(video_path, target_class):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    first_occurrence = binary_search_first(cap, total_frames, target_class)
    last_occurrence = -1
    if first_occurrence != -1:
        last_occurrence = binary_search_last(cap, total_frames, target_class)

    cap.release()
    return first_occurrence, last_occurrence



@app.post("/search")
async def search(video: UploadFile = File(...), target_class: str = Form(...)):
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as f:
        f.write(await video.read())

    first_frame, last_frame = find_first_and_last(temp_path, target_class)
    os.remove(temp_path)

    if first_frame == -1:
        return JSONResponse(content={"message": f"No '{target_class}' detected."}, status_code=200)
    
    return JSONResponse(content={
        "first_frame": first_frame,
        "last_frame": last_frame,
        "message": f"'{target_class}' appears from frame {first_frame} to {last_frame}"
    }, status_code=200)
