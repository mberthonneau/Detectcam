from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
from pathlib import Path
import asyncio

# Import des classes existantes
from config import ConfigManager
from detectors import YOLODetector
from managers import FrameCache, DetectionManager, VideoProcessor, ROI, Detection

app = FastAPI(title="DetectCam API")

# Configuration CORS pour le développement
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles de données
class ROIData(BaseModel):
    start: tuple[int, int]
    end: tuple[int, int]
    id: int

class DetectionData(BaseModel):
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    frame_index: int
    roi_id: int

# État global de l'application
class AppState:
    def __init__(self):
        self.config = ConfigManager.load_config()
        self.detector = YOLODetector(self.config['yolo'])
        self.frame_cache = FrameCache()
        self.detection_manager = DetectionManager()
        self.video_processor = None
        self.current_video_path = None
        self.rois = []

app_state = AppState()

@app.post("/api/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Sauvegarder la vidéo uploadée
        video_path = Path("uploads") / file.filename
        video_path.parent.mkdir(exist_ok=True)
        
        with open(video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Initialiser le processeur vidéo
        app_state.video_processor = VideoProcessor(
            app_state.config,
            app_state.detector,
            app_state.detection_manager,
            app_state.frame_cache
        )
        
        # Charger la vidéo
        video_info = app_state.video_processor.load_video(str(video_path))
        app_state.current_video_path = str(video_path)
        
        return {"status": "success", "video_info": video_info}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rois")
async def add_roi(roi_data: ROIData):
    try:
        roi = ROI(
            start=roi_data.start,
            end=roi_data.end,
            id=roi_data.id
        )
        app_state.rois.append(roi)
        return {"status": "success", "roi_id": roi.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rois")
async def get_rois():
    return [{"id": roi.id, "start": roi.start, "end": roi.end} for roi in app_state.rois]

@app.delete("/api/rois/{roi_id}")
async def delete_roi(roi_id: int):
    app_state.rois = [roi for roi in app_state.rois if roi.id != roi_id]
    return {"status": "success"}

@app.post("/api/start-detection")
async def start_detection():
    if not app_state.current_video_path:
        raise HTTPException(status_code=400, detail="Aucune vidéo chargée")
    if not app_state.rois:
        raise HTTPException(status_code=400, detail="Aucune ROI définie")
    
    async def progress_stream():
        try:
            progress = 0
            while progress < 100:
                await asyncio.sleep(0.5)
                # Simuler la progression pour l'exemple
                progress += 1
                yield f"data: {json.dumps({'progress': progress})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(progress_stream(), media_type="text/event-stream")

@app.get("/api/detections")
async def get_detections():
    detections = app_state.detection_manager.all_detections
    return [det.to_dict() for det in detections]

@app.post("/api/stop-detection")
async def stop_detection():
    if app_state.video_processor:
        app_state.video_processor.stop_processing()
    return {"status": "success"}

@app.get("/api/backend-info")
async def get_backend_info():
    return {"backend": app_state.detector.backend_info}

def start_server():
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server()