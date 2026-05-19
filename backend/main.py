import json
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_manager import ModelManager
from inference_pool import InferencePool

CONFIG_PATH = "model_config.json"
NUM_WORKERS = 1

model_manager: Optional[ModelManager] = None
inference_pool: Optional[InferencePool] = None
current_model_name: Optional[str] = None

frame_times = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager, inference_pool

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    model_manager = ModelManager(config)
    inference_pool = InferencePool(num_workers=NUM_WORKERS)

    print("Server started")

    yield

    if inference_pool:
        inference_pool.stop()

    print("Server stopped")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models")
async def get_models():
    """Returns list of available models"""
    return list(model_manager.config.keys())


@app.post("/set_model/{model_name}")
async def set_model(model_name: str):
    """Sets active model"""
    global current_model_name

    if model_name not in model_manager.config:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model {model_name} not found"}
        )

    current_model_name = model_name
    model_instance = model_manager.get_model(model_name)
    inference_pool.set_model(model_instance)

    print(f"Model changed to: {model_name}")
    return {"status": "ok", "model": model_name}


@app.post("/frame")
async def receive_frame(file: UploadFile = File(...)):
    """Receives frame for inference"""
    global frame_times

    if not current_model_name:
        return {"status": "no_model_selected"}

    frame_data = await file.read()

    inference_pool.add_frame(frame_data)
    frame_times.append(time.time())

    return {"status": "queued"}


@app.get("/result")
async def get_result():
    """Returns latest bounding boxes"""
    return inference_pool.get_latest_result()


@app.get("/metrics")
async def get_metrics():
    """Returns system performance metrics"""
    global frame_times

    current_time = time.time()

    frame_times = [t for t in frame_times if current_time - t < 1.0]
    fps = len(frame_times)

    avg_inference_time = inference_pool.get_average_inference_time()

    return {
        "model": current_model_name or "None",
        "fps": fps,
        "inference_time": avg_inference_time
    }