import time
import threading
from typing import List, Dict, Any, Optional
from base_model import BaseModel


class LatestFrameSlot:
    def __init__(self):
        self._frame: Optional[bytes] = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def put(self, frame: bytes):
        with self._lock:
            self._frame = frame
            self._event.set()

    def take(self, timeout: float) -> Optional[bytes]:
        if not self._event.wait(timeout=timeout):
            return None

        with self._lock:
            frame = self._frame
            self._frame = None
            self._event.clear()
            return frame


class InferenceWorker(threading.Thread):
    def __init__(self, worker_id: int, frame_slot: LatestFrameSlot, result_storage: dict):
        super().__init__(daemon=True)

        self.worker_id = worker_id
        self.frame_slot = frame_slot
        self.result_storage = result_storage

        self.model: Optional[BaseModel] = None
        self.running = True

    def set_model(self, model: BaseModel):
        """Sets the model for the worker."""
        self.model = model

    def run(self):
        """Main worker loop."""
        print(f"Worker {self.worker_id} started")

        while self.running:
            frame_data = self.frame_slot.take(timeout=0.1)

            if frame_data is None:
                continue

            if not self.model:
                continue

            try:
                start_time = time.time()

                boxes = self.model.predict(frame_data)

                inference_time = time.time() - start_time

                self.result_storage["latest_result"] = [
                    b.to_dict() for b in boxes
                ]

                self.result_storage["last_inference_time"] = inference_time

            except Exception as e:
                print(f"  Worker {self.worker_id} error: {e}")

    def stop(self):
        """Stops the worker."""
        self.running = False


class InferencePool:
    """Pool of inference workers. Keeps only the latest frame in the slot."""

    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers

        self.frame_slot = LatestFrameSlot()

        self.result_storage = {
            "latest_result": [],
            "last_inference_time": 0.0
        }

        self.workers: List[InferenceWorker] = []
        self.model: Optional[BaseModel] = None

        for i in range(num_workers):
            worker = InferenceWorker(
                i,
                self.frame_slot,
                self.result_storage
            )

            worker.start()
            self.workers.append(worker)

    def set_model(self, model: BaseModel):
        """Sets the model for all workers."""
        self.model = model

        for worker in self.workers:
            worker.set_model(model)

    def add_frame(self, frame_data: bytes):
        """Pushes a frame into the slot, replacing the previous one (drop-old-replace-new)."""
        self.frame_slot.put(frame_data)

    def get_latest_result(self) -> List[Dict[str, Any]]:
        """Returns the latest bounding boxes."""
        return self.result_storage.get("latest_result", [])

    def get_last_inference_time(self) -> float:
        """Returns the most recent inference time."""
        return self.result_storage.get("last_inference_time", 0.0)

    def stop(self):
        """Stops all workers."""
        for worker in self.workers:
            worker.stop()

        for worker in self.workers:
            worker.join(timeout=1.0)