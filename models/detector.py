import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class YoloDetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)
        self.names = self.model.names 

    def detect(self, img_np: np.ndarray) -> List[Dict[str, Any]]:
        """Returns localized stamps with coordinates and confidence."""
        results = self.model(img_np, verbose=False)
        stamps = []
        
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                if "entry2" in self.names[cls_id].lower():
                    stamp_type = "Entry"
                elif "exit2" in self.names[cls_id].lower():
                    stamp_type = "Exit"
                else:
                    continue
                
                stamps.append({
                    "box": coords,
                    "det_conf": conf,
                    "type": stamp_type
                })
        return stamps