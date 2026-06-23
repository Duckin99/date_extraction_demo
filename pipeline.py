import numpy as np
from settings.config import settings
from models.detector import YoloDetector
from models.extractor import AzureExtractor
from utils.post_process import post_process

class DocumentPipeline:
    def __init__(self):
        # Initialize models once to prevent reloading weights on every image
        self.detector = YoloDetector(settings.YOLO_WEIGHTS_PATH)
        self.extractor = AzureExtractor(settings.AZURE_ENDPOINT, settings.AZURE_API_KEY)

    def process(self, img_np: np.ndarray) -> list[dict]:
        """Executes the full End-to-End OCR pipeline."""
        stamps = self.detector.detect(img_np)
        
        results = []
        for idx, stamp in enumerate(stamps):
            x1, y1, x2, y2 = stamp["box"]
            
            # Add padding to crop
            h, w, _ = img_np.shape
            pad = 10
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            
            crop_np = img_np[cy1:cy2, cx1:cx2]
            
            # Execute OCR
            raw_text, words_data = self.extractor.extract(crop_np)
            
            # Apply post process
            dates = post_process(raw_text, words_data, stamp["type"])
            
            results.append({
                "id": f"STAMP-{idx + 1:02d}",
                "type": stamp["type"],
                "box": stamp["box"],
                "det_conf": stamp["det_conf"],
                "dates": dates
            })
            
        return results