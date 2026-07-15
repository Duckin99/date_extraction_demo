import numpy as np
from settings.config import settings
from models.detector import YoloDetector
from models.extractor import AzureOpenAIExtractor

class DocumentPipeline:
    def __init__(self):
        self.detector = YoloDetector(settings.YOLO_WEIGHTS_PATH)
        self.extractor = AzureOpenAIExtractor(
            endpoint=settings.AZURE_OPENAI_ENDPOINT, 
            deployment_name="gpt-4o-mini"
        )

    def process(self, img_np: np.ndarray) -> list[dict]:
        """Executes the full End-to-End LLM-powered Extraction workflow."""
        stamps = self.detector.detect(img_np)
        
        results = []
        for idx, stamp in enumerate(stamps):
            x1, y1, x2, y2 = stamp["box"]
            stamp_type = stamp["type"]
            
            # Add padding to crop for better LLM vision accuracy
            h, w, _ = img_np.shape
            pad = 12
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
            
            crop_np = img_np[cy1:cy2, cx1:cx2]
            
            extracted_date, confidence = self.extractor.extract(crop_np, stamp_type)
            
            formatted_dates = []
            if extracted_date:
                formatted_dates.append({
                    "value": extracted_date,
                    "type": stamp_type,
                    "ocr_conf": confidence
                })
            else:
                formatted_dates.append({
                    "value": "",
                    "type": stamp_type,
                    "ocr_conf": 0.0
                })
            
            results.append({
                "id": f"STAMP-{idx + 1:02d}",
                "type": stamp_type,
                "box": stamp["box"],
                "det_conf": stamp["det_conf"],
                "dates": formatted_dates
            })
            
        return results