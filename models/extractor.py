import cv2
import numpy as np
from typing import Tuple, List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

class AzureExtractor:
    def __init__(self, endpoint: str, key: str):
        """Initializes the official Azure SDK Client."""
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

    def extract(self, img_crop: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        """Sends cropped image to Azure and returns text and word confidences."""
        if not self.client.endpoint:
            return "", []

        # Convert OpenCV RGB array to JPEG bytes
        _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
        image_bytes = encoded_image.tobytes()
        
        try:
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-read", 
                body=image_bytes
            )
            
            result = poller.result()
            
            full_text = result.content
            words_data = []
            
            if result.pages:
                for page in result.pages:
                    if page.words:
                        for word in page.words:
                            words_data.append({
                                "text": word.content,
                                "conf": word.confidence,
                                "box": word.polygon # [x1, y1, x2, y2, x3, y3, x4, y4]
                            })
                            
            return full_text, words_data
                
        except Exception as e:
            print(f"Azure SDK Error: {e}")
            return "", []