import cv2
import numpy as np
import base64
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from azure.ai.documentintelligence import DocumentIntelligenceClient

class StampExtraction(BaseModel):
    date: Optional[str] = Field(
        description="The extracted primary date exactly in DD MMM YYYY format (e.g., '14 OCT 2026'). Return null if unreadable."
    )
    confidence: float = Field(
        description="Float between 0.0 and 1.0 representing your certainty."
    )

    @field_validator('date')
    @classmethod
    def validate_date(cls, v):
        if v is None:
            return v
        try:
            datetime.strptime(v.strip(), "%d %b %Y")
            return v.strip().upper()
        except ValueError:
            return None

class AzureOpenAIExtractor:
    def __init__(self, endpoint: str, api_version: str = "2024-10-21", deployment_name: str = "gpt-4o-mini"):
        # 1. Create the Azure AD Token Provider
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        
        # 2. Initialize the client using azure_ad_token_provider instead of api_key
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def extract(self, img_crop: np.ndarray, stamp_type: str) -> Tuple[str, float]:
        """
        Sends crop to Azure OpenAI using Entra ID authentication and a dynamic prompt.
        """
        _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        
        if stamp_type == "Entry":
            prompt = (
                "You are analyzing a cropped image of a passport Entry stamp. "
                "Your task is to extract ONLY the primary Entry date. "
                "Entry stamps typically contain two dates: the Entry date and an 'Until' date. "
                "You must return ONLY the Entry date, which is chronologically earlier (lesser than) "
                "and is usually located near the middle of the stamp. "
                "Completely ignore the 'Until' or expiration date."
            )
        else:
            prompt = (
                "You are analyzing a cropped image of a passport Exit stamp. "
                "Your task is to extract ONLY the primary Exit date from the image. "
                "Return the exact date shown."
            )

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.deployment_name,
                response_format=StampExtraction,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0
            )
            
            result = response.choices[0].message.parsed
            final_date = result.date if result.date else ""
            
            return final_date, result.confidence
            
        except Exception as e:
            print(f"Azure OpenAI API Error: {e}")
            return "", 0.0

class AzureExtractor:
    def __init__(self, endpoint: str, key: str):
        """Initializes the official Azure SDK Client."""
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, 
            credential=AzureKeyCredential(key)
        )

    def extract(self, img_crop: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
        """Sends cropped image to Azure and returns text and word confidences."""
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