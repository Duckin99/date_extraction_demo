import cv2
import numpy as np
import base64
import math
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from azure.ai.documentintelligence import DocumentIntelligenceClient

class StampExtraction(BaseModel):
    date: Optional[str] = Field(
        description="The extracted primary date in strictly DD MMM YYYY format. Return null if unreadable."
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
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version
        )
        self.deployment_name = deployment_name

    def extract(self, img_crop: np.ndarray, stamp_type: str) -> Tuple[str, float]:
        _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        
        base_rules = (
            "CRITICAL RULES:\n"
            "- Format: The month is strictly a 3-letter abbreviation (JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC). The year is strictly 4 digits.\n"
            "- Artifacts: Single-digit days are often preceded by a dash or ink artifact (e.g., '-8' or '08'). Do not hallucinate this as a completely different number like '30'. Read the visible digit and normalize it (e.g., '08').\n"
            "- STRICT ACCURACY: If the date is blurry, ink-smudged, invisible, or if you are highly uncertain about ANY digit, you MUST return null. Do not guess."
        )

        if stamp_type == "Entry":
            prompt = (
                "You are analyzing a cropped passport Entry stamp.\n"
                "Task: Extract ONLY the primary Entry date.\n"
                "Note: Entry stamps typically contain two dates. You must return ONLY the chronologically earlier (lesser) date, usually located near the middle of the stamp. Ignore the 'Until' expiration date.\n\n"
                f"{base_rules}"
            )
        else:
            prompt = (
                "You are analyzing a cropped passport Exit stamp.\n"
                "Task: Extract ONLY the primary Exit date.\n\n"
                f"{base_rules}"
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
                temperature=0.0,
                logprobs=True
            )
            
            result = response.choices[0].message.parsed
            final_date = result.date if result.date else ""
            confidence = 0.0
            
            if final_date and response.choices[0].logprobs and response.choices[0].logprobs.content:
                tokens = response.choices[0].logprobs.content
                date_logprobs = []
                
                for t in tokens:
                    clean_t = t.token.replace('"', '').replace('{', '').replace('}', '').replace(':', '').replace('\n', '').strip()
                    
                    if clean_t and clean_t in final_date:
                        date_logprobs.append(math.exp(t.logprob))
                        
                if date_logprobs:
                    confidence = sum(date_logprobs) / len(date_logprobs)
            
            return final_date, confidence
            
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