import cv2
import numpy as np
import base64
import math
from typing import Tuple, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

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
    def __init__(self, endpoint: str, api_version: str = "2024-10-21", deployment_name: str = "gpt-4o"):
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
        
        # 1. High-Precision Rules targeting Hallucination and Spatial Fragility
        base_rules = (
            "CRITICAL EXTRACTION RULES:\n"
            "1. IMAGE ORIENTATION: The stamp may be heavily rotated, flipped, or upside down. Mentally orient the text to its correct reading angle before extracting.\n"
            "2. PARTIAL DATES (NO GUESSING): A valid date MUST clearly show the Day, Month, and Year. If the Day is missing, faded, cropped out, or illegible (e.g., you only see 'OCT 2026'), you MUST return null. Never invent or assume missing digits.\n"
            "3. INK ARTIFACTS: Single-digit days may have stray borders or ink artifacts (e.g., '-8' or ']8'). Read the actual digit (e.g., '08') and ignore the artifact. Do not hallucinate entirely different numbers like '30'.\n"
            "4. STRICT NULL MANDATE: False positives are strictly forbidden. If you are not 100% certain of all three components (DD, MMM, YYYY), return null immediately."
        )

        if stamp_type == "Entry":
            prompt = (
                "You are an expert data extraction system analyzing a cropped passport Entry stamp.\n"
                "Task: Extract ONLY the primary Entry date.\n"
                "Note: Entry stamps often contain two dates. You must return ONLY the chronologically earlier (lesser) date, which is typically near the center. Completely ignore any 'Until' or expiration dates.\n\n"
                f"{base_rules}"
            )
        else:
            prompt = (
                "You are an expert data extraction system analyzing a cropped passport Exit stamp.\n"
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
                                    "detail": "high" # Bump to high for better rotation/resolution handling
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