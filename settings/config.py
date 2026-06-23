import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    YOLO_WEIGHTS_PATH = os.getenv("YOLO_WEIGHTS_PATH", "./weights/best.pt")
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.85"))

settings = Settings()