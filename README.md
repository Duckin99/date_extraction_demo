## 1. Project Structure
```
passport_stamp/
├── .env                  
├── .env.example          # Template for .env
├── app.py                # Streamlit demo
├── settings/
│   ├── __init__.py
│   └── config.py         # Environment loader
├── models/
│   ├── __init__.py
│   ├── detector.py       # YOLO
│   └── extractor.py      # Azure Document Intelligence
├── utils/
│   ├── __init__.py
│   └── post_process.py   # Date parsing logic
└── pipeline.py           # The Orchestrator
```

> Note: `app.py` and `post-process` are fully vibe coded 