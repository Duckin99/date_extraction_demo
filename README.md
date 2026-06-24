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

## 2. End-to-end result on Test set

Definitions: TP, TN, FP, FN on E2E

- True Positive (TP):
    The system correctly identified a stamp that exists on the document and successfully extracted the date. It is the "exact match."

- True Negative (TN):
    The system correctly identified that no stamp exists on a document page (or a specific region) and returned no data.

- False Positive (FP):
    The system reported a stamp date, but it was incorrect. This could be due to an OCR error. This is a critical error because the extracted date was wrong.

- False Negative (FN):
    A stamp exists on the document, but the system failed to extract any date. This is missing. It requires a human to manually complete the data entry.

<p align="center">
  <img src="res/e2e_evaluation_report.png" width="60%" alt="E2E Confusion Matrix">
  <br>
  <em>Figure 1: E2E Confusion Matrix</em>
</p>