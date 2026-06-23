"""
demo_app.py  —  Passport Stamp Extraction Demo
Run: streamlit run demo_app.py
"""

import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime

st.set_page_config(page_title="Passport Stamp Extraction Demo", layout="wide")

# Custom CSS for clean, formal typography using Charcoal
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; color: #333D47; }
.stButton > button { width: 100%; border-radius: 4px; font-weight: 600; }
.info-text { font-size: 14px; color: #333D47; margin-bottom: 15px; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# --- Exact HEX Palette ---
PALETTE = {
    "Entry": "#4C4794",   # PURPLE
    "Exit": "#D31145",    # RED
    "Until": "#FF7A85",   # SALMON
    "Text": "#333D47",    # CHARCOAL
    "Highlight": "#BA0361" # CERISE (Used for confidence shading)
}

def hex_to_bgr(hex_str: str):
    """Converts HEX color to OpenCV BGR format."""
    h = hex_str.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

# --- Mock Pipeline ---
def run_pipeline(img_np: np.ndarray) -> list[dict]:
    time.sleep(0.8) 
    return [
        {
            "id": "STAMP-01",
            "type": "Entry",
            "box": [50, 50, 350, 200], 
            "det_conf": 0.95,
            "dates": [
                {"value": "12 MAR 2026", "type": "Entry", "ocr_conf": 0.98},
                {"value": "10 APR 2026", "type": "Until",  "ocr_conf": 0.55}, 
            ],
        },
        {
            "id": "STAMP-02",
            "type": "Exit",
            "box": [300, 250, 600, 400],
            "det_conf": 0.88, 
            "dates": [
                {"value": "15 MAR 2026", "type": "Exit", "ocr_conf": 0.92},
            ],
        },
    ]

# --- Image Processing Utilities ---
def draw_boxes(img: np.ndarray, stamps: list[dict]) -> np.ndarray:
    """Draws solid bounding boxes based on the exact HEX palette."""
    out = img.copy()
    
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        hex_color = PALETTE.get(s["type"], PALETTE["Text"])
        color_bgr = hex_to_bgr(hex_color)
        
        # Solid border
        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 3)
        
        # Solid Label Background
        label = f"{s['type']} ({s['det_conf']:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color_bgr, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
    return out

def crop_stamp(img: np.ndarray, box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    h, w, _ = img.shape
    x1, y1 = max(0, x1 - 10), max(0, y1 - 10)
    x2, y2 = min(w, x2 + 10), min(h, y2 + 10)
    return img[y1:y2, x1:x2]

# --- Data Styling ---
def style_confidence(val):
    """Applies Cerise color fading logic to table cells based on confidence score."""
    # Convert HEX BA0361 to RGB for rgba string: 186, 3, 97
    alpha = float(val)
    # High confidence = solid color, Low confidence = faded/transparent
    color = f'background-color: rgba(186, 3, 97, {alpha}); color: #FFFFFF;'
    return color

# --- Session State ---
def init_state():
    if "results" not in st.session_state: st.session_state.results = {}
    if "verified" not in st.session_state: st.session_state.verified = set()
    if "processed" not in st.session_state: st.session_state.processed = set()

# --- Main App ---
def main():
    init_state()
    st.title("Document Data Verification")
    
    st.markdown("""
    <div class="info-text">
        <strong>Review Guidelines:</strong> Ensure extracted dates align with the physical document. 
        In the data tables, the Confidence column utilizes opacity shading. 
        Solid Cerise indicates high system confidence, while faded cells highlight uncertain data requiring human review.
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Workspace")
        uploads = st.file_uploader(
            "Select Scanned Documents", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, label_visibility="collapsed"
        )

        if uploads:
            total = len(uploads)
            done = len(st.session_state.verified)
            
            st.markdown(f"**Verification Progress: {done} / {total}**")
            st.progress(done / total if total else 0)
            st.divider()

            for f in uploads:
                status = "[Verified]" if f.name in st.session_state.verified else "[Pending]"
                st.markdown(f"{status} {f.name}")

        if st.session_state.verified:
            st.divider()
            if st.button("Export Verified Data", type="secondary", width="stretch"):
                all_rows = []
                for name, stamps in st.session_state.results.items():
                    if name in st.session_state.verified:
                        for s in stamps:
                            for d in s["dates"]:
                                all_rows.append({
                                    "Document": name,
                                    "Stamp_ID": s["id"],
                                    "Stamp_Type": s["type"],
                                    "Date_Type": d["type"],
                                    "Extracted_Date": d["value"],
                                    "System_Confidence": round(s["det_conf"] * d["ocr_conf"], 3),
                                })
                csv = pd.DataFrame(all_rows).to_csv(index=False)
                st.download_button("Download CSV Record", csv, "verified_records.csv", "text/csv", width="stretch")

    if not uploads:
        st.info("System Ready. Please upload documents in the workspace sidebar to begin processing.")
        return

    pending = [f for f in uploads if f.name not in st.session_state.verified]

    if not pending:
        st.success("All documents have been successfully verified. Data is ready for export.")
        return

    tabs = st.tabs([f.name for f in pending])

    for tab, file in zip(tabs, pending):
        with tab:
            if file.name not in st.session_state.processed:
                img_pil = Image.open(file).convert("RGB")
                img_np  = np.array(img_pil)
                with st.spinner("Processing Document Architecture..."):
                    stamps = run_pipeline(img_np)
                st.session_state.results[file.name] = stamps
                st.session_state.processed.add(file.name)

            stamps = st.session_state.results[file.name]
            img_pil = Image.open(file).convert("RGB")
            img_np = np.array(img_pil)
            
            # Solid standard boxes
            img_drawn = draw_boxes(img_np, stamps)

            col_context, col_review = st.columns([1.2, 1.5])

            with col_context:
                st.markdown("**Full Document Context**")
                st.image(img_drawn, width="stretch")

            with col_review:
                st.markdown("**Targeted Data Review**")
                
                for idx, stamp in enumerate(stamps):
                    st.markdown(f"**Region: {stamp['id']}**")
                    
                    r_col1, r_col2 = st.columns([1, 1.5])
                    
                    with r_col1:
                        stamp_crop = crop_stamp(img_np, stamp["box"])
                        st.image(stamp_crop, width="stretch")
                        st.markdown(f"<span style='font-size: 12px; color: {PALETTE['Text']};'>Detection Conf: {stamp['det_conf']:.2f}</span>", unsafe_allow_html=True)

                    with r_col2:
                        rows = []
                        for d in stamp["dates"]:
                            # Convert string date to actual Python datetime object for the Date Picker widget
                            try:
                                parsed_date = datetime.strptime(d["value"], "%d %b %Y").date()
                            except ValueError:
                                parsed_date = None
                                
                            rows.append({
                                "Data Classification": d["type"],
                                "Extracted Date": parsed_date,
                                "Confidence": round(stamp["det_conf"] * d["ocr_conf"], 3),
                            })
                        
                        df = pd.DataFrame(rows)
                        
                        # Apply pandas styling to the dataframe for the confidence fading logic
                        styled_df = df.style.map(style_confidence, subset=['Confidence'])
                        
                        edited_df = st.data_editor(
                            styled_df,
                            num_rows="dynamic",
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Data Classification": st.column_config.SelectboxColumn(
                                    options=["Entry", "Exit", "Until", "Unknown"],
                                    required=True
                                ),
                                "Extracted Date": st.column_config.DateColumn(
                                    "Date (Click to open calendar)", 
                                    format="DD MMM YYYY",
                                    required=True
                                ),
                                "Confidence": st.column_config.NumberColumn(
                                    format="%.3f", 
                                    disabled=True
                                ),
                            },
                            key=f"editor_{file.name}_{stamp['id']}"
                        )
                    st.divider()

                if st.button("Commit Verification", key=f"verify_{file.name}", type="primary", width="stretch"):
                    st.session_state.verified.add(file.name)
                    st.rerun()

if __name__ == "__main__":
    main()