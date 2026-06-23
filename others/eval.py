"""
evaluation_app.py  —  End-to-End Ground Truth Annotation
Run: streamlit run evaluation_app.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Import your actual pipeline here
# from pipeline import DocumentPipeline 

st.set_page_config(page_title="Pipeline Evaluation Tool", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; color: #333D47; }
.definition-box { background-color: #F8F9FA; border-left: 4px solid #4C4794; padding: 15px; margin-bottom: 20px; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- Mock Pipeline ---
class MockPipeline:
    def process(self, img_np):
        return [
            {
                "id": "STAMP-01", "type": "Entry", "box": [50, 50, 350, 200], "det_conf": 0.95,
                "dates": [{"value": "12 MAR 2026", "type": "Entry", "ocr_conf": 0.98}]
            }
        ]

@st.cache_resource
def get_pipeline():
    return MockPipeline() 

pipeline = get_pipeline()

# --- Utility Functions ---
def draw_boxes(img: np.ndarray, stamps: list[dict]) -> np.ndarray:
    out = img.copy()
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        color = (148, 71, 76) if s["type"] == "Entry" else (69, 17, 211) 
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{s['type']} ({s['det_conf']:.2f})"
        cv2.putText(out, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out

def get_sample_images(folder_path="./samples/"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return []
    valid_exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
        
    summary = df[["TP", "FP", "FN", "TN"]].sum().to_frame().T
    
    tp = summary["TP"].values[0]
    fp = summary["FP"].values[0]
    fn = summary["FN"].values[0]
    tn = summary["TN"].values[0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    summary["Precision"] = precision
    summary["Recall"] = recall
    summary["F1_Score"] = f1
    
    return summary

# --- Session State ---
def init_state():
    if "eval_data" not in st.session_state: st.session_state.eval_data = {}
    if "pipeline_results" not in st.session_state: st.session_state.pipeline_results = {}

# --- Main App ---
def main():
    init_state()
    st.title("End-to-End Pipeline Evaluation")
    
    st.markdown("""
    <div class="definition-box">
        <strong>Evaluation Guidelines (Count per Stamp/Date):</strong><br>
        <ul>
            <li><strong>True Positive (TP):</strong> จำนวนวันที่ที่โมเดลสกัดออกมาได้ <strong>ถูกต้องเป๊ะ</strong> เทียบกับหน้ากระดาษ</li>
            <li><strong>False Positive (FP):</strong> จำนวนวันที่ที่โมเดลสกัดมา <strong>ผิด</strong> (เช่น มั่วเลข, ดึง Until มาตอบแทน Entry)</li>
            <li><strong>False Negative (FN):</strong> จำนวนแสตมป์บนหน้ากระดาษที่โมเดล <strong>หาไม่เจอหรือข้ามไป</strong></li>
            <li><strong>True Negative (TN):</strong> ใส่ 1 ถ้ารูปนี้ไม่มีแสตมป์เลย และโมเดลก็ไม่ดึงอะไรออกมา (ปกติงาน Extraction มักให้ TN เป็น 0)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    image_paths = get_sample_images("./samples/")
    
    if not image_paths:
        st.warning("No images found in the './samples/' directory. Please add some .jpg or .png files.")
        return

    st.sidebar.header("Dataset Progress")
    evaluated_count = len(st.session_state.eval_data)
    total_count = len(image_paths)
    st.sidebar.progress(evaluated_count / total_count if total_count > 0 else 0)
    st.sidebar.markdown(f"**Progress:** {evaluated_count} / {total_count} documents evaluated.")
    
    if st.sidebar.button("Generate Final Report", type="primary", width="stretch"):
        st.session_state.show_report = True
    else:
        if "show_report" not in st.session_state:
            st.session_state.show_report = False

    if st.session_state.show_report:
        st.header("End-to-End Metrics Report")
        if len(st.session_state.eval_data) == 0:
            st.info("Please evaluate at least one document first.")
        else:
            rows = []
            for filepath, evals in st.session_state.eval_data.items():
                filename = os.path.basename(filepath)
                rows.append({
                    "Document": filename,
                    "TP": evals["TP"],
                    "FP": evals["FP"],
                    "FN": evals["FN"],
                    "TN": evals["TN"]
                })
                
            df_eval = pd.DataFrame(rows)
            metrics_df = compute_metrics(df_eval)
            
            st.markdown("### 1. Cumulative Confusion Matrix & Metrics")
            display_df = metrics_df.copy()
            for col in ["Precision", "Recall", "F1_Score"]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                
            st.dataframe(display_df, hide_index=True, width="stretch")
            
            st.markdown("### 2. Business KPI Translation")
            total_tp = metrics_df["TP"].values[0]
            total_fp = metrics_df["FP"].values[0]
            total_fn = metrics_df["FN"].values[0]
            total_expected_stamps = total_tp + total_fn
            
            auto_rate = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
            error_rate = total_fp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
            
            st.markdown(f"""
            - **Automation Rate (Correct Extractions):** {auto_rate:.2%}
            - **Critical Error Rate (Incorrect Extractions):** {error_rate:.2%}
            - **Recall against Ground Truth:** Found {total_tp} out of {total_expected_stamps} expected stamps.
            """)
            
            st.divider()
            csv = df_eval.to_csv(index=False)
            st.download_button("Download Raw Evaluation Data (CSV)", csv, "evaluation_ground_truth.csv", "text/csv", width="stretch")
            
            if st.button("Back to Evaluation", width="stretch"):
                st.session_state.show_report = False
                st.rerun()
        return

    # --- Document Evaluation View ---
    tabs = st.tabs([os.path.basename(p) for p in image_paths])
    
    for idx, filepath in enumerate(image_paths):
        filename = os.path.basename(filepath)
        with tabs[idx]:
            if filepath not in st.session_state.pipeline_results:
                img_pil = Image.open(filepath).convert("RGB")
                img_np = np.array(img_pil)
                with st.spinner("Executing Pipeline..."):
                    results = pipeline.process(img_np)
                st.session_state.pipeline_results[filepath] = {"img": img_np, "stamps": results}
                
            data = st.session_state.pipeline_results[filepath]
            stamps = data["stamps"]
            img_drawn = draw_boxes(data["img"], stamps)
            
            col_img, col_eval = st.columns([1.2, 1])
            
            with col_img:
                st.image(img_drawn, caption="Pipeline Output Visualization", width="stretch")
                
            with col_eval:
                st.markdown("### Model Extraction Results")
                for s in stamps:
                    st.markdown(f"**Region {s['id']} ({s['type']}):**")
                    dates_found = [d["value"] for d in s.get("dates", [])]
                    if dates_found:
                        for d in dates_found:
                            st.markdown(f"- Extracted: `{d}`")
                    else:
                        st.markdown("- `No Valid Dates Found`")
                st.divider()
                
                st.markdown("### Ground Truth Annotation (Count)")
                st.markdown("ระบุจำนวนครั้งที่โมเดลทำถูกและทำพลาดสำหรับภาพนี้:")
                
                saved_data = st.session_state.eval_data.get(filepath, {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
                
                c1, c2 = st.columns(2)
                with c1:
                    tp_val = st.number_input("True Positives (TP)", min_value=0, value=saved_data["TP"], key=f"tp_{filename}")
                    fn_val = st.number_input("False Negatives (FN)", min_value=0, value=saved_data["FN"], key=f"fn_{filename}")
                with c2:
                    fp_val = st.number_input("False Positives (FP)", min_value=0, value=saved_data["FP"], key=f"fp_{filename}")
                    tn_val = st.number_input("True Negatives (TN)", min_value=0, value=saved_data["TN"], key=f"tn_{filename}")
                
                if st.button("Save Annotation", key=f"save_{filename}", type="primary", width="stretch"):
                    st.session_state.eval_data[filepath] = {
                        "TP": tp_val,
                        "FP": fp_val,
                        "FN": fn_val,
                        "TN": tn_val
                    }
                    st.success("Annotation saved.")

if __name__ == "__main__":
    main()