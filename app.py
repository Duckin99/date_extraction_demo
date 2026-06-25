"""
demo_app.py — Passport Stamp Extraction Demo
Run:  streamlit run demo_app.py

Pipeline integration
--------------------
See README.md for the expected return schema and a mock example.
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime
from pipeline import DocumentPipeline

@st.cache_resource
def get_pipeline():
    return DocumentPipeline()

pipeline = get_pipeline()

# ── AIA Colour Palette  (https://design.aia.com/colour) ────────────────────
RED      = "#D31145"  # Digital Red 500       — Exit stamps, primary CTA
CHARCOAL = "#333D47"  # Digital Charcoal 600  — body text, borders
PURPLE   = "#4C4794"  # Digital Purple        — Entry stamps
SALMON   = "#FF7A85"  # Digital Salmon        — Until dates
WHITE    = "#FFFFFF"
GREY     = "#F4F4F4"  # Charcoal 100          — card backgrounds

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED, "Until": SALMON}

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Passport Stamp Review", layout="wide")

st.markdown(f"""<style>
  .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
  p, label {{ color: {CHARCOAL}; font-family: 'Inter', sans-serif; }}
  h1, h2, h3 {{ color: {CHARCOAL} !important; letter-spacing: -0.3px; }}
  .stButton > button {{ border-radius: 4px; font-weight: 600; }}
  .section-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: {CHARCOAL}88; margin-bottom: 8px;
    margin-top: 12px;
  }}
</style>""", unsafe_allow_html=True)


# ── Image helpers ────────────────────────────────────────────────────────────
def to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_boxes(img, stamps, selected_id):
    out = img.copy()
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        color     = to_bgr(TYPE_COLOR.get(s["type"], CHARCOAL))
        is_sel    = s["id"] == selected_id
        thickness = 4 if is_sel else 2
        overlay   = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.addWeighted(overlay, 1.0 if is_sel else 0.5, out, 0.0 if is_sel else 0.5, 0, out)
        label = f"{s['type']}  {s['det_conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 9), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def crop(img, box, pad=16):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]


def conf_bar(value, color):
    pct = int(value * 100)
    return (f"<div style='background:{GREY};border-radius:3px;height:5px;margin-top:4px;'>"
            f"<div style='width:{pct}%;background:{color};height:5px;border-radius:3px;'></div></div>"
            f"<span style='font-size:11px;color:{CHARCOAL}88;'>{pct}%</span>")


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("results", {}), ("selected", {}), ("verified", set())]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{RED};margin-bottom:0;'>Stamp Review</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:12px;color:{CHARCOAL}88;margin-top:2px;'>"
                f"Passport Entry / Exit Detection</p>", unsafe_allow_html=True)
    st.divider()

    uploads = st.file_uploader("Upload passport images", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True)

    if uploads:
        done = len(st.session_state.verified)
        st.markdown(f"**Progress: {done} / {len(uploads)} verified**")
        st.progress(done / len(uploads) if uploads else 0)
        st.divider()
        for f in uploads:
            done_flag = f.name in st.session_state.verified
            dot   = f"<span style='color:{RED};'>&#9679;</span>" if done_flag else \
                    f"<span style='color:{CHARCOAL}44;'>&#9675;</span>"
            label = "Verified" if done_flag else "Pending"
            st.markdown(f"{dot} &nbsp;{f.name} "
                        f"<span style='font-size:11px;color:{CHARCOAL}66;'>({label})</span>",
                        unsafe_allow_html=True)

    if st.session_state.verified:
        st.divider()
        rows = []
        for name, stamps in st.session_state.results.items():
            if name not in st.session_state.verified:
                continue
            for s in stamps:
                for d in s["dates"]:
                    rows.append({
                        "Document":   name,
                        "Stamp":      s["id"],
                        "Stamp Type": s["type"],
                        "Date Type":  d["type"],
                        "Date":       d["value"],
                        "Confidence": round(d["ocr_conf"], 3),
                    })
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button("Export Verified Records", csv, "verified_records.csv",
                           "text/csv", width='stretch', type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────
if not uploads:
    st.markdown("## Passport Stamp Extraction")
    st.markdown(f"<p style='color:{CHARCOAL}88;'>Upload passport page images from the sidebar to begin.</p>",
                unsafe_allow_html=True)
    st.stop()

tabs = st.tabs([f.name for f in uploads])

for tab, file in zip(tabs, uploads):
    with tab:
        fname = file.name

        if fname not in st.session_state.results:
            img_np = np.array(Image.open(file).convert("RGB"))
            with st.spinner("Processing document..."):
                # ── PIPELINE CALL ──────────────────────────────────────────
                stamps = pipeline.process(img_np)
                # ──────────────────────────────────────────────────────────
            st.session_state.results[fname]  = stamps
            st.session_state.selected[fname] = None

        stamps = st.session_state.results[fname]
        sel_id = st.session_state.selected.get(fname)
        img_np = np.array(Image.open(file).convert("RGB"))

        if not stamps:
            st.info("No stamps detected in this document.")
            continue

        col_left, col_right = st.columns([1.2, 1], gap="large")

        # ── LEFT: Document image ──────────────────────────────────────────
        with col_left:
            st.markdown('<p class="section-label">Document View</p>', unsafe_allow_html=True)

            if sel_id:
                sel = next(s for s in stamps if s["id"] == sel_id)
                c   = TYPE_COLOR.get(sel["type"], CHARCOAL)
                st.markdown(
                    f"<div style='border-left:3px solid {c};padding:6px 12px;"
                    f"background:{c}18;border-radius:0 4px 4px 0;margin-bottom:10px;'>"
                    f"<strong style='color:{c};'>{sel['id']} — {sel['type']} Stamp</strong>"
                    f"&nbsp;<span style='font-size:12px;color:{CHARCOAL}88;'>"
                    f"Detection {sel['det_conf']:.0%}</span></div>",
                    unsafe_allow_html=True
                )
                st.image(crop(img_np, sel["box"]), width='stretch')
                st.markdown(f"<p style='font-size:11px;color:{CHARCOAL}55;'>"
                            f"Full document context</p>", unsafe_allow_html=True)

            st.image(draw_boxes(img_np, stamps, sel_id), width='stretch')

        # ── RIGHT: Stamp cards + editor + timeline ────────────────────────
        with col_right:

            # Stamp selector
            for s in stamps:
                ocr_confs = [d["ocr_conf"] for d in s.get("dates", [])]
                s["max_ocr_conf"] = max(ocr_confs) if ocr_confs else 0.0

            stamps_sorted = sorted(stamps, key=lambda x: x["max_ocr_conf"], reverse=True)

            st.markdown('<p class="section-label">Detected Stamps (Sorted by OCR Confidence)</p>', unsafe_allow_html=True)
            STAMPS_PER_ROW = 4
            for i in range(0, len(stamps_sorted), STAMPS_PER_ROW):
                chunk = stamps_sorted[i:i + STAMPS_PER_ROW]
                cols = st.columns(STAMPS_PER_ROW)
                for col, s in zip(cols, chunk):
                    c      = TYPE_COLOR.get(s["type"], CHARCOAL)
                    is_sel = s["id"] == sel_id
                    with col:
                        st.markdown(
                            f"<div style='border:{'2px solid '+c if is_sel else '1px solid '+CHARCOAL+'22'};"
                            f"border-radius:6px;padding:10px 6px;background:{c+'18' if is_sel else WHITE};"
                            f"text-align:center;margin-bottom:8px;'>"
                            f"<span style='color:{c};font-weight:700;font-size:13px;'>{s['type']}</span><br>"
                            f"<span style='font-size:11px;color:{CHARCOAL}88;'>{s['id']}</span><br>"
                            # CHANGED: เปลี่ยนจาก s["det_conf"] เป็น s["max_ocr_conf"] เพื่อแสดงความแม่นยำของตัวอักษรแทน
                            + conf_bar(s["max_ocr_conf"], c) + 
                            f"</div>", unsafe_allow_html=True
                        )
                        if st.button("Select", key=f"sel_{fname}_{s['id']}", width='stretch'):
                            st.session_state.selected[fname] = s["id"]
                            st.rerun()

            st.divider()

            # Date editor
            if sel_id:
                sel = next(s for s in stamps if s["id"] == sel_id)
                st.markdown(f'<p class="section-label">Extracted Dates — {sel_id}</p>',
                            unsafe_allow_html=True)
                rows = []
                for d in sel["dates"]:
                    try:    parsed = datetime.strptime(d["value"], "%d %b %Y").date()
                    except: parsed = None
                    rows.append({
                        "Type":       d["type"],
                        "Date":       parsed,
                        "Confidence": round(d["ocr_conf"] * 100, 3),
                    })
                st.data_editor(
                    pd.DataFrame(rows),
                    num_rows="dynamic",
                    width='stretch',
                    hide_index=True,
                    column_config={
                        "Type": st.column_config.SelectboxColumn(
                            options=["Entry", "Exit", "Until", "Unknown"], required=True),
                        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                        "Confidence": st.column_config.NumberColumn(format="%.2f%%", disabled=True),
                    },
                    key=f"ed_{fname}_{sel_id}",
                )
            else:
                st.markdown(
                    f"<div style='padding:20px;background:{GREY};border-radius:6px;"
                    f"text-align:center;color:{CHARCOAL}66;font-size:13px;'>"
                    f"Select a stamp above to review its extracted dates.</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            # Date timeline
            st.markdown('<p class="section-label">Date Timeline</p>', unsafe_allow_html=True)
            all_dates = []
            for s in stamps:
                for d in s["dates"]:
                    # ignore Until in Date Timeline
                    if d["type"].lower() == "until":
                        continue
                    try:
                        all_dates.append({
                            **d,
                            "parsed": datetime.strptime(d["value"], "%d %b %Y"),
                            "conf":   round(d["ocr_conf"], 3),
                        })
                    except:
                        pass
            all_dates.sort(key=lambda x: x["parsed"])

            if all_dates:
                DATES_PER_ROW = 4
                
                for i in range(0, len(all_dates), DATES_PER_ROW):
                    chunk = all_dates[i:i + DATES_PER_ROW]
                    
                    # Always create 7 columns (4 dates + 3 arrows) to keep widths consistent across multiple rows
                    tl = st.columns(DATES_PER_ROW * 2 - 1)
                    
                    for j, d in enumerate(chunk):
                        c = TYPE_COLOR.get(d["type"], CHARCOAL)
                        with tl[j * 2]:
                            # Fixed a small CSS typo here as well (color:{WHITE}) and added bottom margin for row spacing
                            st.markdown(
                                f"<div style='background:{c};color:{WHITE};border-radius:6px;"
                                f"padding:10px 6px;text-align:center;margin-bottom:12px;'>"
                                f"<div style='font-size:10px;opacity:0.85;letter-spacing:0.5px;color:{WHITE}'>"
                                f"{d['type'].upper()}</div>"
                                f"<div style='font-weight:700;font-size:13px;margin:3px 0;color:{WHITE}'>"
                                f"{d['parsed'].strftime('%d %b %Y')}</div>"
                                f"<div style='font-size:10px;opacity:0.8;color:{WHITE}'>{d['conf']:.0%}</div>"
                                f"</div>", unsafe_allow_html=True
                            )
                        
                        # Add the arrow if it's not the last date in this specific chunk
                        if j < len(chunk) - 1:
                            with tl[j * 2 + 1]:
                                st.markdown(
                                    f"<div style='text-align:center;padding-top:18px;"
                                    f"color:{CHARCOAL}44;font-size:18px;'>&#8594;</div>",
                                    unsafe_allow_html=True
                                )
            else:
                st.markdown(f"<span style='color:{CHARCOAL}55;font-size:13px;'>"
                            f"No relevant entry/exit dates extracted.</span>", unsafe_allow_html=True)

        # ── Verify button ─────────────────────────────────────────────────
        st.divider()
        if fname not in st.session_state.verified:
            if st.button("Mark Document as Verified", type="primary",
                         key=f"verify_{fname}", width='stretch'):
                st.session_state.verified.add(fname)
                st.rerun()
        else:
            st.markdown(
                f"<div style='background:{RED}11;border:1px solid {RED}44;border-radius:6px;"
                f"padding:10px;text-align:center;color:{RED};font-weight:600;'>"
                f"Document Verified</div>", unsafe_allow_html=True
            )