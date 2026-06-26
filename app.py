"""
demo_app.py — Passport Stamp Extraction Demo
Run:  streamlit run demo_app.py
"""

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime, date
from pipeline import DocumentPipeline

@st.cache_resource
def get_pipeline():
    return DocumentPipeline()

pipeline = get_pipeline()

# ── AIA Colour Palette  (https://design.aia.com/colour) ────────────────────
RED      = "#D31145"
CHARCOAL = "#333D47"
PURPLE   = "#4C4794"
SALMON   = "#FF7A85"
WHITE    = "#FFFFFF"
GREY     = "#F4F4F4"
GREY_BD  = "#E0E0E0"

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED, "Until": SALMON}

st.set_page_config(page_title="Passport Stamp Review", layout="wide")
st.markdown(f"""<style>
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
  p, label, span {{ color: {CHARCOAL}; font-family: 'Inter', sans-serif; }}
  h1, h2, h3 {{ color: {CHARCOAL} !important; }}
  .label {{
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: {CHARCOAL}88; margin-bottom: 6px;
  }}
  /* Force White & Bold text on primary buttons */
  button[kind="primary"] p {{
    color: {WHITE} !important;
    font-weight: 700 !important;
  }}
  button[kind="primary"] {{
    background-color: {RED} !important;
    border-color: {RED} !important;
  }}
  button[kind="primary"]:hover {{
    background-color: #B01039 !important;
    border-color: #B01039 !important;
  }}
</style>""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def hex_rgba(h, alpha):
    r, g, b = to_rgb(h)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def draw_boxes(img, stamps, sel_id):
    out = img.copy()
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        c   = to_rgb(TYPE_COLOR.get(s["type"], CHARCOAL))
        sel = s["id"] == sel_id
        ov  = out.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), c, 4 if sel else 2)
        cv2.addWeighted(ov, 1.0 if sel else 0.5, out, 0.0 if sel else 0.5, 0, out)
        lbl = f"{s['type']}  {s['det_conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-9), (x1+tw+8, y1), c, -1)
        cv2.putText(out, lbl, (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def stamp_crop(img, box, pad=16):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    return img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

def sort_key(s):
    dates = [d for d in s.get("dates",[]) if d.get("value") and d["type"] in ("Entry","Exit")]
    if not dates:
        return datetime.max
    try:
        return min(datetime.strptime(d["value"], "%d %b %Y") for d in dates)
    except:
        return datetime.max

def primary_date(s):
    dates = [d for d in s.get("dates",[]) if d.get("value") and d["type"] in ("Entry","Exit")]
    if not dates:
        return None, None
    best = max(dates, key=lambda d: d.get("ocr_conf", 0))
    return best["value"], best["type"]

def to_date(val):
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.strftime("%d %b %Y")
    if isinstance(val, str) and val:
        return val
    return None

def sync_edits(edited_df, fname, sel_id):
    for s in st.session_state.results[fname]:
        if s["id"] != sel_id:
            continue
        updated = []
        for _, row in edited_df.iterrows():
            val  = to_date(row.get("Date"))
            orig = next((d for d in s["dates"] if d["type"] == row.get("Type")), {})
            updated.append({
                "value":    val,
                "type":     row.get("Type", "Unknown"),
                "ocr_conf": orig.get("ocr_conf", 1.0 if val else 0.0),
            })
        if s["dates"] != updated:
            s["dates"] = updated
            return True
    return False


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("results",{}), ("selected",{}), ("verified",set())]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{RED};margin-bottom:0;'>Stamp Review</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:12px;color:{CHARCOAL}88;margin-top:2px;'>"
                f"Passport Entry / Exit Detection</p>", unsafe_allow_html=True)
    st.divider()

    uploads = st.file_uploader("Upload passport images", type=["jpg","jpeg","png"],
                               accept_multiple_files=True)

    if uploads:
        done  = len(st.session_state.verified)
        total = len(uploads)
        st.markdown(f"**Progress: {done} / {total} verified**")
        st.progress(done / total)
        st.divider()

        file_names = [f.name for f in uploads]
        active_file = st.radio(
            "Documents",
            options=file_names,
            format_func=lambda n: (
                f"[Done] {n}" if n in st.session_state.verified else f"[Pending] {n}"
            ),
            label_visibility="collapsed",
        )


# ── Main ──────────────────────────────────────────────────────────────────────
if not uploads:
    st.markdown("## Passport Stamp Extraction")
    st.markdown(f"<p style='color:{CHARCOAL}88;'>Upload passport images from the sidebar to begin.</p>",
                unsafe_allow_html=True)
    st.stop()

if len(st.session_state.verified) == len(uploads):
    st.markdown(
        f"<div style='background:{hex_rgba(RED,0.06)};border:1px solid {hex_rgba(RED,0.3)};"
        f"border-radius:8px;padding:24px;text-align:center;margin-top:40px;'>"
        f"<div style='font-size:20px;font-weight:700;color:{RED};'>All Documents Verified</div>"
        f"<div style='font-size:13px;color:{CHARCOAL}88;margin-top:6px;'>"
        f"{len(uploads)} document(s) reviewed successfully.</div>"
        f"</div>", unsafe_allow_html=True
    )
    st.stop()

file = next(f for f in uploads if f.name == active_file)
fname = file.name

if fname not in st.session_state.results:
    img_np = np.array(Image.open(file).convert("RGB"))
    with st.spinner("Processing document..."):
        stamps = pipeline.process(img_np)
    st.session_state.results[fname]  = stamps
    st.session_state.selected[fname] = None

stamps = st.session_state.results[fname]
sel_id = st.session_state.selected.get(fname)
img_np = np.array(Image.open(file).convert("RGB"))

st.markdown(
    f"<div style='margin-bottom:12px;'>"
    f"<span style='font-size:18px;font-weight:700;color:{CHARCOAL};'>{fname}</span>"
    + (f"&nbsp;<span style='background:{hex_rgba(RED,0.1)};color:{RED};font-size:12px;"
       f"font-weight:600;padding:2px 8px;border-radius:3px;'>Verified</span>"
       if fname in st.session_state.verified else "") +
    f"</div>", unsafe_allow_html=True
)

if not stamps:
    st.info("No stamps detected in this document.")
    st.stop()

n_entry = sum(1 for s in stamps if s["type"] == "Entry")
n_exit  = sum(1 for s in stamps if s["type"] == "Exit")
st.markdown(
    f"<div style='background:{GREY};border-radius:6px;padding:8px 16px;"
    f"display:flex;gap:12px;align-items:center;margin-bottom:14px;'>"
    f"<span style='font-size:11px;font-weight:700;letter-spacing:1px;color:{CHARCOAL}66;'>DETECTED</span>"
    f"<span style='background:{PURPLE};color:{WHITE};padding:2px 10px;"
    f"border-radius:3px;font-size:12px;font-weight:700;'>{n_entry} Entry</span>"
    f"<span style='background:{RED};color:{WHITE};padding:2px 10px;"
    f"border-radius:3px;font-size:12px;font-weight:700;'>{n_exit} Exit</span>"
    f"</div>", unsafe_allow_html=True
)

col_left, col_right = st.columns([1.2, 1], gap="large")

# ── LEFT: Document image ──────────────────────────────────────────────────────
with col_left:
    st.markdown('<p class="label">Document View</p>', unsafe_allow_html=True)

    if sel_id:
        sel = next(s for s in stamps if s["id"] == sel_id)
        c   = TYPE_COLOR.get(sel["type"], CHARCOAL)
        st.markdown(
            f"<div style='border-left:3px solid {c};padding:6px 12px;"
            f"background:{hex_rgba(c,0.08)};border-radius:0 4px 4px 0;margin-bottom:8px;'>"
            f"<strong style='color:{c};'>{sel['id']} — {sel['type']}</strong>"
            f"&nbsp;<span style='font-size:12px;color:{CHARCOAL}88;'>"
            f"Detection {sel['det_conf']:.0%}</span></div>",
            unsafe_allow_html=True
        )
        st.image(stamp_crop(img_np, sel["box"]), width="stretch")
        st.markdown(f"<p style='font-size:11px;color:{CHARCOAL}44;margin-bottom:6px;'>"
                    f"Full document context</p>", unsafe_allow_html=True)

    st.image(draw_boxes(img_np, stamps, sel_id), width="stretch")

# ── RIGHT: Editor (Top) + Timeline Grid (Bottom) ──────────────────────────────
with col_right:
    
    if sel_id:
        sel = next(s for s in stamps if s["id"] == sel_id)
        c   = TYPE_COLOR.get(sel["type"], CHARCOAL)
        st.markdown(
            f"<div style='border-left:3px solid {c};padding:4px 10px;margin-bottom:8px;'>"
            f"<strong style='color:{c};font-size:13px;'>Editing Metadata: {sel_id}</strong></div>",
            unsafe_allow_html=True
        )
        rows = []
        for d in sel.get("dates", []):
            try:    parsed = datetime.strptime(d["value"], "%d %b %Y").date() if d.get("value") else None
            except: parsed = None
            rows.append({"Type": d["type"], "Date": parsed,
                         "Confidence": round(d.get("ocr_conf",0)*100, 1)})

        edited = st.data_editor(
            pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type","Date","Confidence"]),
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    options=["Entry","Exit","Until","Unknown"], required=True),
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Confidence": st.column_config.NumberColumn(format="%.1f%%", disabled=True),
            },
            key=f"ed_{fname}_{sel_id}",
        )
        if sync_edits(edited, fname, sel_id):
            st.rerun()
    else:
        st.markdown(
            f"<div style='padding:16px;background:{GREY};border-radius:6px;"
            f"text-align:center;color:{CHARCOAL}66;font-size:13px;"
            f"border:1px dashed {GREY_BD};margin-bottom:12px;'>"
            f"Select a stamp below from the queue to view or edit details.</div>",
            unsafe_allow_html=True
        )

    st.divider()

    # 1 & 2. FIX: Removed Scrollable Container and implemented Grid layout (2 per line)
    st.markdown('<p class="label">Stamp Queue</p>', unsafe_allow_html=True)
    
    stamps_sorted = sorted(stamps, key=sort_key)
    
    # Grid chunking: 2 cards per row
    CARDS_PER_ROW = 2
    for i in range(0, len(stamps_sorted), CARDS_PER_ROW):
        chunk = stamps_sorted[i:i + CARDS_PER_ROW]
        cols = st.columns(CARDS_PER_ROW)
        
        for col, s in zip(cols, chunk):
            with col:
                date_val, date_type = primary_date(s)
                c          = TYPE_COLOR.get(s["type"], CHARCOAL)
                conf       = max((d.get("ocr_conf",0) for d in s.get("dates",[])), default=0)
                is_sel     = s["id"] == sel_id
                is_unknown = date_val is None

                # 3. FIX: Background strictly tied to confidence score, selection uses bold 3px border
                if is_unknown:
                    border = f"1.5px dashed {GREY_BD}"
                    bg     = GREY
                else:
                    border = f"3px solid {c}" if is_sel else f"1px solid {hex_rgba(c, 0.3)}"
                    bg     = hex_rgba(c, max(0.05, conf * 0.15)) # Background never gets overwritten

                conf_pct = int(conf * 100)
                conf_html = (
                    f"<div style='background:{GREY_BD};border-radius:3px;height:4px;margin-top:6px;'>"
                    f"<div style='width:{conf_pct}%;background:{c};height:4px;border-radius:3px;'></div></div>"
                    f"<span style='font-size:11px;color:{CHARCOAL}88;'>{conf_pct}% OCR confidence</span>"
                ) if not is_unknown else (
                    f"<span style='font-size:11px;color:{CHARCOAL}55;'>No date found</span>"
                )

                badge_bg = c if not is_unknown else f"{CHARCOAL}55"
                st.markdown(
                    f"<div style='border:{border};border-radius:8px;padding:10px 14px;"
                    f"background:{bg};margin-bottom:8px;'>" # Replaced vertical arrows with grid margins
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='background:{badge_bg};color:{WHITE};font-size:10px;font-weight:700;"
                    f"letter-spacing:0.5px;padding:2px 8px;border-radius:3px;'>{s['type'].upper()}</span>"
                    f"<span style='font-size:11px;color:{CHARCOAL}66;'>{s['id']}</span></div>"
                    f"<div style='font-size:16px;font-weight:700;color:{CHARCOAL};margin:4px 0;'>"
                    f"{date_val if date_val else '—'}</div>"
                    + conf_html +
                    f"</div>", unsafe_allow_html=True
                )
                
                if st.button(
                    "Select",
                    key=f"sel_{fname}_{s['id']}",
                    width="stretch",
                    type="primary" if is_sel else "secondary"
                ):
                    st.session_state.selected[fname] = s["id"]
                    st.rerun()

# ── Verify ────────────────────────────────────────────────────────────────────
st.divider()
if fname not in st.session_state.verified:
    if st.button("Mark Document as Verified", type="primary",
                 key=f"verify_{fname}", width="stretch"):
        st.session_state.verified.add(fname)
        st.rerun()