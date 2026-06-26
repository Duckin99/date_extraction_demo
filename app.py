"""
demo_app.py — Passport Stamp Extraction Demo
Run:  streamlit run demo_app.py
See README.md for pipeline schema.
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
RED      = "#D31145"
CHARCOAL = "#333D47"
PURPLE   = "#4C4794"
SALMON   = "#FF7A85"
WHITE    = "#FFFFFF"
GREY     = "#F4F4F4"
GREY_BD  = "#E0E0E0"

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED, "Until": SALMON, "Unknown": CHARCOAL}

st.set_page_config(page_title="Passport Stamp Review", layout="wide")
st.markdown(f"""<style>
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
  p, label, span {{ color: {CHARCOAL}; font-family: 'Inter', sans-serif; }}
  h1, h2, h3 {{ color: {CHARCOAL} !important; }}
  .stButton > button {{ border-radius: 4px; font-weight: 600; }}
  .label {{
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; color: {CHARCOAL}88; margin-bottom: 6px;
  }}
</style>""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return (b, g, r)

def draw_boxes(img, stamps, sel_id):
    out = img.copy()
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        c   = to_bgr(TYPE_COLOR.get(s["type"], CHARCOAL))
        sel = s["id"] == sel_id
        ov  = out.copy()
        cv2.rectangle(ov, (x1,y1), (x2,y2), c, 4 if sel else 2)
        cv2.addWeighted(ov, 1.0 if sel else 0.55, out, 0.0 if sel else 0.45, 0, out)
        lbl = f"{s['type']}  {s['det_conf']:.0%}"
        (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out,(x1,y1-th-9),(x1+tw+8,y1),c,-1)
        cv2.putText(out,lbl,(x1+4,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    return out

def stamp_crop(img, box, pad=16):
    h,w = img.shape[:2]
    x1,y1,x2,y2 = box
    return img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]

def hex_rgba(h, alpha):
    h = h.lstrip("#")
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def sort_key(s):
    """Sort stamps by earliest entry/exit date; unknowns go last."""
    dates = [d for d in s.get("dates",[]) if d.get("value") and d["type"] != "Until"]
    if not dates:
        return datetime.max
    try:
        return min(datetime.strptime(d["value"], "%d %b %Y") for d in dates)
    except:
        return datetime.max

def primary_date(s):
    """Return the most relevant date string and type for a stamp."""
    dates = [d for d in s.get("dates",[]) if d.get("value") and d["type"] in ("Entry","Exit")]
    if not dates:
        return None, "Unknown"
    best = max(dates, key=lambda d: d.get("ocr_conf", 0))
    return best["value"], best["type"]

def sync_edits(edited_df, fname, sel_id):
    """Write data_editor changes back into session state."""
    for s in st.session_state.results[fname]:
        if s["id"] != sel_id:
            continue
        updated = []
        for _, row in edited_df.iterrows():
            raw_date = row.get("Date")
            val = raw_date.strftime("%d %b %Y") if raw_date else None
            # Preserve original ocr_conf if available
            orig = next((d for d in s["dates"] if d["type"] == row["Type"]), {})
            updated.append({
                "value":    val,
                "type":     row["Type"],
                "ocr_conf": orig.get("ocr_conf", 0.0),
            })
        s["dates"] = updated
        break


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("results",{}), ("selected",{}), ("verified",set())]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h3 style='color:{RED};margin-bottom:0;'>Stamp Review</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:12px;color:{CHARCOAL}88;margin-top:2px;'>Passport Entry / Exit Detection</p>", unsafe_allow_html=True)
    st.divider()

    uploads = st.file_uploader("Upload passport images", type=["jpg","jpeg","png"],
                               accept_multiple_files=True)
    if uploads:
        done = len(st.session_state.verified)
        st.markdown(f"**Progress: {done} / {len(uploads)} verified**")
        st.progress(done / len(uploads))
        st.divider()
        for f in uploads:
            is_done = f.name in st.session_state.verified
            dot = f"<span style='color:{RED};'>&#9679;</span>" if is_done else \
                  f"<span style='color:{CHARCOAL}33;'>&#9675;</span>"
            st.markdown(f"{dot} &nbsp;{f.name} <span style='font-size:11px;color:{CHARCOAL}66;'>"
                        f"({'Verified' if is_done else 'Pending'})</span>", unsafe_allow_html=True)

    if st.session_state.verified:
        st.divider()
        rows = []
        for name, stamps in st.session_state.results.items():
            if name not in st.session_state.verified:
                continue
            for s in stamps:
                for d in s["dates"]:
                    rows.append({"Document":name,"Stamp":s["id"],"Stamp Type":s["type"],
                                 "Date Type":d["type"],"Date":d["value"],
                                 "Confidence":round(d["ocr_conf"],3)})
        csv = pd.DataFrame(rows).to_csv(index=False)
        st.download_button("Export Verified Records", csv, "verified_records.csv",
                           "text/csv", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────
if not uploads:
    st.markdown("## Passport Stamp Extraction")
    st.markdown(f"<p style='color:{CHARCOAL}88;'>Upload passport images from the sidebar to begin.</p>",
                unsafe_allow_html=True)
    st.stop()

tabs = st.tabs([f.name for f in uploads])

for tab, file in zip(tabs, uploads):
    with tab:
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

        if not stamps:
            st.info("No stamps detected in this document.")
            continue

        # ── Summary banner ────────────────────────────────────────────────
        n_entry   = sum(1 for s in stamps if s["type"] == "Entry")
        n_exit    = sum(1 for s in stamps if s["type"] == "Exit")
        n_unknown = sum(1 for s in stamps if not any(
            d.get("value") and d["type"] in ("Entry","Exit") for d in s.get("dates",[])))

        st.markdown(
            f"<div style='background:{GREY};border-radius:6px;padding:10px 16px;"
            f"display:flex;gap:20px;align-items:center;margin-bottom:12px;'>"
            f"<span style='font-size:12px;color:{CHARCOAL}88;font-weight:600;'>DETECTED</span>"
            f"<span style='background:{PURPLE};color:{WHITE};padding:2px 10px;"
            f"border-radius:3px;font-size:12px;font-weight:700;'>{n_entry} Entry</span>"
            f"<span style='background:{RED};color:{WHITE};padding:2px 10px;"
            f"border-radius:3px;font-size:12px;font-weight:700;'>{n_exit} Exit</span>"
            + (f"<span style='background:{CHARCOAL}44;color:{WHITE};padding:2px 10px;"
               f"border-radius:3px;font-size:12px;font-weight:700;'>{n_unknown} Unknown</span>"
               if n_unknown else "") +
            f"</div>", unsafe_allow_html=True
        )

        col_left, col_right = st.columns([1.2, 1], gap="large")

        # ── LEFT: Image ───────────────────────────────────────────────────
        with col_left:
            st.markdown(f'<p class="label">Document View</p>', unsafe_allow_html=True)

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
                st.image(stamp_crop(img_np, sel["box"]), use_container_width=True)
                st.markdown(f"<p style='font-size:11px;color:{CHARCOAL}44;margin-bottom:6px;'>"
                            f"Full document context</p>", unsafe_allow_html=True)

            st.image(draw_boxes(img_np, stamps, sel_id), use_container_width=True)

        # ── RIGHT: Timeline + editor ──────────────────────────────────────
        with col_right:
            st.markdown(f'<p class="label">Timeline</p>', unsafe_allow_html=True)

            sorted_stamps = sorted(stamps, key=sort_key)

            for i, s in enumerate(sorted_stamps):
                date_val, date_type = primary_date(s)
                c         = TYPE_COLOR.get(s["type"], CHARCOAL)
                conf      = max((d.get("ocr_conf",0) for d in s.get("dates",[])), default=0)
                is_sel    = s["id"] == sel_id
                is_unknown = date_val is None

                # Card background: tint intensity = confidence
                # Unknown: grey with dashed border
                if is_unknown:
                    border = f"1.5px dashed {GREY_BD}"
                    bg     = GREY
                elif is_sel:
                    border = f"2px solid {c}"
                    bg     = hex_rgba(c, 0.12)
                else:
                    border = f"1px solid {hex_rgba(c, 0.3)}"
                    bg     = hex_rgba(c, max(0.05, conf * 0.15))

                # Arrow between known dates
                if i > 0 and not is_unknown and not (primary_date(sorted_stamps[i-1])[0] is None):
                    st.markdown(
                        f"<div style='text-align:center;color:{CHARCOAL}33;font-size:16px;"
                        f"margin:2px 0;'>&#8595;</div>", unsafe_allow_html=True
                    )

                # Timeline card
                conf_pct = int(conf * 100)
                conf_bar = (
                    f"<div style='background:{GREY_BD};border-radius:3px;height:4px;margin-top:6px;'>"
                    f"<div style='width:{conf_pct}%;background:{c if not is_unknown else CHARCOAL+'66'};"
                    f"height:4px;border-radius:3px;'></div></div>"
                    f"<span style='font-size:11px;color:{CHARCOAL}88;'>{conf_pct}% confidence</span>"
                ) if not is_unknown else (
                    f"<span style='font-size:11px;color:{CHARCOAL}55;'>No date detected</span>"
                )

                st.markdown(
                    f"<div style='border:{border};border-radius:8px;padding:10px 14px;"
                    f"background:{bg};margin-bottom:4px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='background:{c if not is_unknown else CHARCOAL+'55'};"
                    f"color:{WHITE};font-size:10px;font-weight:700;letter-spacing:0.5px;"
                    f"padding:2px 8px;border-radius:3px;'>{s['type'].upper()}</span>"
                    f"<span style='font-size:11px;color:{CHARCOAL}66;'>{s['id']}</span></div>"
                    f"<div style='font-size:16px;font-weight:700;color:{CHARCOAL};"
                    f"margin:4px 0;'>{date_val if date_val else '—'}</div>"
                    + conf_bar +
                    f"</div>", unsafe_allow_html=True
                )

                if st.button(
                    "Edit" if not is_unknown else "Add Date",
                    key=f"sel_{fname}_{s['id']}",
                    use_container_width=True,
                    type="primary" if is_unknown else "secondary"
                ):
                    st.session_state.selected[fname] = s["id"]
                    st.rerun()

            st.divider()

            # ── Date editor ───────────────────────────────────────────────
            if sel_id:
                sel = next(s for s in stamps if s["id"] == sel_id)
                c   = TYPE_COLOR.get(sel["type"], CHARCOAL)
                st.markdown(
                    f"<div style='border-left:3px solid {c};padding:4px 10px;"
                    f"margin-bottom:8px;'><strong style='color:{c};font-size:13px;'>"
                    f"Editing {sel_id}</strong></div>", unsafe_allow_html=True
                )

                rows = []
                for d in sel["dates"]:
                    try:    parsed = datetime.strptime(d["value"], "%d %b %Y").date() if d.get("value") else None
                    except: parsed = None
                    rows.append({"Type":d["type"],"Date":parsed,
                                 "Confidence":round(d.get("ocr_conf",0)*100, 1)})

                edited = st.data_editor(
                    pd.DataFrame(rows) if rows else pd.DataFrame(
                        columns=["Type","Date","Confidence"]),
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Type": st.column_config.SelectboxColumn(
                            options=["Entry","Exit","Until","Unknown"], required=True),
                        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                        "Confidence": st.column_config.NumberColumn(format="%.1f%%", disabled=True),
                    },
                    key=f"ed_{fname}_{sel_id}",
                )
                # Sync edits back so timeline updates on next interaction
                sync_edits(edited, fname, sel_id)
            else:
                st.markdown(
                    f"<div style='padding:16px;background:{GREY};border-radius:6px;"
                    f"text-align:center;color:{CHARCOAL}66;font-size:13px;border:1px dashed {GREY_BD};'>"
                    f"Select a stamp from the timeline to review or edit its date.</div>",
                    unsafe_allow_html=True
                )

        # ── Verify ────────────────────────────────────────────────────────
        st.divider()
        if fname not in st.session_state.verified:
            if st.button("Mark Document as Verified", type="primary",
                         key=f"verify_{fname}", use_container_width=True):
                st.session_state.verified.add(fname)
                st.rerun()
        else:
            st.markdown(
                f"<div style='background:{hex_rgba(RED,0.06)};border:1px solid {hex_rgba(RED,0.3)};"
                f"border-radius:6px;padding:10px;text-align:center;"
                f"color:{RED};font-weight:600;'>Document Verified</div>",
                unsafe_allow_html=True
            )