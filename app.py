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
from datetime import datetime, date
from pipeline import DocumentPipeline

@st.cache_resource
def get_pipeline():
    return DocumentPipeline()

pipeline = get_pipeline()

# ── Qi Design Tokens  (source: Qi_Foundation_and_component_DESIGN.md) ───────
# Primary palette
RED      = "#d31145"   # accent-15  — Digital Red 500, Exit stamps
PURPLE   = "#4c4794"   # accent-117 — Digital Purple, Entry stamps
SALMON   = "#ff7a85"   # accent-7   — Digital Salmon, Until dates
CERISE   = "#ba0361"   # accent-55  — Interactive / highlight accent
# Neutrals
TEXT_PRI = "#14181c"   # text-primary
TEXT_SEC = "#333d47"   # text-secondary (Charcoal)
TEXT_TER = "#858b91"   # text-tertiary
BG       = "#ffffff"   # background
BG_ALT   = "#f5f5f6"   # background-alt
SURFACE  = "#ebeced"   # surface
BORDER   = "#adb1b5"   # border
# Radii (from design token frequency)
R_SM     = "8px"       # radius-sm-7  — small elements
R_MD     = "15px"      # radius-md    — cards, inputs, buttons (dominant)
R_PILL   = "999px"     # radius-lg    — badges, tags
# Shadows
ELEV_3   = "0px 1px 2px 0px rgba(0,0,0,0.15)"
ELEV_7   = "0px 2px 4px 0px rgba(0,0,0,0.08)"
ELEV_13  = "0px 2px 6px 0px rgba(0,0,0,0.06), 0px 5px 8px 0px rgba(0,0,0,0.04)"

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED, "Until": SALMON}

st.set_page_config(page_title="Passport Stamp Review", layout="wide")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');

  /* ── Base ── */
  .block-container {{ padding-top: 0 !important; padding-bottom: 2rem; max-width: 100% !important; }}
  html, body, [class*="css"] {{
    font-family: 'Open Sans', sans-serif;
    color: {TEXT_SEC};
    background: {BG};
  }}
  p, label, span {{ color: {TEXT_SEC}; line-height: 24px; }}
  h1, h2, h3 {{ color: {TEXT_PRI} !important; font-weight: 700; }}

  /* ── Top nav bar — AIA red stripe ── */
  .aia-topbar {{
    background: {RED};
    padding: 12px 24px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .aia-topbar-title {{
    color: {BG};
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.2px;
  }}
  .aia-topbar-sub {{
    color: rgba(255,255,255,0.75);
    font-size: 12px;
    font-weight: 400;
  }}

  /* ── Qi Card ── */
  .qi-card {{
    background: {BG};
    border: 1px solid {SURFACE};
    border-radius: {R_MD};
    padding: 16px;
    box-shadow: {ELEV_13};
    margin-bottom: 8px;
  }}
  .qi-card-selected {{
    background: {BG};
    border-radius: {R_MD};
    padding: 16px;
    box-shadow: {ELEV_13};
    margin-bottom: 8px;
  }}

  /* ── Qi Badge / pill ── */
  .qi-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: {R_PILL};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: {BG};
  }}

  /* ── Label overline ── */
  .qi-overline {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {TEXT_TER};
    margin-bottom: 8px;
  }}

  /* ── Buttons — Qi style ── */
  .stButton > button {{
    border-radius: {R_MD} !important;
    font-family: 'Open Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
    transition: background 0.15s, box-shadow 0.15s;
  }}
  button[kind="primary"] {{
    background: {RED} !important;
    border-color: {RED} !important;
    color: {BG} !important;
    box-shadow: {ELEV_7} !important;
  }}
  button[kind="primary"]:hover {{
    background: #b01039 !important;
    border-color: #b01039 !important;
    box-shadow: {ELEV_13} !important;
  }}
  button[kind="secondary"] {{
    background: {BG} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT_SEC} !important;
  }}
  button[kind="secondary"]:hover {{
    border-color: {RED} !important;
    color: {RED} !important;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: {BG} !important;
    border-right: 1px solid {SURFACE};
  }}
  [data-testid="stSidebar"] .stRadio label {{
    font-size: 13px !important;
    color: {TEXT_SEC} !important;
    padding: 6px 0;
  }}

  /* ── Divider ── */
  hr {{ border-color: {SURFACE} !important; margin: 16px 0 !important; }}

  /* ── Data editor ── */
  .stDataFrame, [data-testid="stDataEditorContainer"] {{
    border-radius: {R_MD} !important;
    border: 1px solid {SURFACE} !important;
    overflow: hidden;
  }}

  /* ── Info box ── */
  .stAlert {{ border-radius: {R_MD} !important; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def hex_rgba(h, a):
    r, g, b = to_rgb(h)
    return f"rgba({r},{g},{b},{a:.2f})"

def draw_boxes(img, stamps, sel_id):
    out = img.copy()
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        c   = to_rgb(TYPE_COLOR.get(s["type"], TEXT_SEC))
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

def to_date_str(val):
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
            val  = to_date_str(row.get("Date"))
            orig = next((d for d in s["dates"] if d["type"] == row.get("Type")), {})
            updated.append({"value": val, "type": row.get("Type","Unknown"),
                            "ocr_conf": orig.get("ocr_conf", 0.0)})
        s["dates"] = updated
        break


# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("results",{}), ("selected",{}), ("verified",set())]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Top nav bar ───────────────────────────────────────────────────────────────
st.markdown(
    f"<div class='aia-topbar'>"
    f"<div>"
    f"<div class='aia-topbar-title'>Passport Stamp Review</div>"
    f"<div class='aia-topbar-sub'>Entry / Exit Detection &amp; Verification</div>"
    f"</div>"
    f"</div>",
    unsafe_allow_html=True
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<p class='qi-overline' style='margin-top:8px;'>Workspace</p>",
                unsafe_allow_html=True)

    uploads = st.file_uploader("Upload passport images", type=["jpg","jpeg","png"],
                               accept_multiple_files=True, label_visibility="collapsed")

    if uploads:
        done  = len(st.session_state.verified)
        total = len(uploads)
        st.markdown(
            f"<div style='background:{BG_ALT};border-radius:{R_MD};padding:12px 14px;"
            f"margin-bottom:12px;box-shadow:{ELEV_3};'>"
            f"<div style='font-size:11px;color:{TEXT_TER};font-weight:600;"
            f"letter-spacing:0.8px;text-transform:uppercase;'>Progress</div>"
            f"<div style='font-size:20px;font-weight:700;color:{TEXT_PRI};'>"
            f"{done}<span style='font-size:13px;color:{TEXT_TER};font-weight:400;'> / {total}</span></div>"
            f"<div style='background:{SURFACE};border-radius:{R_PILL};height:4px;margin-top:6px;'>"
            f"<div style='width:{int(done/total*100) if total else 0}%;background:{RED};"
            f"height:4px;border-radius:{R_PILL};'></div></div>"
            f"</div>", unsafe_allow_html=True
        )

        st.markdown(f"<p class='qi-overline'>Documents</p>", unsafe_allow_html=True)
        file_names = [f.name for f in uploads]
        active_file = st.radio(
            "docs", options=file_names,
            format_func=lambda n: (f"✓  {n}" if n in st.session_state.verified else f"   {n}"),
            label_visibility="collapsed",
        )
    else:
        active_file = None


# ── Empty state ───────────────────────────────────────────────────────────────
if not uploads:
    st.markdown(
        f"<div style='margin:80px auto;max-width:480px;text-align:center;'>"
        f"<div style='font-size:32px;font-weight:700;color:{TEXT_PRI};margin-bottom:12px;'>"
        f"Upload a document to begin</div>"
        f"<p style='color:{TEXT_TER};font-size:15px;line-height:24px;'>"
        f"Select one or more passport images from the sidebar to start stamp detection and date verification.</p>"
        f"</div>", unsafe_allow_html=True
    )
    st.stop()

# ── All verified ──────────────────────────────────────────────────────────────
if len(st.session_state.verified) == len(uploads):
    st.markdown(
        f"<div style='margin:80px auto;max-width:480px;text-align:center;"
        f"background:{BG};border:1px solid {SURFACE};border-radius:{R_MD};"
        f"padding:40px 32px;box-shadow:{ELEV_13};'>"
        f"<div style='width:48px;height:48px;background:{hex_rgba(RED,0.1)};border-radius:{R_PILL};"
        f"margin:0 auto 16px;display:flex;align-items:center;justify-content:center;'>"
        f"<span style='color:{RED};font-size:22px;font-weight:700;'>✓</span></div>"
        f"<div style='font-size:22px;font-weight:700;color:{TEXT_PRI};'>All documents verified</div>"
        f"<p style='color:{TEXT_TER};margin-top:8px;'>{len(uploads)} document(s) reviewed successfully.</p>"
        f"</div>", unsafe_allow_html=True
    )
    st.stop()

# ── Load & process ────────────────────────────────────────────────────────────
file = next(f for f in uploads if f.name == active_file)
fname = file.name

if fname not in st.session_state.results:
    img_np = np.array(Image.open(file).convert("RGB"))
    with st.spinner("Detecting stamps..."):
        stamps = pipeline.process(img_np)
    st.session_state.results[fname]  = stamps
    st.session_state.selected[fname] = None

stamps = st.session_state.results[fname]
sel_id = st.session_state.selected.get(fname)
img_np = np.array(Image.open(file).convert("RGB"))

# ── Document header ───────────────────────────────────────────────────────────
n_entry   = sum(1 for s in stamps if s["type"] == "Entry")
n_exit    = sum(1 for s in stamps if s["type"] == "Exit")
n_unknown = sum(1 for s in stamps if primary_date(s)[0] is None)
is_verified = fname in st.session_state.verified

header_right = ""
if is_verified:
    header_right = (f"<span style='background:{hex_rgba(RED,0.08)};color:{RED};"
                    f"border-radius:{R_PILL};padding:3px 12px;font-size:12px;font-weight:700;'>"
                    f"Verified</span>")

badges = (
    f"<span class='qi-badge' style='background:{PURPLE};'>{n_entry} Entry</span> "
    f"<span class='qi-badge' style='background:{RED};'>{n_exit} Exit</span>"
    + (f" <span class='qi-badge' style='background:{TEXT_TER};'>{n_unknown} Unknown</span>"
       if n_unknown else "")
)

st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center;"
    f"margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid {SURFACE};'>"
    f"<div>"
    f"<div style='font-size:18px;font-weight:700;color:{TEXT_PRI};'>{fname}</div>"
    f"<div style='margin-top:6px;display:flex;gap:6px;'>{badges}</div>"
    f"</div>{header_right}</div>", unsafe_allow_html=True
)

if not stamps:
    st.info("No stamps detected in this document.")
    st.stop()

col_left, col_right = st.columns([1.2, 1], gap="large")

# ── LEFT: Document view ───────────────────────────────────────────────────────
with col_left:
    st.markdown(f'<p class="qi-overline">Document View</p>', unsafe_allow_html=True)

    if sel_id:
        sel = next(s for s in stamps if s["id"] == sel_id)
        c   = TYPE_COLOR.get(sel["type"], TEXT_SEC)
        st.markdown(
            f"<div style='border-left:3px solid {c};padding:8px 14px;"
            f"background:{hex_rgba(c,0.06)};border-radius:0 {R_SM} {R_SM} 0;"
            f"margin-bottom:10px;box-shadow:{ELEV_3};'>"
            f"<div style='font-weight:700;color:{c};font-size:14px;'>"
            f"{sel['id']} &mdash; {sel['type']} Stamp</div>"
            f"<div style='font-size:12px;color:{TEXT_TER};margin-top:2px;'>"
            f"Detection confidence: {sel['det_conf']:.0%}</div></div>",
            unsafe_allow_html=True
        )
        st.image(stamp_crop(img_np, sel["box"]), width="stretch")
        st.markdown(
            f"<p style='font-size:11px;color:{TEXT_TER};margin:4px 0 8px;'>"
            f"Full document context below</p>", unsafe_allow_html=True
        )

    st.image(draw_boxes(img_np, stamps, sel_id), width="stretch")


# ── RIGHT: Timeline + editor ──────────────────────────────────────────────────
with col_right:
    st.markdown(f'<p class="qi-overline">Timeline</p>', unsafe_allow_html=True)

    sorted_stamps = sorted(stamps, key=sort_key)

    for i, s in enumerate(sorted_stamps):
        date_val, date_type = primary_date(s)
        c          = TYPE_COLOR.get(s["type"], TEXT_SEC)
        conf       = max((d.get("ocr_conf",0) for d in s.get("dates",[])), default=0)
        is_sel     = s["id"] == sel_id
        is_unknown = date_val is None
        conf_pct   = int(conf * 100)

        # Arrow connector between consecutive dated entries
        if i > 0:
            prev_val = primary_date(sorted_stamps[i-1])[0]
            if date_val and prev_val:
                st.markdown(
                    f"<div style='text-align:center;color:{BORDER};font-size:14px;"
                    f"margin:2px 0;line-height:1;'>&#8595;</div>",
                    unsafe_allow_html=True
                )

        # Card border & background
        if is_unknown:
            border = f"1.5px dashed {BORDER}"
            bg     = BG_ALT
            shadow = ELEV_3
        elif is_sel:
            border = f"2px solid {c}"
            bg     = hex_rgba(c, 0.06)
            shadow = ELEV_13
        else:
            border = f"1px solid {hex_rgba(c,0.25)}"
            bg     = BG
            shadow = ELEV_7

        badge_bg = c if not is_unknown else TEXT_TER
        conf_bar_html = (
            f"<div style='background:{SURFACE};border-radius:{R_PILL};height:3px;margin-top:8px;'>"
            f"<div style='width:{conf_pct}%;background:{c};height:3px;border-radius:{R_PILL};'>"
            f"</div></div>"
            f"<span style='font-size:11px;color:{TEXT_TER};'>{conf_pct}% OCR confidence</span>"
        ) if not is_unknown else (
            f"<span style='font-size:12px;color:{TEXT_TER};'>No date detected</span>"
        )

        st.markdown(
            f"<div style='border:{border};border-radius:{R_MD};padding:12px 16px;"
            f"background:{bg};box-shadow:{shadow};margin-bottom:4px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-start;'>"
            f"<div>"
            f"<span class='qi-badge' style='background:{badge_bg};'>{s['type']}</span>"
            f"</div>"
            f"<span style='font-size:11px;color:{TEXT_TER};'>{s['id']}</span>"
            f"</div>"
            f"<div style='font-size:18px;font-weight:700;color:{TEXT_PRI};margin:8px 0 2px;'>"
            f"{date_val if date_val else '&mdash;'}</div>"
            + conf_bar_html +
            f"</div>", unsafe_allow_html=True
        )

        btn_label = "Add Date" if is_unknown else "Edit"
        btn_type  = "primary" if is_unknown else "secondary"
        if st.button(btn_label, key=f"sel_{fname}_{s['id']}",
                     width="stretch", type=btn_type):
            st.session_state.selected[fname] = s["id"]

    st.divider()

    # ── Editor ────────────────────────────────────────────────────────────
    if sel_id:
        sel = next(s for s in stamps if s["id"] == sel_id)
        c   = TYPE_COLOR.get(sel["type"], TEXT_SEC)
        st.markdown(
            f"<div style='border-left:3px solid {c};padding:4px 12px;margin-bottom:10px;'>"
            f"<span style='font-weight:700;color:{c};font-size:13px;'>Editing {sel_id}</span>"
            f"</div>", unsafe_allow_html=True
        )
        rows = []
        for d in sel.get("dates", []):
            try:    parsed = datetime.strptime(d["value"], "%d %b %Y").date() if d.get("value") else None
            except: parsed = None
            rows.append({"Type": d["type"], "Date": parsed,
                         "Confidence": round(d.get("ocr_conf",0)*100, 1)})

        edited = st.data_editor(
            pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Type","Date","Confidence"]),
            num_rows="dynamic", width="stretch", hide_index=True,
            column_config={
                "Type": st.column_config.SelectboxColumn(
                    options=["Entry","Exit","Until","Unknown"], required=True),
                "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "Confidence": st.column_config.NumberColumn(format="%.1f%%", disabled=True),
            },
            key=f"ed_{fname}_{sel_id}",
        )
        sync_edits(edited, fname, sel_id)
    else:
        st.markdown(
            f"<div style='padding:20px 16px;background:{BG_ALT};border-radius:{R_MD};"
            f"text-align:center;border:1.5px dashed {BORDER};'>"
            f"<p style='color:{TEXT_TER};font-size:13px;margin:0;'>"
            f"Select a stamp from the timeline to review or edit its date.</p>"
            f"</div>", unsafe_allow_html=True
        )

# ── Verify ────────────────────────────────────────────────────────────────────
st.divider()
if fname not in st.session_state.verified:
    if st.button("Mark as Verified", type="primary",
                 key=f"verify_{fname}", width="stretch"):
        st.session_state.verified.add(fname)
        st.rerun()