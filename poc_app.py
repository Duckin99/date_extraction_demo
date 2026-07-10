"""
app.py — AIA Immigration Stamp Review
Run: python app.py   →  http://localhost:8080

Underwriter workflow this app automates:
  Manually scanning each passport page for entry/exit stamps and building
  a chronological travel timeline, one stamp at a time.

This app runs detection + OCR automatically and gives the underwriter a
single prioritized view: which stamps to trust, which to double-check.

Pipeline integration
--------------------
from pipeline import DocumentPipeline
Expected stamps schema — see README.md.
"""

import io
from datetime import datetime, date

import numpy as np
from PIL import Image
from nicegui import ui, events

from pipeline import DocumentPipeline

pipeline = DocumentPipeline()

# ── Qi Design Tokens ──────────────────────────────────────────────────────
RED, PURPLE, SALMON = "#d31145", "#4c4794", "#ff7a85"
TEXT_PRI, TEXT_SEC, TEXT_TER = "#14181c", "#333d47", "#858b91"
BG, BG_ALT, SURFACE, BORDER = "#ffffff", "#f5f5f6", "#ebeced", "#adb1b5"
R_MD, R_PILL = "15px", "999px"
ELEV_7  = "0px 2px 4px rgba(0,0,0,0.08)"
ELEV_13 = "0px 2px 6px rgba(0,0,0,0.06), 0px 5px 8px rgba(0,0,0,0.04)"

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED}
REVIEW_THRESHOLD = 0.75  # cumulative confidence below this is flagged


# ── Domain helpers ──────────────────────────────────────────────────────────
def cumulative_conf(s):
    """det_conf x best ocr_conf — the combined trust score for a stamp."""
    dates = s.get("dates", [])
    ocr = max((d.get("ocr_conf", 0) for d in dates), default=0)
    return round(s.get("det_conf", 0) * (ocr if dates else 1), 3)

def primary_date(s):
    dates = [d for d in s.get("dates", []) if d.get("value") and d["type"] in ("Entry", "Exit")]
    if not dates:
        return None, None
    best = max(dates, key=lambda d: d.get("ocr_conf", 0))
    return best["value"], best["type"]

def needs_review(s):
    val, _ = primary_date(s)
    return val is None or cumulative_conf(s) < REVIEW_THRESHOLD

def chrono_key(s):
    val, _ = primary_date(s)
    if not val:
        return datetime.max
    try:
        return datetime.strptime(val, "%d %b %Y")
    except Exception:
        return datetime.max

def to_date_str(val):
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.strftime("%d %b %Y")
    if isinstance(val, str) and val:
        try:
            return datetime.strptime(val, "%Y-%m-%d").strftime("%d %b %Y")
        except Exception:
            return val
    return None

def to_html_date(val):
    """DD MMM YYYY -> YYYY-MM-DD for <input type=date>."""
    if not val:
        return ""
    try:
        return datetime.strptime(val, "%d %b %Y").strftime("%Y-%m-%d")
    except Exception:
        return ""


# ── Page ──────────────────────────────────────────────────────────────────
@ui.page("/")
def main_page():
    # Per-browser-session state (fresh on each page load)
    docs = {}          # filename -> {"img": PIL.Image, "stamps": [...]}
    state = {"active": None, "selected": None, "sort_mode": "Timeline"}
    verified = set()

    ui.add_head_html("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
      body { font-family:'Open Sans',sans-serif; background:#ffffff; margin:0; }
      .qi-card { transition: box-shadow .15s, border-color .15s; cursor:pointer; }
      .qi-card:hover { box-shadow: %s; }
      ::-webkit-scrollbar { width:6px; }
      ::-webkit-scrollbar-thumb { background:%s; border-radius:4px; }
    </style>
    """ % (ELEV_13, BORDER))

    # ── Top bar ──────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center").style(
        f"background:{RED};padding:14px 28px;margin:0;"
    ):
        ui.label("Immigration Stamp Review").style(
            "color:white;font-weight:700;font-size:17px;"
        )
        ui.label("Automated entry / exit timeline extraction").style(
            "color:rgba(255,255,255,.75);font-size:12px;margin-left:12px;"
        )

    with ui.row().classes("w-full no-wrap").style("padding:20px 28px;gap:24px;align-items:flex-start;"):

        # ── Sidebar ──────────────────────────────────────────────────────
        with ui.column().style("width:300px;flex-shrink:0;gap:12px;"):
            ui.label("WORKSPACE").style(
                f"font-size:11px;font-weight:700;letter-spacing:1.2px;color:{TEXT_TER};"
            )

            upload = ui.upload(
                label="Upload passport images",
                multiple=True, auto_upload=True,
            ).props("accept=.jpg,.jpeg,.png flat").classes("w-full").style(
                f"border:1.5px dashed {BORDER};border-radius:{R_MD};"
            )

            doc_list = ui.column().classes("w-full").style("gap:6px;")
            export_slot = ui.column().classes("w-full")

        # ── Main content ─────────────────────────────────────────────────
        main = ui.column().classes("flex-1").style("gap:0;min-width:0;")

        # ── Renderers ────────────────────────────────────────────────────
        def render_sidebar():
            doc_list.clear()
            with doc_list:
                for name, d in docs.items():
                    stamps = d["stamps"]
                    avg_conf = (sum(cumulative_conf(s) for s in stamps) / len(stamps)
                                if stamps else 0)
                    is_active = name == state["active"]
                    is_done = name in verified
                    bg = hex_rgba(RED, 0.06) if is_active else BG
                    border = f"1.5px solid {RED}" if is_active else f"1px solid {SURFACE}"

                    def select_doc(n=name):
                        state["active"] = n
                        state["selected"] = None
                        render_sidebar()
                        render_main()

                    with ui.card().style(
                        f"width:100%;background:{bg};border:{border};border-radius:{R_MD};"
                        f"padding:10px 12px;box-shadow:none;"
                    ).classes("qi-card").on("click", select_doc):
                        with ui.row().classes("items-center justify-between w-full").style("gap:6px;"):
                            ui.label(name).style(
                                f"font-size:13px;font-weight:600;color:{TEXT_PRI};"
                                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:170px;"
                            )
                            if is_done:
                                ui.label("Verified").style(
                                    f"font-size:10px;font-weight:700;color:{RED};"
                                )
                        with ui.row().classes("items-center w-full").style("gap:6px;margin-top:4px;"):
                            bar_color = RED if avg_conf < REVIEW_THRESHOLD else PURPLE
                            with ui.element("div").style(
                                f"flex:1;background:{SURFACE};border-radius:{R_PILL};height:4px;"
                            ):
                                ui.element("div").style(
                                    f"width:{int(avg_conf*100)}%;background:{bar_color};"
                                    f"height:4px;border-radius:{R_PILL};"
                                )
                            ui.label(f"{avg_conf:.0%}").style(
                                f"font-size:11px;color:{TEXT_TER};"
                            )

            export_slot.clear()
            if verified:
                with export_slot:
                    ui.button("Export Verified Records", on_click=do_export).props(
                        "unelevated"
                    ).style(
                        f"background:{RED};color:white;border-radius:{R_MD};"
                        f"font-weight:600;width:100%;margin-top:8px;"
                    )

        def do_export():
            rows = ["Document,Stamp,Stamp Type,Date Type,Date,Confidence"]
            for name in verified:
                for s in docs[name]["stamps"]:
                    for dte in s.get("dates", []):
                        rows.append(
                            f'{name},{s["id"]},{s["type"]},{dte["type"]},'
                            f'{dte.get("value","")},{dte.get("ocr_conf",0):.3f}'
                        )
            ui.download("\n".join(rows).encode(), "verified_records.csv")

        @ui.refreshable
        def render_main():
            main.clear()
            with main:
                if not docs:
                    with ui.column().style("margin:80px auto;text-align:center;max-width:420px;"):
                        ui.label("Upload a document to begin").style(
                            f"font-size:26px;font-weight:700;color:{TEXT_PRI};"
                        )
                        ui.label(
                            "Select one or more passport images from the sidebar to "
                            "start automated stamp detection and date extraction."
                        ).style(f"color:{TEXT_TER};font-size:14px;margin-top:8px;")
                    return

                if state["active"] is None:
                    state["active"] = next(iter(docs))

                if len(verified) == len(docs) and docs:
                    with ui.column().style(
                        f"margin:60px auto;max-width:440px;text-align:center;"
                        f"background:{BG};border:1px solid {SURFACE};border-radius:{R_MD};"
                        f"padding:36px 28px;box-shadow:{ELEV_13};"
                    ):
                        ui.label("All documents verified").style(
                            f"font-size:20px;font-weight:700;color:{TEXT_PRI};"
                        )
                        ui.label(f"{len(docs)} document(s) reviewed successfully.").style(
                            f"color:{TEXT_TER};font-size:13px;margin-top:6px;"
                        )
                    return

                name = state["active"]
                d = docs[name]
                stamps = d["stamps"]
                img_pil = d["img"]
                sel_id = state["selected"]

                n_entry = sum(1 for s in stamps if s["type"] == "Entry")
                n_exit  = sum(1 for s in stamps if s["type"] == "Exit")
                n_flag  = sum(1 for s in stamps if needs_review(s))
                is_verified = name in verified

                # Header
                with ui.row().classes("items-center justify-between w-full").style(
                    f"border-bottom:1px solid {SURFACE};padding-bottom:12px;margin-bottom:16px;"
                ):
                    with ui.column().style("gap:4px;"):
                        ui.label(name).style(
                            f"font-size:18px;font-weight:700;color:{TEXT_PRI};"
                        )
                        with ui.row().style("gap:6px;"):
                            pill(f"{n_entry} Entry", PURPLE)
                            pill(f"{n_exit} Exit", RED)
                            if n_flag:
                                pill(f"{n_flag} Need Review", TEXT_TER)
                    if is_verified:
                        pill("Verified", RED, outline=True)

                with ui.row().classes("w-full no-wrap").style("gap:24px;align-items:flex-start;"):

                    # ── LEFT: image ─────────────────────────────────────
                    with ui.column().style("flex:1.2;min-width:0;gap:8px;"):
                        overline("Document View")

                        sel = next((s for s in stamps if s["id"] == sel_id), None)
                        if sel:
                            c = TYPE_COLOR.get(sel["type"], TEXT_SEC)
                            with ui.row().classes("items-center w-full").style(
                                f"border-left:3px solid {c};background:{hex_rgba(c,0.06)};"
                                f"border-radius:0 8px 8px 0;padding:8px 14px;gap:10px;"
                            ):
                                with ui.column().style("gap:0;"):
                                    ui.label(f"{sel['id']} — {sel['type']} Stamp").style(
                                        f"font-weight:700;color:{c};font-size:14px;"
                                    )
                                    ui.label(f"Detection confidence: {sel['det_conf']:.0%}").style(
                                        f"font-size:12px;color:{TEXT_TER};"
                                    )
                            crop = img_pil.crop(tuple(pad_box(sel["box"], img_pil.size)))
                            ui.image(crop).style(
                                f"border-radius:{R_MD};box-shadow:{ELEV_7};width:100%;"
                            )
                            ui.label("Full document below — click any stamp to inspect").style(
                                f"font-size:11px;color:{TEXT_TER};"
                            )

                        w, h = img_pil.size
                        img_widget = ui.interactive_image(
                            source=img_pil,
                            content=build_svg_overlay(stamps, sel_id),
                            size=(w, h),
                            events=["click"],
                        ).classes("w-full").style(
                            f"border-radius:{R_MD};box-shadow:{ELEV_13};border:1px solid {SURFACE};"
                        )

                        def on_click(e, sname=name, stamps=stamps, widget=None):
                            for s in stamps:
                                x1, y1, x2, y2 = s["box"]
                                if x1 <= e.image_x <= x2 and y1 <= e.image_y <= y2:
                                    state["selected"] = s["id"]
                                    render_main.refresh()
                                    return
                        img_widget.on_mouse(on_click)

                    # ── RIGHT: timeline / priority list ──────────────────
                    with ui.column().style("flex:1;min-width:0;gap:10px;"):
                        with ui.row().classes("items-center justify-between w-full"):
                            overline("Stamp Timeline")
                            with ui.row().style("gap:4px;"):
                                for mode in ("Timeline", "Priority"):
                                    active_mode = state["sort_mode"] == mode
                                    def set_mode(m=mode):
                                        state["sort_mode"] = m
                                        render_main.refresh()
                                    ui.button(mode, on_click=set_mode).props("flat dense").style(
                                        f"font-size:11px;font-weight:700;padding:2px 10px;"
                                        f"border-radius:{R_PILL};min-width:0;"
                                        f"background:{RED if active_mode else 'transparent'};"
                                        f"color:{'white' if active_mode else TEXT_TER};"
                                    )

                        ordered = (
                            sorted(stamps, key=chrono_key)
                            if state["sort_mode"] == "Timeline"
                            else sorted(stamps, key=cumulative_conf)
                        )

                        with ui.column().style(
                            "gap:0;max-height:640px;overflow-y:auto;padding-right:4px;width:100%;"
                        ):
                            for i, s in enumerate(ordered):
                                if state["sort_mode"] == "Timeline" and i > 0:
                                    prev_val, _ = primary_date(ordered[i - 1])
                                    cur_val, _ = primary_date(s)
                                    if prev_val and cur_val:
                                        ui.label("↓").style(
                                            f"text-align:center;width:100%;color:{BORDER};"
                                            f"font-size:14px;margin:0;line-height:1;"
                                        )
                                render_stamp_card(s, sel_id, name)

                # Verify button
                ui.element("div").style("height:20px;")
                if not is_verified:
                    def do_verify(n=name):
                        verified.add(n)
                        render_sidebar()
                        render_main.refresh()
                    ui.button("Mark Document as Verified", on_click=do_verify).props(
                        "unelevated"
                    ).style(
                        f"background:{RED};color:white;border-radius:{R_MD};"
                        f"font-weight:600;width:100%;padding:10px;"
                    )

        def render_stamp_card(s, sel_id, fname):
            c = TYPE_COLOR.get(s["type"], TEXT_SEC)
            is_sel = s["id"] == sel_id
            flagged = needs_review(s)
            val, _ = primary_date(s)
            conf = cumulative_conf(s)

            border = f"2px solid {c}" if is_sel else (
                f"1.5px dashed {RED}" if flagged else f"1px solid {hex_rgba(c,0.25)}"
            )
            bg = hex_rgba(c, 0.06) if is_sel else BG

            def select(sid=s["id"]):
                state["selected"] = sid
                render_main.refresh()

            with ui.card().style(
                f"width:100%;border:{border};border-radius:{R_MD};background:{bg};"
                f"box-shadow:{ELEV_7 if not is_sel else ELEV_13};padding:12px 16px;margin-bottom:8px;"
            ).classes("qi-card").on("click", select):

                with ui.row().classes("items-center justify-between w-full"):
                    pill(s["type"], c)
                    with ui.row().style("gap:6px;align-items:center;"):
                        if flagged:
                            ui.label("REVIEW").style(
                                f"font-size:10px;font-weight:700;color:{RED};"
                                f"background:{hex_rgba(RED,0.1)};padding:2px 8px;"
                                f"border-radius:{R_PILL};"
                            )
                        ui.label(s["id"]).style(f"font-size:11px;color:{TEXT_TER};")

                ui.label(val if val else "—").style(
                    f"font-size:18px;font-weight:700;color:{TEXT_PRI};margin:6px 0 2px;"
                )

                if val:
                    with ui.element("div").style(
                        f"background:{SURFACE};border-radius:{R_PILL};height:3px;margin-top:4px;"
                    ):
                        ui.element("div").style(
                            f"width:{int(conf*100)}%;background:{RED if flagged else c};"
                            f"height:3px;border-radius:{R_PILL};"
                        )
                    ui.label(f"{conf:.0%} combined confidence").style(
                        f"font-size:11px;color:{TEXT_TER};"
                    )
                else:
                    ui.label("No date detected").style(f"font-size:12px;color:{TEXT_TER};")

                if is_sel:
                    render_inline_editor(s, fname)

        def render_inline_editor(s, fname):
            ui.separator().style(f"margin:10px 0;background:{SURFACE};")
            rows_container = ui.column().style("gap:8px;width:100%;")

            def redraw_rows():
                rows_container.clear()
                with rows_container:
                    for idx, dte in enumerate(s["dates"]):
                        with ui.row().classes("items-center w-full").style("gap:6px;"):
                            type_sel = ui.select(
                                ["Entry", "Exit", "Until", "Unknown"],
                                value=dte.get("type", "Unknown"),
                            ).style("width:100px;")
                            date_in = ui.input(
                                value=to_html_date(dte.get("value")),
                            ).props("type=date dense").style("flex:1;")

                            def commit(idx=idx, type_sel=type_sel, date_in=date_in):
                                s["dates"][idx]["type"] = type_sel.value
                                s["dates"][idx]["value"] = to_date_str(date_in.value)
                                render_main.refresh()

                            type_sel.on("update:model-value", lambda e, c=commit: c())
                            date_in.on("update:model-value", lambda e, c=commit: c())

                            def remove(idx=idx):
                                s["dates"].pop(idx)
                                render_main.refresh()
                            ui.button(icon="close", on_click=remove).props(
                                "flat dense round size=sm"
                            ).style(f"color:{TEXT_TER};")

                    def add_row():
                        s["dates"].append({"value": None, "type": "Entry", "ocr_conf": 0.0})
                        render_main.refresh()
                    ui.button("+ Add Date", on_click=add_row).props("flat dense").style(
                        f"font-size:12px;color:{RED};font-weight:600;padding:0;"
                    )

            redraw_rows()


        # ── Upload handler ───────────────────────────────────────────────
        def handle_upload(e: events.UploadEventArguments):
            # In NiceGUI 3.0+, the uploaded file is stored in e.file, not e.content
            img_data = e.file.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_np = np.array(img)
            
            # Run pipeline
            stamps = pipeline.process(img_np)
            docs[e.name] = {"img": img, "stamps": stamps}
            
            # Set active if this is the first doc
            if state["active"] is None:
                state["active"] = e.name
            
            # Refresh UI
            render_sidebar()
            render_main.refresh()

        upload.on_upload(handle_upload)

        render_sidebar()
        render_main()


# ── Small render helpers (module scope) ─────────────────────────────────────
def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.2f})"

def pad_box(box, size, pad=16):
    w, h = size
    x1, y1, x2, y2 = box
    return (max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad))

def build_svg_overlay(stamps, sel_id):
    parts = []
    for s in stamps:
        x1, y1, x2, y2 = s["box"]
        c = TYPE_COLOR.get(s["type"], TEXT_SEC)
        is_sel = s["id"] == sel_id
        flagged = needs_review(s)
        stroke_w = 4 if is_sel else 2
        opacity = 1.0 if is_sel else 0.55
        dash = 'stroke-dasharray="6,4"' if flagged and not is_sel else ""
        parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
            f'fill="none" stroke="{c}" stroke-width="{stroke_w}" opacity="{opacity}" '
            f'rx="4" {dash}/>'
        )
        label = f'{s["type"]} {s["det_conf"]:.0%}'
        parts.append(
            f'<rect x="{x1}" y="{max(0,y1-24)}" width="{len(label)*7+16}" height="22" '
            f'fill="{c}" opacity="{opacity}" rx="3"/>'
            f'<text x="{x1+8}" y="{max(0,y1-8)}" fill="white" font-size="12" '
            f'font-family="Open Sans">{label}</text>'
        )
    return "".join(parts)

def overline(text):
    ui.label(text.upper()).style(
        f"font-size:11px;font-weight:700;letter-spacing:1.2px;color:{TEXT_TER};"
    )

def pill(text, color, outline=False):
    if outline:
        ui.label(text).style(
            f"font-size:11px;font-weight:700;color:{color};"
            f"background:{hex_rgba(color,0.08)};padding:3px 12px;"
            f"border-radius:999px;"
        )
    else:
        ui.label(text).style(
            f"font-size:11px;font-weight:700;color:white;background:{color};"
            f"padding:3px 12px;border-radius:999px;"
        )


ui.run(title="AIA Immigration Stamp Review", port=8080, show=False, reload=False)