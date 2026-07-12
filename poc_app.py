"""
app.py — AIA Immigration Stamp Review
Run: python app.py   ->  http://localhost:8080
"""

import io
from datetime import datetime, date

import numpy as np
from PIL import Image
from nicegui import ui, events

from pipeline import DocumentPipeline

pipeline = DocumentPipeline()

# ── Qi Design Tokens ──────────────────────────────────────────────────────
RED, PURPLE, SALMON, CERISE = "#d31145", "#4c4794", "#ff7a85", "#ba0361"
TEXT_PRI, TEXT_SEC, TEXT_TER = "#14181c", "#333d47", "#858b91"
BG, BG_ALT, SURFACE, BORDER = "#ffffff", "#f5f5f6", "#ebeced", "#adb1b5"
R_MD, R_PILL = "15px", "999px"
ELEV_7  = "0px 2px 4px rgba(0,0,0,0.08)"
ELEV_13 = "0px 2px 6px rgba(0,0,0,0.06), 0px 5px 8px rgba(0,0,0,0.04)"

TYPE_COLOR = {"Entry": PURPLE, "Exit": RED}
REVIEW_THRESHOLD = 0.75
VW, VH = 600, 420          # fixed zoom-viewport pixel size (kept modest to avoid overlap)
SIDEBAR_W = 300


# ── Domain helpers ───────────────────────────────────────────────────────────
def primary_entry(s):
    dates = [d for d in s.get("dates", []) if d.get("value") and d["type"] in ("Entry", "Exit")]
    if not dates:
        return None
    return max(dates, key=lambda d: (d.get("ocr_conf") or 0))

def needs_review(s):
    entry = primary_entry(s)
    if entry is None:
        return True
    conf = entry.get("ocr_conf")
    return conf is not None and conf < REVIEW_THRESHOLD

def chrono_key(s):
    entry = primary_entry(s)
    if not entry:
        return datetime.max
    try:
        return datetime.strptime(entry["value"], "%d %b %Y")
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
    if not val:
        return ""
    try:
        return datetime.strptime(val, "%d %b %Y").strftime("%Y-%m-%d")
    except Exception:
        return ""

def hex_rgba(h, a):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a:.2f})"


def build_svg_overlay(stamps, sel_id, pending_point=None):
    parts = []
    for s in stamps:
        if s.get("deleted"):
            continue
        x1, y1, x2, y2 = s["box"]
        c = TYPE_COLOR.get(s["type"], TEXT_SEC)
        is_sel = s["id"] == sel_id
        flagged = needs_review(s)
        stroke_w = 4 if is_sel else 2
        opacity = 1.0 if is_sel else 0.55
        dash = 'stroke-dasharray="6,4"' if flagged and not is_sel else ""
        parts.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
            f'fill="none" stroke="{c}" stroke-width="{stroke_w}" opacity="{opacity}" rx="4" {dash}/>'
        )
        label = f'{s["type"]} {s["det_conf"]:.0%}'
        parts.append(
            f'<rect x="{x1}" y="{max(0,y1-24)}" width="{len(label)*7+16}" height="22" '
            f'fill="{c}" opacity="{opacity}" rx="3"/>'
            f'<text x="{x1+8}" y="{max(0,y1-8)}" fill="white" font-size="12" '
            f'font-family="Open Sans">{label}</text>'
        )
    if pending_point:
        px, py = pending_point
        parts.append(
            f'<circle cx="{px}" cy="{py}" r="8" fill="none" stroke="{RED}" stroke-width="3"/>'
            f'<circle cx="{px}" cy="{py}" r="2" fill="{RED}"/>'
        )
    return "".join(parts)


def overline(text):
    ui.label(text.upper()).style(
        f"font-size:11px;font-weight:700;letter-spacing:1.2px;color:{TEXT_TER};"
    )

def pill(text, color, outline=False):
    if outline:
        ui.label(text).style(
            f"font-size:11px;font-weight:700;color:{color};background:{hex_rgba(color,0.08)};"
            f"padding:3px 12px;border-radius:999px;"
        )
    else:
        ui.label(text).style(
            f"font-size:11px;font-weight:700;color:white;background:{color};"
            f"padding:3px 12px;border-radius:999px;"
        )


# ── Page ──────────────────────────────────────────────────────────────────
@ui.page("/")
def main_page():
    docs = {}
    state = {"active": None, "selected": None, "add_mode": False,
             "pending_point": None, "panel": None, "show_removed": {},
             "sidebar_open": True}
    verified = set()

    ui.add_head_html(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
      body {{ font-family:'Open Sans',sans-serif; background:#ffffff; margin:0; }}
      .qi-row {{ cursor:pointer; transition:background .12s; }}
      .qi-row:hover {{ background:{BG_ALT} !important; }}
      .qi-card {{ transition: box-shadow .15s; cursor:pointer; }}
      .qi-card:hover {{ box-shadow:{ELEV_13}; }}
      ::-webkit-scrollbar {{ width:6px; }}
      ::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:4px; }}
    </style>
    """)

    # ── Top bar ──────────────────────────────────────────────────────────
    with ui.row().classes("w-full items-center justify-between").style(
        f"background:{RED};padding:14px 28px;margin:0;"
    ):
        with ui.row().classes("items-center").style("gap:12px;"):
            sidebar_toggle_btn = ui.button(icon="menu", on_click=lambda: toggle_sidebar()).props(
                "flat round dense").style("color:white;")
            with ui.column().style("gap:0;"):
                ui.label("Immigration Stamp Review").style("color:white;font-weight:700;font-size:17px;")
                ui.label("Automated entry / exit timeline extraction").style(
                    "color:rgba(255,255,255,.75);font-size:12px;")

    with ui.row().classes("w-full").style("padding:20px 28px;gap:24px;align-items:flex-start;flex-wrap:nowrap;"):

        # ── Sidebar (collapsible) ───────────────────────────────────────
        sidebar_wrapper = ui.element("div").style(
            f"width:{SIDEBAR_W}px;flex-shrink:0;overflow:hidden;"
            f"transition:width .25s ease, opacity .2s ease;opacity:1;"
        )
        with sidebar_wrapper:
            with ui.column().style(f"width:{SIDEBAR_W}px;gap:12px;"):
                ui.label("WORKSPACE").style(
                    f"font-size:11px;font-weight:700;letter-spacing:1.2px;color:{TEXT_TER};")
                upload = ui.upload(label="Upload passport images", multiple=True, auto_upload=True).props(
                    "accept=.jpg,.jpeg,.png flat").classes("w-full").style(
                    f"border:1.5px dashed {BORDER};border-radius:{R_MD};")
                doc_list = ui.column().classes("w-full").style("gap:6px;")
                export_slot = ui.column().classes("w-full")

        def toggle_sidebar():
            state["sidebar_open"] = not state["sidebar_open"]
            if state["sidebar_open"]:
                sidebar_wrapper.style(
                    f"width:{SIDEBAR_W}px;flex-shrink:0;overflow:hidden;"
                    f"transition:width .25s ease, opacity .2s ease;opacity:1;"
                )
            else:
                sidebar_wrapper.style(
                    f"width:0px;flex-shrink:0;overflow:hidden;"
                    f"transition:width .25s ease, opacity .2s ease;opacity:0;"
                )

        main = ui.column().classes("flex-1").style("gap:0;min-width:0;")

        # ── Sidebar list rendering ───────────────────────────────────────
        summary_slot = ui.column().classes("w-full").style("gap:4px;margin-bottom:4px;")

        def render_sidebar():
            summary_slot.clear()
            with summary_slot:
                if docs:
                    done, total = len(verified), len(docs)
                    with ui.row().classes("items-center justify-between w-full"):
                        ui.label("Documents reviewed").style(
                            f"font-size:11px;color:{TEXT_TER};font-weight:600;")
                        ui.label(f"{done} / {total}").style(
                            f"font-size:12px;color:{TEXT_PRI};font-weight:700;")
                    with ui.element("div").style(
                        f"background:{SURFACE};border-radius:{R_PILL};height:5px;overflow:hidden;"
                    ):
                        pct = int((done / total) * 100) if total else 0
                        ui.element("div").style(
                            f"width:{pct}%;background:{RED};height:5px;transition:width .3s ease;")

            doc_list.clear()
            with doc_list:
                for name, d in docs.items():
                    active_stamps = [s for s in d["stamps"] if not s.get("deleted")]
                    flagged = sum(1 for s in active_stamps if needs_review(s))
                    total = len(active_stamps)
                    is_active = name == state["active"]
                    is_done = name in verified
                    bg = hex_rgba(RED, 0.06) if is_active else BG
                    border = f"1.5px solid {RED}" if is_active else f"1px solid {SURFACE}"

                    def select_doc(n=name):
                        state["active"] = n
                        state["selected"] = None
                        state["add_mode"] = False
                        state["pending_point"] = None
                        render_sidebar()
                        render_header.refresh()
                        render_document_pane.refresh()
                        render_table.refresh()

                    with ui.card().style(
                        f"width:100%;background:{bg};border:{border};border-radius:{R_MD};"
                        f"padding:10px 12px;box-shadow:none;"
                    ).classes("qi-card").on("click", select_doc):
                        with ui.row().classes("items-center justify-between w-full").style("gap:6px;"):
                            ui.label(name).style(
                                f"font-size:13px;font-weight:600;color:{TEXT_PRI};"
                                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;")
                            if is_done:
                                ui.icon("check_circle").style(f"color:{RED};font-size:16px;")
                        with ui.row().classes("items-center w-full").style("gap:6px;margin-top:4px;"):
                            with ui.element("div").style(
                                f"flex:1;background:{SURFACE};border-radius:{R_PILL};height:4px;overflow:hidden;"
                            ):
                                pct = int((flagged / total) * 100) if total else 0
                                ui.element("div").style(f"width:{pct}%;background:{RED};height:4px;")
                            if total == 0:
                                label_txt = "no stamps"
                            elif flagged:
                                label_txt = f"{flagged} need review"
                            else:
                                label_txt = "all clear"
                            ui.label(label_txt).style(
                                f"font-size:11px;color:{RED if flagged else CERISE};"
                                f"font-weight:{'700' if flagged else '600'};white-space:nowrap;")

            export_slot.clear()
            with export_slot:
                if verified:
                    ui.button("Export Verified Records", on_click=export_verified).props("unelevated").style(
                        f"background:{RED};color:white;border-radius:{R_MD};font-weight:600;width:100%;")
                any_feedback = any(
                    s.get("source") == "user_added" or s.get("deleted")
                    for d in docs.values() for s in d["stamps"]
                )
                if any_feedback:
                    ui.button("Export Retrain Feedback", on_click=export_feedback).props(
                        "unelevated outline").style(
                        f"color:{RED};border:1px solid {RED};border-radius:{R_MD};"
                        f"font-weight:600;width:100%;margin-top:6px;background:transparent;")

        def export_verified():
            rows = ["Document,Stamp,Stamp Type,Date Type,Date,OCR Confidence"]
            for name in verified:
                for s in docs[name]["stamps"]:
                    if s.get("deleted"):
                        continue
                    for dte in s.get("dates", []):
                        rows.append(
                            f'{name},{s["id"]},{s["type"]},{dte["type"]},'
                            f'{dte.get("value","")},{dte.get("ocr_conf") or ""}')
            ui.download("\n".join(rows).encode(), "verified_records.csv")

        def export_feedback():
            rows = ["Document,Stamp,Type,Box,Source,Status"]
            for name, d in docs.items():
                for s in d["stamps"]:
                    status = "removed" if s.get("deleted") else "active"
                    box = "|".join(str(v) for v in s["box"])
                    rows.append(f'{name},{s["id"]},{s["type"]},{box},{s.get("source","model_detected")},{status}')
            ui.download("\n".join(rows).encode(), "retrain_feedback.csv")

        # ── Zoom mechanics ───────────────────────────────────────────────
        def apply_zoom(px, py, scale, animate=True):
            panel = state["panel"]
            if not panel:
                return
            tx = VW / 2 - px * scale
            ty = VH / 2 - py * scale
            transition = "transform .5s cubic-bezier(.4,0,.2,1)" if animate else "none"
            panel["wrapper"].style(
                f"transform:translate({tx}px,{ty}px) scale({scale});"
                f"transform-origin:0 0;transition:{transition};"
            )

        def zoom_to_stamp(s):
            x1, y1, x2, y2 = s["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
            target = min(VW * 0.55 / bw, VH * 0.55 / bh)
            fit = state["panel"]["fit"]
            target = max(fit * 1.2, min(target, fit * 6))
            apply_zoom(cx, cy, target)

        def zoom_reset():
            panel = state["panel"]
            if not panel:
                return
            W, H = panel["size"]
            apply_zoom(W / 2, H / 2, panel["fit"])

        # ── Image click handling ─────────────────────────────────────────
        def on_image_click(e):
            name = state["active"]
            stamps = docs[name]["stamps"]
            if state["add_mode"]:
                if state["pending_point"] is None:
                    state["pending_point"] = (e.image_x, e.image_y)
                    state["panel"]["img_widget"].set_content(
                        build_svg_overlay(stamps, None, state["pending_point"]))
                    render_toolbar.refresh()
                else:
                    x1, y1 = state["pending_point"]
                    x2, y2 = e.image_x, e.image_y
                    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                    if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                        return
                    doc = docs[name]
                    uid = f"STAMP-USER-{doc['next_uid']}"
                    doc["next_uid"] += 1
                    new_stamp = {"id": uid, "type": "Entry", "box": box, "det_conf": 1.0,
                                 "dates": [], "source": "user_added", "deleted": False}
                    doc["stamps"].append(new_stamp)
                    state["add_mode"] = False
                    state["pending_point"] = None
                    state["selected"] = uid
                    zoom_to_stamp(new_stamp)
                    state["panel"]["img_widget"].set_content(build_svg_overlay(doc["stamps"], uid))
                    render_toolbar.refresh()
                    render_table.refresh()
                    render_sidebar()
                return

            for s in stamps:
                if s.get("deleted"):
                    continue
                x1, y1, x2, y2 = s["box"]
                if x1 <= e.image_x <= x2 and y1 <= e.image_y <= y2:
                    select_stamp(s["id"])
                    return

        def select_stamp(sid):
            state["selected"] = sid
            name = state["active"]
            stamps = docs[name]["stamps"]
            s = next((x for x in stamps if x["id"] == sid), None)
            if s:
                zoom_to_stamp(s)
                state["panel"]["img_widget"].set_content(build_svg_overlay(stamps, sid))
            render_table.refresh()

        def toggle_add_mode():
            state["add_mode"] = not state["add_mode"]
            state["pending_point"] = None
            state["selected"] = None
            if state["add_mode"]:
                zoom_reset()
            stamps = docs[state["active"]]["stamps"]
            state["panel"]["img_widget"].set_content(build_svg_overlay(stamps, None))
            render_toolbar.refresh()
            render_table.refresh()

        # ── Toolbar ──────────────────────────────────────────────────────
        @ui.refreshable
        def render_toolbar():
            add_on = state["add_mode"]
            with ui.row().classes("items-center w-full").style("gap:8px;margin-bottom:8px;flex-wrap:wrap;"):
                ui.button(
                    "Cancel Add Stamp" if add_on else "+ Add Stamp",
                    on_click=toggle_add_mode
                ).props("unelevated dense").style(
                    f"background:{RED if add_on else BG};color:{'white' if add_on else TEXT_SEC};"
                    f"border:1px solid {RED if add_on else BORDER};border-radius:{R_MD};"
                    f"font-size:12px;font-weight:600;padding:6px 14px;"
                )
                ui.button("Reset View", on_click=lambda: zoom_reset()).props("flat dense").style(
                    f"color:{TEXT_TER};font-size:12px;font-weight:600;"
                )
                if add_on:
                    hint = ("Click the first corner of the stamp"
                            if state["pending_point"] is None
                            else "Click the opposite corner to finish")
                    ui.label(hint).style(f"font-size:12px;color:{RED};font-weight:600;")

        # ── Document image pane (rebuilt only on doc switch) ──────────────
        @ui.refreshable
        def render_document_pane():
            name = state["active"]
            if name is None:
                return
            d = docs[name]
            img_pil = d["img"]
            W, H = img_pil.size
            fit = min(VW / W, VH / H)
            d["fit"] = fit

            render_toolbar()

            with ui.element("div").style(
                f"width:{VW}px;max-width:100%;height:{VH}px;overflow:hidden;position:relative;"
                f"border-radius:{R_MD};box-shadow:{ELEV_13};border:1px solid {SURFACE};"
                f"background:{BG_ALT};"
            ):
                wrapper = ui.element("div").style(
                    f"width:{W}px;height:{H}px;position:absolute;top:0;left:0;"
                    f"transform-origin:0 0;transition:transform .5s cubic-bezier(.4,0,.2,1);"
                )
                with wrapper:
                    img_widget = ui.interactive_image(
                        source=img_pil,
                        content=build_svg_overlay(d["stamps"], state["selected"]),
                        size=(W, H),
                        events=["click"],
                    ).style(f"width:{W}px;height:{H}px;")
                    img_widget.on_mouse(on_image_click)

            state["panel"] = {"wrapper": wrapper, "img_widget": img_widget, "fit": fit, "size": (W, H)}
            apply_zoom(W / 2, H / 2, fit, animate=False)

            ui.label("Click a stamp to inspect it, or use Add Stamp to mark a missed one.").style(
                f"font-size:11px;color:{TEXT_TER};text-align:center;margin-top:6px;"
            )

        # ── Header (independent refreshable — verify toggle lives here) ──
        @ui.refreshable
        def render_header():
            name = state["active"]
            if name is None:
                return
            stamps = docs[name]["stamps"]
            active_stamps = [s for s in stamps if not s.get("deleted")]
            n_entry = sum(1 for s in active_stamps if s["type"] == "Entry")
            n_exit  = sum(1 for s in active_stamps if s["type"] == "Exit")
            n_flag  = sum(1 for s in active_stamps if needs_review(s))
            is_verified = name in verified

            def toggle_verify():
                if name in verified:
                    verified.discard(name)
                else:
                    verified.add(name)
                render_sidebar()
                render_header.refresh()

            with ui.row().classes("items-center justify-between w-full").style(
                f"border-bottom:1px solid {SURFACE};padding-bottom:12px;margin-bottom:16px;flex-wrap:wrap;gap:8px;"
            ):
                with ui.column().style("gap:4px;"):
                    ui.label(name).style(f"font-size:18px;font-weight:700;color:{TEXT_PRI};")
                    with ui.row().style("gap:6px;"):
                        pill(f"{n_entry} Entry", PURPLE)
                        pill(f"{n_exit} Exit", RED)
                        if n_flag:
                            pill(f"{n_flag} Need Review", TEXT_TER)

                with ui.row().classes("items-center").style("gap:6px;cursor:pointer;").on(
                    "click", toggle_verify
                ):
                    ui.label("Reviewed" if is_verified else "Mark as reviewed").style(
                        f"font-size:12px;font-weight:600;color:{RED if is_verified else TEXT_TER};"
                    )
                    ui.icon("check_circle" if is_verified else "radio_button_unchecked").style(
                        f"color:{RED if is_verified else TEXT_TER};font-size:22px;"
                    )

        # ── Timeline table ───────────────────────────────────────────────
        def render_editor_row(s, name):
            with ui.column().style(
                f"width:100%;background:{BG_ALT};border-radius:0 0 {R_MD} {R_MD};"
                f"padding:12px 16px;gap:8px;"
            ):
                rows_container = ui.column().style("gap:6px;width:100%;")

                def redraw():
                    rows_container.clear()
                    with rows_container:
                        for idx, dte in enumerate(s["dates"]):
                            with ui.row().classes("items-center w-full").style("gap:6px;flex-wrap:wrap;"):
                                type_sel = ui.select(
                                    ["Entry", "Exit", "Until", "Unknown"],
                                    value=dte.get("type", "Unknown"),
                                ).style("width:100px;")
                                date_in = ui.input(value=to_html_date(dte.get("value"))).props(
                                    "type=date dense").style("flex:1;min-width:140px;")
                                conf = dte.get("ocr_conf")
                                ui.label(f"{conf:.0%}" if conf is not None else "Manual").style(
                                    f"font-size:11px;color:{TEXT_TER};width:56px;"
                                )

                                def commit(idx=idx, type_sel=type_sel, date_in=date_in):
                                    s["dates"][idx]["type"] = type_sel.value
                                    s["dates"][idx]["value"] = to_date_str(date_in.value)
                                    s["dates"][idx]["ocr_conf"] = None
                                    if state["panel"]:
                                        state["panel"]["img_widget"].set_content(
                                            build_svg_overlay(docs[name]["stamps"], s["id"]))
                                    render_table.refresh()

                                type_sel.on("update:model-value", lambda e, c=commit: c())
                                date_in.on("update:model-value", lambda e, c=commit: c())

                                def remove_date(idx=idx):
                                    s["dates"].pop(idx)
                                    render_table.refresh()
                                ui.button(icon="close", on_click=remove_date).props(
                                    "flat dense round size=sm").style(f"color:{TEXT_TER};")

                        def add_date():
                            s["dates"].append({"value": None, "type": "Entry", "ocr_conf": None})
                            render_table.refresh()
                        ui.button("+ Add Date", on_click=add_date).props("flat dense").style(
                            f"font-size:12px;color:{RED};font-weight:600;padding:0;")

                redraw()

                def remove_stamp():
                    s["deleted"] = True
                    state["selected"] = None
                    if state["panel"]:
                        state["panel"]["img_widget"].set_content(
                            build_svg_overlay(docs[name]["stamps"], None))
                    render_table.refresh()
                    render_sidebar()
                ui.separator().style(f"background:{SURFACE};margin:4px 0;")
                ui.button("Remove This Stamp (false detection)", on_click=remove_stamp).props(
                    "flat dense").style(f"font-size:12px;color:{RED};font-weight:600;padding:0;")

        @ui.refreshable
        def render_table():
            name = state["active"]
            if name is None:
                return
            stamps = docs[name]["stamps"]
            active_stamps = [s for s in stamps if not s.get("deleted")]
            ordered = sorted(active_stamps, key=chrono_key)
            sel_id = state["selected"]

            with ui.element("div").style(
                f"display:grid;grid-template-columns:10px 120px 80px 130px 1fr 30px;"
                f"gap:12px;padding:8px 14px;background:{BG_ALT};border-radius:{R_MD} {R_MD} 0 0;"
                f"align-items:center;"
            ):
                ui.label("")
                ui.label("DATE").style(f"font-size:10px;font-weight:700;color:{TEXT_TER};letter-spacing:.6px;")
                ui.label("TYPE").style(f"font-size:10px;font-weight:700;color:{TEXT_TER};letter-spacing:.6px;")
                ui.label("STAMP").style(f"font-size:10px;font-weight:700;color:{TEXT_TER};letter-spacing:.6px;")
                ui.label("OCR CONFIDENCE").style(f"font-size:10px;font-weight:700;color:{TEXT_TER};letter-spacing:.6px;")
                ui.label("")

            with ui.column().style(
                f"width:100%;gap:0;border:1px solid {SURFACE};border-top:none;"
                f"border-radius:0 0 {R_MD} {R_MD};max-height:520px;overflow-y:auto;"
            ):
                if not ordered:
                    with ui.row().style("padding:20px;justify-content:center;"):
                        ui.label("No stamps detected. Use Add Stamp on the image to add one.").style(
                            f"color:{TEXT_TER};font-size:13px;")

                for s in ordered:
                    entry = primary_entry(s)
                    c = TYPE_COLOR.get(s["type"], TEXT_SEC)
                    is_sel = s["id"] == sel_id
                    flagged = needs_review(s)
                    val = entry["value"] if entry else None
                    conf = entry.get("ocr_conf") if entry else None

                    dot_color = RED if flagged else CERISE
                    row_bg = hex_rgba(c, 0.05) if is_sel else BG

                    def onclick(sid=s["id"]):
                        select_stamp(sid)

                    with ui.element("div").style(
                        f"display:grid;grid-template-columns:10px 120px 80px 130px 1fr 30px;"
                        f"gap:12px;padding:10px 14px;align-items:center;background:{row_bg};"
                        f"border-left:3px solid {c};border-bottom:1px solid {SURFACE};"
                    ).classes("qi-row").on("click", onclick):
                        with ui.element("div").style(
                            f"width:8px;height:8px;border-radius:50%;background:{dot_color};"
                        ):
                            pass
                        ui.label(val if val else "—").style(
                            f"font-size:14px;font-weight:700;color:{TEXT_PRI};")
                        pill(s["type"], c)
                        with ui.column().style("gap:0;"):
                            ui.label(s["id"]).style(f"font-size:11px;color:{TEXT_TER};")
                            ui.label(f"Detection {s['det_conf']:.0%}").style(
                                f"font-size:10px;color:{TEXT_TER};")
                        if val:
                            with ui.column().style("gap:2px;width:100%;"):
                                if conf is not None:
                                    with ui.element("div").style(
                                        f"background:{SURFACE};border-radius:{R_PILL};height:4px;overflow:hidden;"
                                    ):
                                        ui.element("div").style(
                                            f"width:{int(conf*100)}%;background:{RED if flagged else CERISE};height:4px;")
                                    ui.label(
                                        f"{conf:.0%} OCR" + ("  •  Needs review" if flagged else "")
                                    ).style(f"font-size:11px;color:{RED if flagged else TEXT_TER};"
                                            f"font-weight:{'700' if flagged else '400'};")
                                else:
                                    ui.label("Manual entry").style(f"font-size:12px;color:{TEXT_TER};")
                        else:
                            ui.label("No date detected — click to add").style(
                                f"font-size:12px;color:{RED};font-weight:600;")
                        ui.label("−" if is_sel else "+").style(
                            f"font-size:16px;color:{TEXT_TER};text-align:center;")

                    if is_sel:
                        render_editor_row(s, name)

            removed = [s for s in stamps if s.get("deleted")]
            if removed:
                shown = state["show_removed"].get(name, False)
                def toggle_removed():
                    state["show_removed"][name] = not state["show_removed"].get(name, False)
                    render_table.refresh()
                ui.button(
                    f"{'Hide' if shown else 'Show'} removed stamps ({len(removed)})",
                    on_click=toggle_removed
                ).props("flat dense").style(f"font-size:12px;color:{TEXT_TER};margin-top:6px;")
                if shown:
                    for s in removed:
                        with ui.row().classes("items-center justify-between w-full").style(
                            f"padding:8px 14px;background:{BG_ALT};border-radius:{R_MD};margin-top:4px;opacity:.6;"
                        ):
                            ui.label(f'{s["id"]} — {s["type"]} (removed)').style(
                                f"font-size:12px;color:{TEXT_TER};text-decoration:line-through;")
                            def restore(s=s):
                                s["deleted"] = False
                                render_table.refresh()
                                render_sidebar()
                                if state["panel"]:
                                    state["panel"]["img_widget"].set_content(
                                        build_svg_overlay(docs[name]["stamps"], None))
                            ui.button("Restore", on_click=restore).props("flat dense").style(
                                f"font-size:11px;color:{RED};font-weight:600;")

        # ── Main shell (rebuilt only on structural changes) ───────────────
        @ui.refreshable
        def render_main():
            main.clear()
            with main:
                if not docs:
                    with ui.column().style("margin:80px auto;text-align:center;max-width:420px;"):
                        ui.label("Upload a document to begin").style(
                            f"font-size:26px;font-weight:700;color:{TEXT_PRI};")
                        ui.label(
                            "Select one or more passport images from the sidebar to start "
                            "automated stamp detection and date extraction."
                        ).style(f"color:{TEXT_TER};font-size:14px;margin-top:8px;")
                    return

                if state["active"] is None:
                    state["active"] = next(iter(docs))

                render_header()

                with ui.row().classes("w-full").style(
                    "gap:24px;align-items:flex-start;flex-wrap:wrap;"
                ):
                    with ui.column().style(f"flex:0 0 auto;min-width:0;gap:4px;max-width:{VW}px;"):
                        overline("Document View")
                        render_document_pane()

                    with ui.column().style("flex:1 1 380px;min-width:340px;overflow:hidden;gap:8px;"):
                        overline("Stamp Timeline — chronological, colour shows priority")
                        render_table()

        # ── Upload handler ───────────────────────────────────────────────
        async def handle_upload(e: events.UploadEventArguments):
            content = await e.file.read() if hasattr(e.file, "read") else e.content.read()
            img = Image.open(io.BytesIO(content)).convert("RGB")
            img_np = np.array(img)
            stamps = pipeline.process(img_np)
            for s in stamps:
                s.setdefault("source", "model_detected")
                s.setdefault("deleted", False)
            fname = getattr(e, "name", None) or getattr(e.file, "filename", "upload.jpg")
            was_empty = state["active"] is None
            docs[fname] = {"img": img, "stamps": stamps, "next_uid": 1}
            if was_empty:
                state["active"] = fname
            render_sidebar()
            if was_empty:
                render_main.refresh()

        upload.on_upload(handle_upload)

        render_sidebar()
        render_main()


ui.run(title="AIA Immigration Stamp Review", port=8080, show=False, reload=False)