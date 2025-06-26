#!/usr/bin/env python3
"""
focus_youtube_tile.py

Drive the Home-screen grid until the red-outlined tile containing the
YouTube icon is in focus, then stop.

Usage
-----
  python focus_youtube_tile.py <PortMask>    # e.g. 8
  python focus_youtube_tile.py <PortMask> -f KWGN   # optional OCR filter

The script re-uses helpers from your Netflix version (REST snapshot,
send_command_ex, etc.) but fixes the template path, missing globals,
and robustness issues.
"""
from __future__ import annotations

import os, sys, time, uuid, json, cv2, re, subprocess, pathlib
from typing import Any

import numpy as np
import pytesseract
import requests
from skimage.metrics import structural_similarity as ssim   # runtime import ok

# ─────────────────────────────────────────────────────────────
# CONFIG & paths
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR   = pathlib.Path(__file__).parent
DOTFINDER_DIR = SCRIPT_DIR / "dotfinder"
DOTFINDER_DIR.mkdir(exist_ok=True)
STUDIO_URL = "http://127.0.0.1:5000"  # os.getenv("DP_STUDIO")  # e.g. "http://<studio-ip>:5000"
USER = ""  # os.getenv("DP_USER", "")
YOUTUBE_LOGO = SCRIPT_DIR / "youtube_logo_166x100.png"   # provide a crisp PNG!
DEBUG_SAVE_ROI = True
_ICON: np.ndarray | None = None

# ─────────────────────────────────────────────────────────────
# optional “-f TEXT” command-line filter
# ─────────────────────────────────────────────────────────────
filter_text = "youtube"
if "-f" in sys.argv:
    i = sys.argv.index("-f")
    if i == len(sys.argv) - 1:
        sys.exit("Error: -f requires a filter text")
    filter_text = sys.argv[i + 1]
    del sys.argv[i : i + 2]

# ─────────────────────────────────────────────────────────────
# Debug helper
# ─────────────────────────────────────────────────────────────
def _save_dbg(name: str, img: np.ndarray) -> None:
    cv2.imwrite(str(DOTFINDER_DIR / name), img)


YOUTUBE_WM_LOGO = SCRIPT_DIR / "youtube_wm_48x48.png"  # your watermark crop

_WM: np.ndarray | None = None
def _ensure_watermark() -> np.ndarray:
    global _WM
    if _WM is None:
        wm = cv2.imread(str(YOUTUBE_WM_LOGO), cv2.IMREAD_UNCHANGED)
        if wm is None:
            raise FileNotFoundError(f"Cannot load {YOUTUBE_WM_LOGO}")
        if wm.shape[2] == 4:
            wm = cv2.cvtColor(wm, cv2.COLOR_BGRA2BGR)
        _WM = wm
    return _WM

def _tile_has_watermark(tile: np.ndarray, tag: str = "") -> bool:
    """
    Template-match the small YouTube watermark.
    """
    tmpl = _ensure_watermark()
    g1 = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    _, b1 = cv2.threshold(g1, 200, 255, cv2.THRESH_BINARY_INV)  # watermark is bright
    _, b2 = cv2.threshold(g2, 200, 255, cv2.THRESH_BINARY_INV)
    res = cv2.matchTemplate(b1, b2, cv2.TM_CCOEFF_NORMED)
    mx = cv2.minMaxLoc(res)[1]
    if tag:
        print(f"[WM  {tag}] {mx:.2f}")
    return mx >= 0.80  # adjust down if needed

def _save(name, img): cv2.imwrite(str(DOT / name), img)

def convertPortViewMask2String(mask: int) -> str:
    binary_str = format(mask, '08b')[::-1]
    return ''.join(str(idx + 1) for idx, bit in enumerate(binary_str) if bit == '1')

def send_command_ex(mask: int, command: str):
    portview = convertPortViewMask2String(mask)
    payload = {"portviews": portview, "keyname": command, "username": "", "keypressduration": 0, "delayduration": 0}
    try:
        requests.post("http://127.0.0.1:5005/api/execute-irsenderagent2", json=payload)
        print(f"Sent command '{command}' to portview {portview}")
    except Exception as e:
        print(f"Error sending command: {e}")

def grab_current_frame(mask: int) -> np.ndarray:
    tmp_png = fr"\\INDEVW-DEVPART03\DataStorage\Snapshots\{uuid.uuid4()}.png"
    payload = {
        "PortView": mask, "Operation": 3, "OutputFilePath": tmp_png,
        "xLeftTopRectangle": 0, "yLeftTopRectangle": 0, "RectangleWidth": 960, "RectangleHeight": 540,
        "RecordDuration": 0, "Threshold": 0, "Duration": 0, "MonitorDuration": 0,
        "AudioCutOffThreshold": 0
    }
    r = requests.post(f"{STUDIO_URL}/api/v4/video", json=payload)
    r.raise_for_status()
    result = r.json()
    img = cv2.imread(result["ResultFilePath"])
    if img is None:
        raise IOError(f"Cannot read {result['ResultFilePath']}")
    return img

def _red_mask_strict(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array((0, 150, 150)); upper1 = np.array((10, 255, 255))
    lower2 = np.array((160, 150, 150)); upper2 = np.array((180, 255, 255))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)

def _find_highlight_rect_strict(frame: np.ndarray, tag: str) -> tuple[int, int, int, int] | None:
    mask = _red_mask_strict(frame)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 800 < cv2.contourArea(c) and 0.4 < w/h < 2.5:
            best = (x, y, w, h)
    if best:
        dbg = frame.copy()
        cv2.rectangle(dbg, (best[0], best[1]), (best[0]+best[2], best[1]+best[3]), (0, 255, 0), 3)
        _save(f"{tag}_highlight.png", dbg)
        return best
    _save(f"{tag}_no_highlight.png", frame)
    return None

def _ensure_icon() -> np.ndarray:
    global _ICON
    if _ICON is None:
        icon = cv2.imread(str(YOUTUBE_LOGO), cv2.IMREAD_UNCHANGED)
        if icon is None:
            raise FileNotFoundError(f"Cannot load {YOUTUBE_LOGO}")
        if icon.shape[2] == 4:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGRA2BGR)
        _ICON = icon
    return _ICON


def _red_mask_strict(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1, upper1 = (0, 150, 150), (10, 255, 255)
    lower2, upper2 = (160, 150, 150), (180, 255, 255)
    return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)


def _find_highlight_rect_strict(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = _red_mask_strict(frame)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area, aspect = cv2.contourArea(c), w / h if h else 0
        if area > 800 and 0.4 < aspect < 2.5:
            if best is None or area > best[-1]:
                best = (x, y, w, h, area)
    return best[:4] if best else None


def _tile_has_icon(tile: np.ndarray, tag: str = "") -> bool:
    tmpl = _ensure_icon()
    g1 = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    _, b1 = cv2.threshold(g1, 180, 255, cv2.THRESH_BINARY_INV)
    _, b2 = cv2.threshold(g2, 180, 255, cv2.THRESH_BINARY_INV)
    res = cv2.matchTemplate(b1, b2, cv2.TM_CCOEFF_NORMED)
    mx = cv2.minMaxLoc(res)[1]
    if tag:
        print(f"[TMPL {tag}] {mx:.2f}")
    return mx >= 0.60  # generous


def _label_is_youtube(frame: np.ndarray, rect, tag: str = "") -> bool:
    """
    OCR the text **below** the highlighted tile and decide if it says YOUTUBE.
    Returns False instantly if the ROI would fall outside the frame.
    """
    x, y, w, h = rect
    H, W = frame.shape[:2]

    # Expected band for the label: a strip 5-60 % of tile-height below the tile.
    y1 = y + h + int(0.05 * h)
    y2 = y + h + int(0.60 * h)

    # Clamp to frame; if it collapses, skip OCR.
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if y2 - y1 < 5:            # nothing to read
        return False

    roi = frame[y1:y2, x : min(W, x + w)]
    if roi.size == 0:
        return False

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(
        b,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ).strip().upper()

    if tag:
        print(f"[OCR {tag}] {txt!r}")

    return "YOUTUBE" in txt or "YOU TUBE" in txt


# ─────────────────────────────────────────────────────────────
# Navigation
# ─────────────────────────────────────────────────────────────
def navigate(mask: int, max_passes: int = 3) -> None:
    cols, rows = 6, 3  # default grid; adjusted once highlight found

    def _row_reset():
        for _ in range(cols - 1):
            send_command_ex(mask, "LEFT")
            time.sleep(0.08)

    for p in range(max_passes):
        frame = grab_current_frame(mask)
        hi = _find_highlight_rect_strict(frame)
        if hi is None:
            send_command_ex(mask, "DOWN")
            time.sleep(0.3)
            continue

        cols = 5 if hi[2] > 150 else 6  # once per pass is fine
        for r in range(rows):
            for c in range(cols):
                tag = f"{p}_{r}_{c}"
                frame = grab_current_frame(mask)
                rect = _find_highlight_rect_strict(frame)
                if rect is None:
                    break  # highlight lost; punt to outer loop

                x, y, w, h = rect
                tile = frame[y : y + h, x : x + w]

                if _tile_has_watermark(tile, tag) \
                        or _tile_has_icon(tile, tag) \
                        or _label_is_youtube(frame, rect, tag):
                    print(f"[OK] YouTube tile in focus at pass:{p} row:{r} col:{c}")
                    return

                if c < cols - 1:
                    send_command_ex(mask, "RIGHT")
                    time.sleep(0.25)

            _row_reset()
            if r < rows - 1:
                send_command_ex(mask, "DOWN")
                time.sleep(0.28)

    raise AssertionError("YouTube tile not found after navigation")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: focus_youtube_tile.py <PortMask>")

    mask = int(sys.argv[1])
    try:
        navigate(mask)
    except AssertionError as e:
        # capture the last frame for debugging
        try:
            frame = grab_current_frame(mask)
            ts = time.strftime("%Y%m%d_%H%M%S")
            _save_dbg(f"FAIL_{mask}_{ts}.png", frame)
            print(f"[FAIL] saved last frame to dotfinder/FAIL_{mask}_{ts}.png")
        except Exception as save_err:
            print(f"[FAIL] couldn’t save last frame: {save_err!r}")
        print("[FAIL]", e)
        sys.exit(4)

    sys.exit(0)

if __name__ == "__main__":
    main()