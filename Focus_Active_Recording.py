#!/usr/bin/env python3
"""
Focus_Active_Recording.py

Given exactly five arguments:

  1) PortMask
  2) JobRunHistoryID
  3) UIJobHistoryID
  4) UITestCaseHistoryID
  5) UIStepHistoryID

This script will:

  1) Repeatedly snapshot the DVR UI
  2) Detect the red highlight border
  3) Within that tile, look for a red “REC” dot + OCR “Recording”
  4) If not found, arrow‐key around until we find it
  5) As soon as the recording tile is confirmed, press SELECT once

If it cannot locate a recording after the allowed passes, it will call:
    execute_classifyfailure("FAIL", <UIStepHistoryID>)
and then exit with code 4.

Usage:
    DP_STUDIO="http://<studio-ip>:5000" \
      python Find_Active_Recording.py \
        <PortMask> <JobRunHistoryID> <UIJobHistoryID> \
        <UITestCaseHistoryID> <UIStepHistoryID>

If `DP_STUDIO` is unset, it will fall back to `dp_lib` (for local debugging).
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import pathlib
import json
import cv2
import numpy as np
import requests
import pytesseract
import re
import subprocess
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & “dotfinder” debug folder
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
DOTFINDER_DIR = SCRIPT_DIR / "dotfinder"
DOTFINDER_DIR.mkdir(exist_ok=True)

DEBUG_SAVE_ROI = False  # set True if you want every ROI crop saved


# ——— pull off optional “-f FILTER_TEXT” ———
filter_text = "KWGN"
if "-f" in sys.argv:
    idx = sys.argv.index("-f")
    if idx == len(sys.argv) - 1:
        print("Error: -f requires a filter text")
        sys.exit(1)
    filter_text = sys.argv[idx+1]
    # remove both from argv so the rest of your logic (argc checks) still works
    del sys.argv[idx:idx+2]

def _save_dbg(filename: str, img: np.ndarray) -> None:
    """
    Save the image under SCRIPT_DIR/"dotfinder"/filename.
    """
    out_path = (DOTFINDER_DIR / filename)
    cv2.imwrite(str(out_path), img)


# ─────────────────────────────────────────────────────────────────────────────
#  execute_classifyfailure (on any failure)
# ─────────────────────────────────────────────────────────────────────────────
def execute_classifyfailure(_value: str, uistephistoryid: str) -> None:
    """
    Invoke the external ClassifyFailure.exe tool with (_value, uistephistoryid).
    """
    print("execute_classifyfailure")
    folder_path = "C:/DPUnified/Tools/ClassifyFailureAgent/"
    try:
        exe_path = folder_path + "ClassifyFailure.exe"
        arguments = [_value, uistephistoryid]
        proc = subprocess.Popen(
            [exe_path] + arguments,
            cwd=folder_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = proc.communicate()
        print("ClassifyFailure Output:", out.decode("utf-8"))
        print("ClassifyFailure Error: ", err.decode("utf-8"))
    except Exception as e:
        print("ClassifyFailure exception:", str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  REST ADAPTER  (Studio)  –  or fallback to dp_lib
# ─────────────────────────────────────────────────────────────────────────────
STUDIO_URL = "http://127.0.0.1:5000"  # os.getenv("DP_STUDIO")  # e.g. "http://<studio-ip>:5000"
USER = ""  # os.getenv("DP_USER", "")

def _ocr_pane_full(frame: np.ndarray, dbg_prefix: str | None = None) -> str:
    """
    Crop the right‐pane (75→100% W, 20→90% H), preprocess aggressively,
    OCR the entire block, return the uppercase string.
    """
    h, w = frame.shape[:2]
    x0, y0 = int(0.75*w), int(0.20*h)
    y1     = int(0.90*h)
    pane   = frame[y0:y1, x0:w]
    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_full.png", frame)
        _save_dbg(f"{dbg_prefix}_pane.png", pane)

    # upscale + gray
    roi = cv2.resize(pane, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE + blur + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    blur  = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # close
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw   = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern, iterations=1)
    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_bin.png", bw)

    # OCR letters+digits+punctuation
    cfg = (
        "--oem 3 --psm 6 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-,:. "
        "-c user_defined_dpi=300"
    )
    raw = pytesseract.image_to_string(bw, config=cfg)
    result = raw.strip().upper()
    print(f"[OCR {dbg_prefix or 'pane'}] → {result!r}")
    return result

def _studio_on() -> bool:
    return bool(STUDIO_URL)


_session: requests.Session | None = None


def _sess() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type": "application/json"})
    return _session


def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    POST to STUDIO_URL + endpoint with JSON payload, return parsed JSON.
    Raises if status code != 200.
    """
    url = f"{STUDIO_URL}{endpoint}"
    rsp = _sess().post(url, data=json.dumps(payload), timeout=30)
    rsp.raise_for_status()
    return rsp.json()


def mask_to_single_port(mask: int) -> int:
    """
    Convert a bitmask (1 << (port-1)) into the port index (1–16).
    Raises if mask is zero or not a single bit.
    """
    if mask == 0 or (mask & (mask - 1)) != 0:
        raise ValueError("PortMask must have exactly one bit set")
    return mask.bit_length()


def send_command(mask: int, cmd: str) -> None:
    """
    Send a one-button “UISet” command via REST (or dp_lib fallback).
    Valid commands: digits "0"…"9" or any key in _CMD_TO_UISET.
    """
    ui_name = _CMD_TO_UISET.get(cmd, cmd)

    payload = {
        "UISetName": ui_name,
        "User": USER,
        "PortViews": mask
    }
    result = _post("/api/v1/ui-set", payload)
    if result.get("ResultCode") != 0:
        raise RuntimeError(f"UISet '{ui_name}' failed → {result}")


def grab_current_frame(mask: int) -> np.ndarray:
    """
    Request a snapshot via REST (operation=3). The DVR writes PNG to a shared UNC path.
    Returns the loaded image as a BGR numpy array.
    """
    # print(f"Mask: {mask}")
    # port = mask_to_single_port(mask)
    # NOTE: adjust this UNC path to wherever Studio can write snapshots
    # tmp_png = fr"\\{os.getenv('COMPUTERNAME','localhost')}\Snapshot\{uuid.uuid4()}.png"
    tmp_png = fr"\\INDEVW-DEVPART03\DataStorage\Snapshots\{uuid.uuid4()}.png"

    payload = {
        "PortView": mask,
        "Operation": 3,
        "OutputFilePath": tmp_png,
        "xLeftTopRectangle": 0, "yLeftTopRectangle": 0,
        "RectangleWidth": 960, "RectangleHeight": 540,
        "RecordDuration": 0, "Threshold": 0,
        "Duration": 0, "MonitorDuration": 0,
        "AudioCutOffThreshold": 0
    }
    result = _post("/api/v4/video", payload)
    if result.get("ResultCode") != 0 or not result.get("ResultFilePath"):
        raise RuntimeError(f"Snapshot failed → {result}")
    img = cv2.imread(result["ResultFilePath"])
    if img is None:
        raise IOError(f"Cannot read PNG at {result['ResultFilePath']}")
    return img


# Fallback to dp_lib if DP_STUDIO is not set
if not _studio_on():
    import dp_lib as _gng

    send_command = lambda p, c: _gng.send_command(str(p), c)
    grab_current_frame = lambda m: _gng.grab_current_frame(m)
    mask_to_single_port = _gng.mask_to_single_port

# ─────────────────────────────────────────────────────────────────────────────
#  Key‐map: map from step‐code → Studio UISets
# ─────────────────────────────────────────────────────────────────────────────
_CMD_TO_UISET = {
    "UP": "CmdUp",
    "DOWN": "CmdDown",
    "LEFT": "CmdLeft",
    "RIGHT": "CmdRight",
    "SELECT": "CmdSelect",
    "OPTIONS": "CmdOptions",
    "LIVE": "CmdLiveTV",  # used for CMD_LIVE_TV
    "RESET_USER_SETTINGS": "RESET_USER_SETTINGS",
    "DVR": "CmdDVR",  # used for CMD_DVR
}


def _pane_has_text(frame: np.ndarray,
                   search: str,
                   dbg_prefix: str | None = None) -> bool:
    """
    Crop the right‐hand info pane, isolate the top channel line, aggressively
    preprocess & OCR it, and return True if `search` appears.
    """
    h, w = frame.shape[:2]
    x0, y0 = int(0.75*w), int(0.25*h)    # you said you dialed this in
    pane = frame[y0:h, x0:w]

    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_full.png", frame)
        _save_dbg(f"{dbg_prefix}_text_crop.png", pane)

    # 1) Grab only the top ~20% for the channel line
    top_h = int(0.20 * pane.shape[0])
    channel_roi = pane[0:top_h, :]
    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_channel_crop.png", channel_roi)

    # 2) Upscale for resolution
    roi2 = cv2.resize(channel_roi, None, fx=2.0, fy=2.0,
                      interpolation=cv2.INTER_CUBIC)

    # 2) Grayscale + fixed threshold (bright text on dark background)
    gray = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, binar = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Tiny open to knock out specks, then close to fill glyph gaps
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    binar = cv2.morphologyEx(binar, cv2.MORPH_OPEN,  kern, iterations=1)
    binar = cv2.morphologyEx(binar, cv2.MORPH_CLOSE, kern, iterations=2)

    # 4) OCR as one line, include comma+colon in whitelist
    cfg = (
            "--oem 3 --psm 7 "
            "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-,: "
            "-c user_defined_dpi=300"
            )
    raw = pytesseract.image_to_string(binar, config=cfg).strip().upper()

    print(f"[OCR {dbg_prefix or 'flt'}] channel text → {raw!r}")

    clean = re.sub(r"[^A-Z0-9 ]+", " ", raw)
    print(f"[CLEAN {dbg_prefix or 'flt'}] → {clean!r}")

  # only match whole words, e.g. 'KCDO' not 'XKCDONY'
    pat = rf"\b{re.escape(search.upper())}\b"
    found = bool(re.search(pat, clean))
    print(f"[MATCH {dbg_prefix or 'flt'}] regex {pat!r} → {found}")

    return found


# ─────────────────────────────────────────────────────────────────────────────
#  Vision Helpers: red‐highlight detection, red‐dot + OCR “Recording”
# ─────────────────────────────────────────────────────────────────────────────
def _crop(frame: np.ndarray, rel: tuple[float, float, float, float] | None) -> np.ndarray:
    if rel is None:
        return frame
    h, w = frame.shape[:2]
    a, b, c, d = rel
    # if any coord >1, interpret as pixel coords
    if any(v > 1 for v in (a, b, c, d)):
        xs = sorted((a, c))
        ys = sorted((b, d))
        x1, x2 = map(int, xs)
        y1, y2 = map(int, ys)
    else:
        x1, y1 = int(a * w), int(b * h)
        x2, y2 = int(c * w), int(d * h)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    return frame[y1:y2, x1:x2]


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return ssim(ga, gb)


def _red_mask_strict(bgr: np.ndarray) -> np.ndarray:
    """
    Strict HSV mask that picks out only very‐bright, very‐saturated reds:
      Hue:   0–10 or 160–180
      Sat:   ≥150
      Val:   ≥150
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array((0, 150, 150))
    upper1 = np.array((10, 255, 255))
    lower2 = np.array((160, 150, 150))
    upper2 = np.array((180, 255, 255))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)


def _find_highlight_rect_strict(frame: np.ndarray, debug_prefix: str = "") -> tuple[int, int, int, int] | None:
    """
    Locate the thin red “highlight” rectangle on the DVR → Recordings screen.
    Returns (x,y,w,h) of that rectangle, or None if nothing is found.

    If debug_prefix != "", saves:
      dotfinder/{debug_prefix}_highlight.png   → frame + green box around matched rect
      dotfinder/{debug_prefix}_no_highlight.png → raw frame if no border found
    """
    mask = _red_mask_strict(frame)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_area = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = (w / h) if h else 0
        # Accept only a “thin but fairly large” red rectangle:
        if area > 800 and 0.4 < aspect < 2.5:
            if area > best_area:
                best_area = area
                best_rect = (x, y, w, h)

    if best_rect:
        x, y, w, h = best_rect
        if debug_prefix:
            dbg_img = frame.copy()
            cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            _save_dbg(f"{debug_prefix}_highlight.png", dbg_img)
        return best_rect

    if debug_prefix:
        _save_dbg(f"{debug_prefix}_no_highlight.png", frame)
    return None


def _has_dot_under_highlight(frame: np.ndarray, rect: tuple[int, int, int, int], debug_prefix: str = "") -> bool:
    """
    Return True if a genuine recording‐dot (round, bright red blob) appears
    in the top‐right of the highlighted tile.

    Steps:
      1) Inset the ROI by 4px from the red outline so that the highlight border
         itself is not included.
      2) Take roughly the top‐right 40%×20% region of that tile (where the dot lives).
      3) Apply a very strict “pure‐red” HSV mask.
      4) Find all contours in that mask; for each:
         – Compute its area A_contour.
         – Compute the minEnclosingCircle → (x_c, y_c, radius).
         – Let A_circle = π * r².
         – Compute circularity = A_contour / A_circle.
         – If (30 ≤ A_contour ≤ 400) and (circularity ≥ 0.70), accept as “REC” dot.
    """
    x, y, w, h = rect
    inset = 4

    # 1) ROI coords (top‐right ~40%×20%), then inset 4px
    x1 = x + int(0.60 * w) + inset
    y1 = y + inset
    x2 = x + w - inset
    y2 = y + int(0.20 * h) - inset

    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return False

    roi = frame[y1:y2, x1:x2]
    if debug_prefix:
        _save_dbg(f"{debug_prefix}_ROI.png", roi)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower1 = np.array((0, 200, 200))
    upper1 = np.array((10, 255, 255))
    lower2 = np.array((160, 200, 200))
    upper2 = np.array((180, 255, 255))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(m1, m2)

    if debug_prefix:
        _save_dbg(f"{debug_prefix}_mask.png", mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 30 or area > 400:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius <= 0:
            continue
        circle_area = np.pi * (radius ** 2)
        circularity = float(area) / float(circle_area)

        if circularity >= 0.70:
            if debug_prefix:
                x_c, y_c, w_c, h_c = cv2.boundingRect(c)
                disp = roi.copy()
                cv2.rectangle(disp, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                _save_dbg(f"{debug_prefix}_dot_cand.png", disp)
            return True

    return False

def _pane_has_recording(frame: np.ndarray,
                        hi_rect: tuple[int, int, int, int],
                        dbg_prefix: str | None = None) -> bool:
    """
    Return True if the right‐hand info pane contains “Recording” (case‐insensitive).
    We crop from 75→100% width, 20→90% height, then apply the same robust
    OCR pipeline as for channel detection.
    """
    h, w = frame.shape[:2]
    x0 = int(0.75 * w)
    y0 = int(0.20 * h)
    y1 = int(0.90 * h)  # crop just below the footer icons
    pane = frame[y0:y1, x0:w]

    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_rec_full.png", frame)
        _save_dbg(f"{dbg_prefix}_rec_pane.png", pane)

    # 1) Upscale & grayscale
    roi = cv2.resize(pane, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 2) CLAHE + blur + Otsu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, bw = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Close to fill broken strokes
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern, iterations=1)

    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_rec_bin.png", bw)

    # 4) OCR full block for letters only
    cfg = "--oem 3 --psm 6 -c tessedit_char_whitelist=RECORDING"
    raw = pytesseract.image_to_string(bw, config=cfg).strip().upper()
    print(f"[OCR {dbg_prefix or 'rec'}] pane text → {raw!r}")

    return "RECORDING" in raw



def convertPortViewMask2String(mask: str):
    binary_str = format(mask, '08b')
    reversed_binary_str = binary_str[::-1]

    portViewString = ''.join(
        str(idx + 1) for idx, bit in enumerate(reversed_binary_str) if bit == '1')

    #print("Binary string:", binary_str)
    #print("Reversed binary string:", reversed_binary_str)
    #print("Resultant string:", portViewString)

    return portViewString


def send_command_ex(mask: int, command: str):
    portview = convertPortViewMask2String(mask)
    url = "http://127.0.0.1:5005/api/execute-irsenderagent2"
    payload = {
        "portviews": portview,
        "keyname": command,
        "username": "",
        "keypressduration": 0,
        "delayduration": 0
    }
    try:
        response = requests.post(url, json=payload)
        print(
            f"Sent command '{command}' to portview {portview}")
    except Exception as e:
        print(f"Error sending command: {e}")


def execute_ui_set(mask: int, uiset: str):
    portview = convertPortViewMask2String(mask)
    url = "http://127.0.0.1:5005/api/execute-uiset"
    payload = {
        "portviews": portview,
        "uisetname": uiset,
        "logname": ""
    }
    try:
        response = requests.post(url, json=payload)
        print(
            f"Sent command set '{uiset}' to portview {portview}")
    except Exception as e:
        print(f"Error sending command: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN NAVIGATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
def navigate_to_recording(mask: int,
                          filter_text: str | None = None,
                          max_passes: int = 3) -> None:
    """
    Keep grabbing frames and scanning each row/col of the DVR grid up to max_passes.
    Only return as soon as we detect:
      • a red “rec” dot under the highlight,
      • the word “Recording” in the right‐pane,
      • AND (if filter_text provided) that filter_text in the pane.
    Otherwise arrow‐key around. Raises AssertionError after max_passes.
    """
    # 1) grab first frame to detect grid shape
    first_frame = grab_current_frame(mask)
    hi_rect = _find_highlight_rect_strict(first_frame, debug_prefix="init")
    if hi_rect is None:
        raise AssertionError("Cannot locate highlight border on initial frame")

    grid_cols = 5 if hi_rect[2] > 150 else 6
    grid_rows = 3

    for nav_pass in range(max_passes):
        time.sleep(0.25)
        attempt = 0

        for row in range(grid_rows):
            for col in range(grid_cols):
                debug_prefix = f"nav_{nav_pass}_{row}_{col}"
                frame = grab_current_frame(mask)

                # find the red highlight border
                rect = _find_highlight_rect_strict(frame, debug_prefix=debug_prefix)
                if rect is None:
                    raise AssertionError(
                        f"Highlight not found (pass={nav_pass}, row={row}, col={col})"
                    )

                # 1) red dot?
                dot_ok = _has_dot_under_highlight(frame, rect, debug_prefix=debug_prefix)

                # 2) filter_text?
                txt_ok = True
                if filter_text:
                    pane_text = _ocr_pane_full(frame, dbg_prefix=f"{debug_prefix}_flt")
                    txt_ok = filter_text.upper() in pane_text

                # if we have both dot + desired text, we’re done
                if dot_ok and txt_ok:
                    msg = f"[INFO ] Found rec‐tile + “{filter_text}” at pass={nav_pass}, row={row}, col={col}"
                    print(msg)
                    return

                # otherwise move right (or wrap to next row)
                if col < grid_cols - 1:
                    send_command_ex(mask, "RIGHT")
                    time.sleep(0.20)
                attempt += 1

            # end of row → move back to col=0, then go down if possible
            for _ in range(grid_cols - 1):
                send_command_ex(mask, "LEFT")
                time.sleep(0.08)
            if row < grid_rows - 1:
                send_command_ex(mask, "DOWN")
                time.sleep(0.25)
            attempt += 1

    # after exhausting all passes:
    raise AssertionError(
        f"Could not locate active recording after {max_passes} passes"
    )



# ─────────────────────────────────────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    """
    Usage:
      focus_and_select_recording.py <PortMask> [-f TEXT] [<JobRunHistoryID> <UIJobHistoryID> <UITestCaseHistoryID> <UIStepHistoryID>] [...]

    You can pass as many extra args after the above as you like—they’ll be ignored.
    """
    # ——— pull off optional “-f FILTER_TEXT” ———
    args = sys.argv[1:]
    filter_text = None
    if "-f" in args:
        idx = args.index("-f")
        if idx == len(args) - 1:
            print("Error: -f requires a filter text")
            sys.exit(1)
        filter_text = args[idx + 1]
        # remove both from args so the rest of your logic is simpler
        del args[idx:idx + 2]

    # now args[0] must be the PortMask
    if not args:
        print("Usage: focus_and_select_recording.py <PortMask> [-f TEXT] [...]")
        sys.exit(1)

    # Parse required PortMask
    try:
        mask = int(args[0])
    except ValueError:
        print("PortMask must be an integer (e.g. 1, 2, 4, 8, …).")
        sys.exit(1)

    # If a UIStepHistoryID was provided in the first 5 args, grab it; else None
    UIStepHistoryID = None
    if len(args) >= 5:
        # args layout: [mask, JobRunHistoryID, UIJobHistoryID, UITestCaseHistoryID, UIStepHistoryID, ...]
        UIStepHistoryID = args[4]

    # 1) Navigate until “Recording” is found
    try:
        navigate_to_recording(mask, filter_text)
    except AssertionError as e:
        print(f"\n[FAILED] {e}\n")
        if UIStepHistoryID is not None:
            execute_classifyfailure("FAIL", UIStepHistoryID)
        sys.exit(4)

    # 2) Press SELECT once to open that recording (if desired)
    #    port = mask_to_single_port(mask)
    #    send_command_ex(port, "ENTER")

    print("[OK] Recording tile FOUND.")
    sys.exit(0)


if __name__ == "__main__":
    main()
