#!/usr/bin/env python3
"""
set_timer_from_now.py

Usage (two forms):

  1) Without any history‐IDs:
     python set_timer_from_now.py <PortMask>

  2) With history‐IDs (all five arguments):
     python set_timer_from_now.py \
       <PortMask> <JobRunHistoryID> <UIJobHistoryID> \
       <UITestCaseHistoryID> <UIStepHistoryID>

This script assumes you have already navigated to the “Set Time” dialog (so
the dialog with “Start Time … End Time … Save” is on screen). It will:

  A) Read local time → compute:
       start_time = now + 2 minutes
       end_time   = now + 5 minutes

  B) Enter those times via number keys (no scrolling):
       • type digits for start_time  (HMM or HHMM)
       • press RIGHT
       • type digits for end_time    (HMM or HHMM)
       • press RIGHT, RIGHT, SELECT

  C) Verify (via OCR) that the UI’s Start/End fields now actually match the
     times we computed in (A). If they do not match (or OCR fails), then:
       execute_classifyfailure("FAIL", UIStepHistoryID)
       sys.exit(4)

  On success, prints “Saved …” and exits(0).

If DP_STUDIO is defined (e.g. "http://<studio-ip>:5000"), uses REST. Otherwise
falls back to dp_lib for local debugging.
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
from datetime import datetime, timedelta
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & “dotfinder” debug folder
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = pathlib.Path(__file__).parent
DOTFINDER_DIR = SCRIPT_DIR / "dotfinder"
DOTFINDER_DIR.mkdir(exist_ok=True)

DEBUG_SAVE_ROI = False  # If True, every crop used for OCR will be saved under dotfinder/.

def _save_dbg(filename: str, img: np.ndarray) -> None:
    """
    Save the image under SCRIPT_DIR/"dotfinder"/filename (creating folder if needed).
    """
    out_path = (DOTFINDER_DIR / filename)
    cv2.imwrite(str(out_path), img)


# ─────────────────────────────────────────────────────────────────────────────
#  execute_classifyfailure (run on OCR or verification failure)
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
STUDIO_URL = os.getenv("DP_STUDIO")  # e.g. "http://<studio-ip>:5000"
USER       = os.getenv("DP_USER", "")

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
    Raises if HTTP status != 200.
    """
    url = f"{STUDIO_URL}{endpoint}"
    rsp = _sess().post(url, data=json.dumps(payload), timeout=30)
    rsp.raise_for_status()
    return rsp.json()

def mask_to_single_port(mask: int) -> int:
    """
    Convert a bitmask (1 << (port-1)) into the port index (1–16).
    Raises if mask is zero or not exactly one bit.
    """
    if mask == 0 or (mask & (mask - 1)) != 0:
        raise ValueError("PortMask must have exactly one bit set")
    return mask.bit_length()


def convertPortViewMask2String(mask: str):
    binary_str = format(mask, '08b')
    reversed_binary_str = binary_str[::-1]

    portViewString = ''.join(
        str(idx + 1) for idx, bit in enumerate(reversed_binary_str) if bit == '1')

    print("Binary string:", binary_str)
    print("Reversed binary string:", reversed_binary_str)
    print("Resultant string:", portViewString)

    return portViewString


def send_command(mask: int, command: str):
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



def grab_current_frame(mask: int) -> np.ndarray:
    """
    Request a snapshot (operation=3). DVR writes PNG to a shared UNC path.
    Returns the loaded image as a BGR numpy array.
    """
    port = mask_to_single_port(mask)
    tmp_png = fr"\\{os.getenv('COMPUTERNAME','localhost')}\share\snaps\{uuid.uuid4()}.png"
    payload = {
        "PortView":         1 << (port - 1),
        "Operation":        3,
        "OutputFilePath":   tmp_png,
        "xLeftTopRectangle": 0, "yLeftTopRectangle": 0,
        "RectangleWidth":   0, "RectangleHeight": 0,
        "RecordDuration":   0, "Threshold": 0,
        "Duration":         0, "MonitorDuration": 0,
        "AudioCutOffThreshold": 0
    }
    result = _post("/api/v4/video", payload)
    if result.get("ResultCode") != 0 or not result.get("ResultFilePath"):
        raise RuntimeError(f"Snapshot failed → {result}")
    img = cv2.imread(result["ResultFilePath"])
    if img is None:
        raise IOError(f"Cannot read snapshot file {result['ResultFilePath']}")
    return img

# If DP_STUDIO is not set, fall back to dp_lib for local debugging
if not _studio_on():
    import dp_lib as _gng
    send_command        = lambda p, c: _gng.send_command(str(p), c)
    grab_current_frame  = lambda m: _gng.grab_current_frame(m)
    mask_to_single_port = _gng.mask_to_single_port


# ─────────────────────────────────────────────────────────────────────────────
#  Key‐map: step‐code → Studio UISets (same as other scripts)
# ─────────────────────────────────────────────────────────────────────────────
_CMD_TO_UISET = {
    "UP":                  "UP",
    "DOWN":                "DOWN",
    "LEFT":                "LEFT",
    "RIGHT":               "RIGHT",
    "SELECT":              "SELECT",
    "OPTIONS":             "OPTIONS",
    "LIVE":                "LIVE",            # used for CMD_LIVE_TV
    "RESET_USER_SETTINGS": "RESET_USER_SETTINGS",
    "DVR":                 "DVR",             # used for CMD_DVR
}


# ─────────────────────────────────────────────────────────────────────────────
#  Vision‐helper: crop + OCR the Start/End time fields for verification
# ─────────────────────────────────────────────────────────────────────────────
def _verify_times(frame: np.ndarray,
                  expected_start: str,
                  expected_end: str,
                  dbg_prefix: str | None = None) -> bool:
    """
    Verifies:
      • The Start and End fields match expected_start / expected_end
      • The Duration line reads “Duration N minutes”
    """
    import re
    from datetime import datetime

    h, w = frame.shape[:2]

    # 1) Rough crop of the dialog’s two halves
    y0, y1 = int(0.58 * h), int(0.67 * h)
    x0_s, x1_s = int(0.39 * w), int(0.57 * w)
    x0_e, x1_e = int(0.56 * w), int(0.75 * w)
    start_roi = frame[y0:y1, x0_s:x1_s]
    end_roi   = frame[y0:y1, x0_e:x1_e]

    # 2) Shrink away thick borders
    def shrink(roi, pct=0.08):
        rh, rw = roi.shape[:2]
        mh, mw = int(rh * pct), int(rw * pct)
        return roi[mh:rh-mh, mw:rw-mw]

    start_roi, end_roi = shrink(start_roi), shrink(end_roi)
    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_start_crop.png", start_roi)
        _save_dbg(f"{dbg_prefix}_end_crop.png",   end_roi)

    # 3) OCR helpers
    def ocr_digits(roi: np.ndarray, tag: str="") -> tuple[list[str], float]:
        roi2 = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        g    = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        g    = cv2.GaussianBlur(g, (5,5), 0)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        bw   = cv2.dilate(bw, kern, iterations=1)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(bw)
        clean = np.zeros_like(bw)
        for lbl in range(1, n):
            if stats[lbl, cv2.CC_STAT_AREA] > 100:
                clean[labels == lbl] = 255
        bw = clean

        if dbg_prefix:
            _save_dbg(f"{dbg_prefix}_{tag}_bw.png", bw)

        cfg  = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789"
        data = pytesseract.image_to_data(bw,
                                         output_type=pytesseract.Output.DICT,
                                         config=cfg)
        toks  = [t for t in data["text"] if t.strip().isdigit()]
        confs = [int(c) for c in data["conf"]
                 if str(c).isdigit() and int(c) >= 0]
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        return toks, avg_conf

    def ocr_suffix(roi: np.ndarray, tag: str="") -> tuple[str, float]:
        g    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        bw   = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kern, iterations=1)

        if dbg_prefix:
            _save_dbg(f"{dbg_prefix}_{tag}_bw.png", bw)

        cfg  = "--oem 1 --psm 7 -c tessedit_char_whitelist=APM"
        data = pytesseract.image_to_data(bw,
                                         output_type=pytesseract.Output.DICT,
                                         config=cfg)
        txts  = [t for t in data["text"] if t.strip()]
        confs = [int(c) for c in data["conf"]
                 if str(c).isdigit() and int(c) >= 0]
        avg_conf = (sum(confs) / len(confs)) if confs else 0.0
        return "".join(txts).upper(), avg_conf

    # 4) Parse one time field (H:MM SU)
    def parse_time(roi: np.ndarray, tag: str) -> tuple[str, float]:
        rh, rw = roi.shape[:2]
        # widen the hour box slightly to avoid clipping "11"
        x_hr_end = int(rw * (0.30 if tag=="end" else 0.22))
        x_mn_end = int(rw * 0.60)

        hr_roi = roi[:, :x_hr_end]
        mn_roi = roi[:, x_hr_end:x_mn_end]
        su_roi = roi[:, x_mn_end:]

        def ic(r, top_pct=0.05, side_pct=None):
            h2, w2 = r.shape[:2]
            mh = int(h2 * top_pct)
            if side_pct is None:
                side_pct = 0.10 if tag=="end" else 0.15
            mw = int(w2 * side_pct)
            return r[mh:h2-mh, mw:w2-mw]

        hr = ic(hr_roi)
        mn = ic(mn_roi)
        su = ic(su_roi)

        hrs, cf_h = ocr_digits(hr, f"{tag}_hr")
        mns, cf_m = ocr_digits(mn, f"{tag}_mn")
        sfx, cf_s = ocr_suffix(su, f"{tag}_su")

        if dbg_prefix:
            _save_dbg(f"{dbg_prefix}_{tag}_hr.png", hr)
            _save_dbg(f"{dbg_prefix}_{tag}_mn.png", mn)
            _save_dbg(f"{dbg_prefix}_{tag}_su.png", su)

        try:
            hval = int(hrs[0]); mval = int(mns[0])
            txt  = f"{hval}:{mval:02d} {sfx}"
        except:
            txt = ""
        conf = min(cf_h, cf_m, cf_s)
        return txt, conf

    # perform OCR
    start_text, start_cf = parse_time(start_roi, "start")
    end_text,   end_cf   = parse_time(end_roi,   "end")

    if dbg_prefix:
        print(f"[FINAL OCR] start → “{start_text}” conf={start_cf:.0f}")
        print(f"[FINAL OCR] end   → “{end_text}”   conf={end_cf:.0f}")

    # 5) low‐confidence bail
    if start_cf < 0 or end_cf < 0:
        return False

    # 6) exact‐match start & end
    def pat(s: str):
        m = re.match(r"(\d{1,2}:\d{2})\s*([AP]M)", s)
        return re.compile(rf"\b{re.escape(m.group(1))}\s*{m.group(2)}\b")

    if not (pat(expected_start).search(start_text)
            and pat(expected_end).search(end_text)):
        return False

    # 7) verify duration (in minutes)
    def to_dt(s: str):
        hh_mm, suf = s.split()
        h, m = map(int, hh_mm.split(":"))
        if suf=="PM" and h<12: h+=12
        if suf=="AM" and h==12: h=0
        return datetime(2000,1,1,h,m)

    mins = int((to_dt(expected_end) - to_dt(expected_start)).total_seconds() // 60)

    y2, y3        = int(0.67*h), int(0.75*h)
    x0_d, x1_d    = int(0.40*w), int(0.75*w)
    dur_roi       = frame[y2:y3, x0_d:x1_d]
    if dbg_prefix:
        _save_dbg(f"{dbg_prefix}_dur_crop.png", dur_roi)

    gray = cv2.cvtColor(dur_roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cfg  = "--oem 3 --psm 6"
    raw  = pytesseract.image_to_string(bw, config=cfg).strip().upper()
    if dbg_prefix:
        print(f"[OCR {dbg_prefix or 'dur'}] → {raw!r}")

    return bool(re.search(rf"\bDURATION\s+{mins}\s+MINUTES\b", raw))


def _get_current_times(frame: np.ndarray) -> tuple[int,int,str,int,int,str]:
    # Crop exactly as in _verify_times
    h, w = frame.shape[:2]
    y0, y1 = int(0.58*h), int(0.65*h)
    x0_s, x1_s = int(0.40*w), int(0.57*w)
    x0_e, x1_e = int(0.56*w), int(0.75*w)
    start_roi = frame[y0:y1, x0_s:x1_s]
    end_roi   = frame[y0:y1, x0_e:x1_e]

    def shrink(roi):
        pct = 0.08
        rh, rw = roi.shape[:2]
        mh, mw = int(rh*pct), int(rw*pct)
        return roi[mh:rh-mh, mw:rw-mw]

    s = shrink(start_roi)
    e = shrink(end_roi)

    def ocr_box(roi, tag):
        # split into thirds exactly as parse()
        rh, rw = roi.shape[:2]
        x_hr = int(rw * (0.22 if tag=="end" else 0.22))
        x_mn = int(rw*0.60)
        hr = roi[:, :x_hr]
        mn = roi[:, x_hr:x_mn]
        su = roi[:, x_mn:]
        # inner‐crop
        def ic(r):
            h2,w2 = r.shape[:2]
            m2h,m2w = int(h2*0.05), int(w2*0.15)
            return r[m2h:h2-m2h, m2w:w2-m2w]
        hr, mn, su = ic(hr), ic(mn), ic(su)
        # OCR digits for hr & mn
        def digit(roi):
            # very tight digits only
            cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789"
            g = cv2.cvtColor(cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                             cv2.COLOR_BGR2GRAY)
            _,bw = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            d = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=cfg)
            toks = [t for t in d["text"] if t.strip().isdigit()]
            return int(toks[0]) if toks else 0
        hh = digit(hr)
        mm = digit(mn)
        # suffix
        cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=APM"
        g2 = cv2.cvtColor(su, cv2.COLOR_BGR2GRAY)
        _,bw2 = cv2.threshold(g2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        d2 = pytesseract.image_to_data(bw2, output_type=pytesseract.Output.DICT, config=cfg)
        txts = [t for t in d2["text"] if t.strip()]
        suffix = txts[0].upper() if txts else "AM"
        return hh, mm, suffix

    sh, sm, ssuf = ocr_box(s, "start")
    eh, em, esuf = ocr_box(e, "end")
    return sh, sm, ssuf, eh, em, esuf


# ────────────────────────────────────────────────────────────────
#  NEW  helper – find which widget is highlighted
#     returns an enum 0-6  (see _FOCUS_ENUM below)
# ────────────────────────────────────────────────────────────────
_FOCUS_ENUM = (
    "START_HR", "START_MN", "START_SU",
    "END_HR",   "END_MN",   "END_SU",
    "SAVE"
)

def _red_mask_strict(bgr: np.ndarray) -> np.ndarray:
    """
    Strict HSV mask for the bright red highlight border.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array((0, 150, 150))
    upper1 = np.array((10, 255, 255))
    lower2 = np.array((160, 150, 150))
    upper2 = np.array((180, 255, 255))
    m1 = cv2.inRange(hsv, lower1, upper1)
    m2 = cv2.inRange(hsv, lower2, upper2)
    return cv2.bitwise_or(m1, m2)


def _detect_focus(frame: np.ndarray) -> int | None:
    """
    Locate the single red highlight rectangle and
    map it to one of the seven logical widgets:
         0 1 2   3 4 5   6
        [H][M][A] [H][M][A]   SAVE▷
    Returns index 0–6 or None if nothing is found.
    """
    h, w = frame.shape[:2]

    # pre‐computed logical boxes on a 1920×1080, will scale
    base_w, base_h = 1920, 1080
    sx, sy = w / base_w, h / base_h
    boxes = [
        ( 780, 630,  840, 700),   # START_HR
        ( 860, 630,  945, 700),   # START_MN
        ( 950, 630, 1015, 700),   # START_SU
        (1080, 630, 1155, 700),   # END_HR
        (1170, 630, 1255, 700),   # END_MN
        (1260, 630, 1325, 700),   # END_SU
        (1390, 625, 1495, 715),   # SAVE
    ]
    # scale them to the actual resolution
    boxes = [
        (
            int(x1 * sx), int(y1 * sy),
            int(x2 * sx), int(y2 * sy),
        )
        for x1, y1, x2, y2 in boxes
    ]

    # make one big red‐border mask
    red_mask = _red_mask_strict(frame)

    # debug: dump the mask
    _save_dbg("detect_focus_redmask.png", red_mask)

    # see which box has significant red pixels
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        roi = red_mask[y1:y2, x1:x2]
        ratio = cv2.countNonZero(roi) / float(roi.size)
        #print(f"[_detect_focus] box={idx}({_FOCUS_ENUM[idx]}) → red‐ratio={ratio:.2%}")
        if ratio > 0.03:
            #print(f"[_detect_focus] → focus detected on {_FOCUS_ENUM[idx]}")
            return idx

    print("[_detect_focus] → no focus detected")
    return None

# ────────────────────────────────────────────────────────────────
#  NEW  helper – move highlight to a target widget safely
# ────────────────────────────────────────────────────────────────
def _move_to(mask: int, target_idx: int, max_hops: int = 10) -> None:
    """
    Arrow‐keys until the highlight sits on `target_idx` (0–6 per _FOCUS_ENUM).
    Dumps a debug snapshot each hop and logs every decision.
    """
    port = mask_to_single_port(mask)
    target_name = _FOCUS_ENUM[target_idx]
    for hop in range(max_hops):
        frame = grab_current_frame(mask)
        _save_dbg(f"move_{target_name}_hop{hop}.png", frame)

        pos = _detect_focus(frame)
        pos_name = _FOCUS_ENUM[pos] if pos is not None else "None"
        #print(f"[_move_to] hop={hop}: saw focus={pos!r} ({pos_name}); target={target_idx} ({target_name})")

        if pos == target_idx:
            #print(f"[_move_to] reached {target_name} at hop={hop}")
            return

        # pick direction
        if pos is None or pos > target_idx:
            direction = "LEFT"
        else:
            direction = "RIGHT"
        #print(f"[_move_to] pressing {direction} to move from {pos_name} → {target_name}")
        send_command(port, direction)
        time.sleep(0.12)

    raise RuntimeError(f"Could not move focus to {target_name} after {max_hops} hops")

def _navigate_and_adjust(mask: int,
                         start_now: tuple[int,int,str],
                         end_now:   tuple[int,int,str],
                         start_target: tuple[int,int,str],
                         end_target:   tuple[int,int,str]) -> None:
    """
    Adjust in this order:
      1) START_MIN
      2) START_HR
      3) START_SU
      4) END_MN
      5) END_HR
      6) END_SU
      7) SAVE

    Always re-OCR the current value just before each field, and
    take the shortest path around the dial.
    """
    port = mask_to_single_port(mask)

    def shortest_delta(current: int, target: int, modulus: int):
        """
        Returns (delta_steps, direction),
        where direction is "UP" or "DOWN", and steps is minimal under wrap.
        """
        forward = (target - current) % modulus
        backward = (current - target) % modulus
        if forward <= backward:
            return forward, "UP"
        else:
            return backward, "DOWN"

    def refresh():
        frame = grab_current_frame(mask)
        return _get_current_times(frame)

    # 1) START_MN
    sh, sm, ssuf, eh, em, esuf = refresh()
    t_s_hr, t_s_mn, t_s_suf = start_target

    if sm != t_s_mn:
        _move_to(mask, 1)

        # compute wrap-aware distances
        up   = (t_s_mn - sm) % 60
        down = (sm    - t_s_mn) % 60

        # pick the shorter direction
        if up <= down:
            key, steps = "UP", up
        else:
            key, steps = "DOWN", down

        for i in range(1, steps+1):
            print(f"[ADJUST] START_MN: pressing {key} ({i}/{steps})")
            send_command(port, key)
            time.sleep(0.05)
    else:
        print(f"[SKIP] START_MN already at {sm}")

    # 2) START_HR
    sh, sm, ssuf, eh, em, esuf = refresh()
    if sh != t_s_hr:
        _move_to(mask, 0)  # START_HR

        # compute wrap‐aware distances on a 1–12 dial
        up   = (t_s_hr - sh) % 12
        down = (sh    - t_s_hr) % 12

        if up <= down:
            key, steps = "UP", up
        else:
            key, steps = "DOWN", down

        for i in range(1, steps+1):
            print(f"[ADJUST] START_HR: pressing {key} ({i}/{steps})")
            send_command(port, key)
            time.sleep(0.05)
    else:
        print(f"[SKIP] START_HR already at {sh}")

    # 3) START_SU
    sh, sm, ssuf, eh, em, esuf = refresh()
    if ssuf != t_s_suf:
        #print(f"[NAV] Moving to START_SU (current={ssuf}, target={t_s_suf})")
        _move_to(mask, 2)
        print("[ADJUST] START_SU: toggling AM/PM")
        send_command(port, "UP"); time.sleep(0.05)
    else:
        print(f"[SKIP] START_SU already {ssuf}")

    # 4) END_MN
    sh, sm, ssuf, eh, em, esuf = refresh()
    t_e_hr, t_e_mn, t_e_suf = end_target

    if em != t_e_mn:
        _move_to(mask, 4)

        # compute wrap-aware distances
        up   = (t_e_mn - em) % 60
        down = (em   - t_e_mn) % 60

        # pick the shorter direction
        if up <= down:
            key, steps = "UP", up
        else:
            key, steps = "DOWN", down

        for i in range(1, steps+1):
            print(f"[ADJUST] END_MN: pressing {key} ({i}/{steps})")
            send_command(port, key)
            time.sleep(0.05)
    else:
        print(f"[SKIP] END_MN already at {em}")


    # 5) END_HR
    sh, sm, ssuf, eh, em, esuf = refresh()
    if eh != t_e_hr:
        _move_to(mask, 3)  # END_HR

        # same wrap logic on 1–12
        up   = (t_e_hr - eh) % 12
        down = (eh    - t_e_hr) % 12

        if up <= down:
            key, steps = "UP", up
        else:
            key, steps = "DOWN", down

        for i in range(1, steps+1):
            print(f"[ADJUST] END_HR: pressing {key} ({i}/{steps})")
            send_command(port, key)
            time.sleep(0.05)
    else:
        print(f"[SKIP] END_HR already at {eh}")

    # 6) END_SU
    sh, sm, ssuf, eh, em, esuf = refresh()
    if esuf != t_e_suf:
        #print(f"[NAV] Moving to END_SU (current={esuf}, target={t_e_suf})")
        _move_to(mask, 5)
        print("[ADJUST] END_SU: toggling AM/PM")
        send_command(port, "UP"); time.sleep(0.05)
    else:
        print(f"[SKIP] END_SU already {esuf}")

    # 7) back to SAVE
    #print("[NAV] Moving back to SAVE")
    _move_to(mask, 6)
    print("[ADJUST] Focus is now on SAVE")

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRYPOINT (main)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import sys, time
    from datetime import datetime, timedelta

    # 1) Parse args
    argc = len(sys.argv)
    if argc not in (2, 6):
        print("Usage: set_timer_from_now.py <PortMask> [<JobRunHistoryID> <UIJobHistoryID> <UITestCaseHistoryID> <UIStepHistoryID>]")
        sys.exit(1)

    try:
        mask = int(sys.argv[1])
    except ValueError:
        print("PortMask must be an integer (e.g. 1, 2, 4, 8, …).")
        sys.exit(1)

    UIStepHistoryID = None
    if argc == 6:
        _, JobRunHistoryID, UIJobHistoryID, UITestCaseHistoryID, UIStepHistoryID = sys.argv

    # 2) Compute targets (now+2 → start, now+5 → end)
    now      = datetime.now()
    start_dt = now + timedelta(minutes=3)
    end_dt   = now + timedelta(minutes=15)

    def to_tuple(dt):
        h = dt.hour % 12 or 12
        suf = "AM" if dt.hour < 12 else "PM"
        return (h, dt.minute, suf)

    start_target = to_tuple(start_dt)
    end_target   = to_tuple(end_dt)

    expected_start_text = f"{start_target[0]}:{start_target[1]:02d} {start_target[2]}"
    expected_end_text   = f"{end_target[0]}:{end_target[1]:02d} {end_target[2]}"

    # 3) Read current on‐screen values
    frame = grab_current_frame(mask)
    sh, sm, ssuf, eh, em, esuf = _get_current_times(frame)

    # 4) Adjust every widget (End H/M/SU then Start H/M/SU)
    _navigate_and_adjust(
        mask,
        (sh, sm, ssuf),
        (eh, em, esuf),
        start_target,
        end_target
    )

    # 5) Verification: capture a fresh frame, OCR Start/End fields
    max_retries = 1
    for attempt in range(max_retries + 1):
        frame = grab_current_frame(mask)
        if _verify_times(frame, expected_start_text, expected_end_text, dbg_prefix="verify"):
            print(f"[OK] Timer set: {expected_start_text} → {expected_end_text}")
            sys.exit(0)

        if attempt == max_retries:
            print(f"\n[FAILED] Start/End time verification failed after retry.\n"
                  f"  • expected start: {expected_start_text}\n"
                  f"  • expected end:   {expected_end_text}\n")
            if UIStepHistoryID:
                execute_classifyfailure("FAIL", UIStepHistoryID)
            sys.exit(4)

        # otherwise, compute current vs target and adjust again
        sh, sm, ssuf, eh, em, esuf = _get_current_times(frame)
        print(f"[RETRY] Adjusting from {sh:02d}:{sm:02d}{ssuf}→{expected_start_text} and "
              f"{eh:02d}:{em:02d}{esuf}→{expected_end_text}")
        _navigate_and_adjust(
            mask,
            (sh, sm, ssuf),
            (eh, em, esuf),
            start_target,
            end_target
        )

    # 6) Final verification (just in case)
    frame = grab_current_frame(mask)
    if _verify_times(frame, expected_start_text, expected_end_text, dbg_prefix="verify"):
        print(f"[OK] Timer set: {expected_start_text} → {expected_end_text}")
        sys.exit(0)
    else:
        print(f"\n[FAILED] Start/End time verification failed.\n"
              f"  • expected start: {expected_start_text}\n"
              f"  • expected end:   {expected_end_text}\n")
        if UIStepHistoryID:
            execute_classifyfailure("FAIL", UIStepHistoryID)
        sys.exit(4)


if __name__ == "__main__":
    main()
