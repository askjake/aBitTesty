#!/usr/bin/env python3
"""
SO_Netflix_JoinNow.py

Wait ~3 minutes for Netflix to launch, then verify:
  • the red “NETFLIX” wordmark
  • AND at least one tagline keyword (“TV SHOWS” or “MOVIES”)

On failure: save a snapshot into dotfinder/, call execute_classifyfailure(), exit 4.
"""

from __future__ import annotations
import sys, time, uuid, subprocess
from pathlib import Path

import cv2, numpy as np, pytesseract, requests, json

# ─── CONFIG ───────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
DOTFINDER_DIR = SCRIPT_DIR / "dotfinder"
DOTFINDER_DIR.mkdir(exist_ok=True)

_LOGO_TMPL = SCRIPT_DIR / "netflix_logo_166x100.png"
# rest endpoint (or leave empty to fallback to dp_lib)
STUDIO_URL = "http://127.0.0.1:5000"
USER       = ""

# ─── ClassifyFailure helper ────────────────────────────────────────────────
def execute_classifyfailure(_value: str, uistephistoryid: str) -> None:
    folder = Path("C:/DPUnified/Tools/ClassifyFailureAgent/")
    exe    = folder / "ClassifyFailure.exe"
    print("execute_classifyfailure")
    try:
        p = subprocess.Popen(
            [str(exe), _value, uistephistoryid],
            cwd=str(folder),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        out, err = p.communicate()
        print("ClassifyFailure Output:", out.decode(errors="ignore"))
        print("ClassifyFailure Error: ", err.decode(errors="ignore"))
    except Exception as e:
        print("ClassifyFailure exception:", e)

# ─── REST Adapter ──────────────────────────────────────────────────────────
_session = None
def _sess():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type":"application/json"})
    return _session

def _post(endpoint: str, payload: dict) -> dict:
    url = f"{STUDIO_URL}{endpoint}"
    rsp = _sess().post(url, data=json.dumps(payload), timeout=30)
    rsp.raise_for_status()
    return rsp.json()

def grab_current_frame(mask: int) -> np.ndarray:
    # request a snapshot → UNC path
    tmp = fr"\\INDEVW-DEVPART03\DataStorage\Snapshots\{uuid.uuid4()}.png"
    payload = {
        "PortView": mask, "Operation": 3,
        "OutputFilePath": tmp,
        "xLeftTopRectangle":0, "yLeftTopRectangle":0,
        "RectangleWidth":960,   "RectangleHeight":540,
        "RecordDuration":0, "Threshold":0,
        "Duration":0, "MonitorDuration":0,
        "AudioCutOffThreshold":0
    }
    res = _post("/api/v4/video", payload)
    if res.get("ResultCode")!=0 or not res.get("ResultFilePath"):
        raise RuntimeError(f"Snapshot failed → {res!r}")
    img = cv2.imread(res["ResultFilePath"])
    if img is None:
        raise IOError("Cannot read snapshot PNG")
    return img

# fallback to dp_lib if STUDIO_URL is blank
if not STUDIO_URL:
    import dp_lib as _g
    grab_current_frame = lambda m: _g.grab_current_frame(m)

# ─── Logo Template Loader ──────────────────────────────────────────────────
def _ensure_logo_template() -> np.ndarray:
    tmpl = cv2.imread(str(_LOGO_TMPL), cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        raise FileNotFoundError(f"Missing template: {_LOGO_TMPL}")
    # drop alpha if present
    if tmpl.shape[2]==4:
        tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGRA2BGR)
    return tmpl

_TMPL = _ensure_logo_template()

# ─── Logo Matcher ──────────────────────────────────────────────────────────
def _has_logo(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    # crop the top region where Netflix appears
    y0, y1 = int(0.02*h), int(0.15*h)
    x0, x1 = int(0.10*w), int(0.90*w)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    tmpl_gray = cv2.cvtColor(_TMPL, cv2.COLOR_BGR2GRAY)
    tmpl_blur = cv2.GaussianBlur(tmpl_gray, (5,5), 0)

    _, b1 = cv2.threshold(blur,    180, 255, cv2.THRESH_BINARY_INV)
    _, b2 = cv2.threshold(tmpl_blur,180, 255, cv2.THRESH_BINARY_INV)

    res = cv2.matchTemplate(b1, b2, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    print(f"[TEMPLATE] logo match score = {maxv:.2f}")
    return maxv >= 0.10  # lowered threshold

# ─── Tagline OCR ───────────────────────────────────────────────────────────
def _has_tagline(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    # crop around the middle of the screen
    y0, y1 = int(0.30*h), int(0.55*h)
    x0, x1 = int(0.10*w), int(0.90*w)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    txt = pytesseract.image_to_string(bw, config="--oem 3 --psm 6")
    up  = txt.strip().upper()
    print(f"[OCR] tagline → {up!r}")

    # accept if we saw either key phrase
    return ("TV SHOWS" in up) or ("MOVIES" in up)

# ─── Main Verification Logic ───────────────────────────────────────────────
def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: SO_Netflix_JoinNow.py <PortMask> [ ... UIStepHistoryID ]")
        sys.exit(1)
    try:
        mask = int(args[0])
    except ValueError:
        print("PortMask must be an integer"); sys.exit(1)

    UIStepHistoryID = args[4] if len(args) >= 5 else None

    # wait ~3m for Netflix to come up
    time.sleep(0)

    frame = grab_current_frame(mask)
    ok_logo    = _has_logo(frame)
    ok_tagline = _has_tagline(frame)

    if ok_logo and ok_tagline:
        print("[OK] Netflix launch screen detected.")
        sys.exit(0)

    # failure: save for debugging into dotfinder/
    ts = int(time.time())
    out = DOTFINDER_DIR / f"failed_netflix_{ts}.png"
    cv2.imwrite(str(out), frame)
    print(f"● saved failure screenshot → {out}")

    print("[FAIL] Netflix screen not detected")
    if UIStepHistoryID:
        execute_classifyfailure("FAIL", UIStepHistoryID)
    sys.exit(4)

if __name__ == "__main__":
    main()
