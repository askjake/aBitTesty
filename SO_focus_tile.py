#!/usr/bin/env python3
"""
SO_focus_tile.py

SmartObject to navigate the grid until the red-highlighted tile
matching the given name (by template or OCR) is in focus.

Usage:
  SO_focus_tile.py <PortMask> <TileName> [<JobRunHistoryID> <UIJobHistoryID>
                                           <UITestCaseHistoryID> <UIStepHistoryID>]

Example:
  SO_focus_tile.py 2 "TV Activity" 1234 abcd efgh ijkl
"""
from __future__ import annotations
import sys, time, subprocess, pathlib
import json, uuid
import cv2, numpy as np, pytesseract, requests
from typing import Any

# ─── CONFIG & DEBUG ─────────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
DOTFINDER = SCRIPT_DIR / "dotfinder"
DOTFINDER.mkdir(exist_ok=True)
# Template file must be named "<TileName>.jpg" alongside this script:
# e.g. "TV Activity.jpg"
# If missing, template-matching will be skipped.
# ─────────────────────────────────────────────────────────────────────────────

def execute_classifyfailure(value: str, step_id: str) -> None:
    print("execute_classifyfailure")
    folder = "C:/DPUnified/Tools/ClassifyFailureAgent/"
    try:
        proc = subprocess.Popen(
            [folder + "ClassifyFailure.exe", value, step_id],
            cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = proc.communicate()
        print("ClassifyFailure Output:", out.decode())
        print("ClassifyFailure Error: ", err.decode())
    except Exception as e:
        print("ClassifyFailure exception:", e)

# ─── DP STUDIO REST ADAPTER ──────────────────────────────────────────────────
STUDIO_URL = "http://127.0.0.1:5000"
_session: requests.Session | None = None

def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type":"application/json"})
    rsp = _session.post(STUDIO_URL + endpoint, data=json.dumps(payload), timeout=30)
    rsp.raise_for_status()
    return rsp.json()

def grab_current_frame(mask: int) -> np.ndarray:
    tmp = fr"\\INDEVW-DEVPART03\DataStorage\Snapshots\{uuid.uuid4()}.png"
    payload = {
        "PortView": mask, "Operation":3,
        "OutputFilePath": tmp,
        "xLeftTopRectangle":0, "yLeftTopRectangle":0,
        "RectangleWidth":960, "RectangleHeight":540,
        "RecordDuration":0, "Threshold":0,
        "Duration":0, "MonitorDuration":0,
        "AudioCutOffThreshold":0
    }
    r = _post("/api/v4/video", payload)
    if r.get("ResultCode")!=0 or not r.get("ResultFilePath"):
        raise RuntimeError("Snapshot failed → "+str(r))
    img = cv2.imread(r["ResultFilePath"])
    if img is None:
        raise IOError("Cannot read PNG at "+r["ResultFilePath"])
    return img

def convertPortViewMask2String(mask: int) -> str:
    b = format(mask, '08b')[::-1]
    return ''.join(str(i+1) for i,bit in enumerate(b) if bit=='1')

def send_command_ex(mask: int, key: str) -> None:
    payload = {
        "portviews": convertPortViewMask2String(mask),
        "keyname": key, "username":"", "keypressduration":0, "delayduration":0
    }
    requests.post("http://127.0.0.1:5005/api/execute-irsenderagent2", json=payload)

# ─── VISION HELPERS ─────────────────────────────────────────────────────────
def _red_mask_strict(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    l1,u1 = np.array((0,150,150)), np.array((10,255,255))
    l2,u2 = np.array((160,150,150)), np.array((180,255,255))
    return cv2.bitwise_or(
        cv2.inRange(hsv, l1,u1), cv2.inRange(hsv, l2,u2)
    )

def _find_highlight_rect(frame: np.ndarray) -> tuple[int,int,int,int] | None:
    m = _red_mask_strict(frame)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best,area = None,0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        A = cv2.contourArea(c)
        aspect = w/h if h else 0
        if A>800 and 0.4<aspect<2.5 and A>area:
            best,area = (x,y,w,h),A
    return best

def _save_dbg(name: str, img: np.ndarray):
    cv2.imwrite(str(DOTFINDER/f"{name}.png"), img)

# Template matcher
def _load_template(name: str) -> np.ndarray | None:
    path = SCRIPT_DIR / f"{name}.jpg"
    if not path.exists(): return None
    tpl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if tpl is None: return None
    if tpl.shape[2]==4:
        tpl = cv2.cvtColor(tpl, cv2.COLOR_BGRA2BGR)
    return tpl

def _match_template(tile: np.ndarray, tpl: np.ndarray) -> float:
    def prep(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g,(5,5),0)
    b1 = cv2.threshold(prep(tile),180,255,cv2.THRESH_BINARY_INV)[1]
    b2 = cv2.threshold(prep(tpl),180,255,cv2.THRESH_BINARY_INV)[1]
    res = cv2.matchTemplate(b1,b2,cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(res)[1])

# OCR label under tile
def _ocr_label(frame: np.ndarray, rect: tuple[int,int,int,int]) -> str:
    x,y,w,h = rect
    y1,y2 = y+h+int(0.05*h), y+h+int(0.60*h)
    roi = frame[y1:y2, x:x+w]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,bw = cv2.threshold(g,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(bw,
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ ").strip().upper()
    return txt

# ─── MAIN NAVIGATION ────────────────────────────────────────────────────────
def navigate_to_tile(mask: int, name: str, max_passes: int=3) -> None:
    tpl = _load_template(name)
    # detect initial highlight and grid dims
    f0 = grab_current_frame(mask)
    hi = _find_highlight_rect(f0)
    if hi is None:
        raise AssertionError("Cannot locate initial highlight")
    COLS,ROWS = 3, 3

    for p in range(max_passes):
        time.sleep(0.2)
        for r in range(ROWS):
            for c in range(COLS):
                tag = f"nav_{p}_{r}_{c}"
                f = grab_current_frame(mask)
                hi = _find_highlight_rect(f)
                if hi is None:
                    raise AssertionError("Highlight lost")
                x,y,w,h = hi
                tile = f[y+4:y+h-4, x+4:x+w-4]
                # 1) template?
                tmpl_ok = False
                score = 0.0
                if tpl is not None:
                    score = _match_template(tile, tpl)
                    print(f"[TEMPLATE {tag}] score={score:.2f}")
                    tmpl_ok = score>=0.40
                # 2) label?
                lbl = _ocr_label(f, hi)
                print(f"[OCR {tag}] label='{lbl}'")
                txt_ok = name.upper() in lbl
                if tmpl_ok or txt_ok:
                    print(f"[OK] Found '{name}' at pass={p}, row={r}, col={c}")
                    return
                # move right or wrap
                if c<COLS-1:
                    send_command_ex(mask, "RIGHT"); time.sleep(0.2)
                else:
                    # back to col=0
                    for _ in range(COLS-1):
                        send_command_ex(mask,"LEFT"); time.sleep(0.08)
                    if r<ROWS-1:
                        send_command_ex(mask,"DOWN"); time.sleep(0.25)
    raise AssertionError(f"Tile '{name}' not found")

def main():
    args = sys.argv[1:]
    if len(args)<2:
        print("Usage: SO_focus_tile.py <PortMask> <TileName> [<4 history IDs>]")
        sys.exit(1)
    mask = int(args[0])
    name = args[1]
    # grab optional UIStepHistoryID (4th extra)
    step_id = args[5] if len(args)>=6 else None

    try:
        navigate_to_tile(mask, name)
    except AssertionError as e:
        print("[FAIL]", e)
        if step_id:
            execute_classifyfailure("FAIL", step_id)
        sys.exit(4)
    print("[OK] Tile focused")
    sys.exit(0)

if __name__=="__main__":
    main()
