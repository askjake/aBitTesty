#!/usr/bin/env python3
"""
youtube_splash_screen.py

Wait up to 180 s, grab a frame, confirm YouTube splash / profile screen.
"""

from __future__ import annotations
import cv2, numpy as np, pytesseract, sys, time, uuid
from pathlib import Path
from typing import Any
import json

STUDIO_URL = "http://127.0.0.1:5000"  # os.getenv("DP_STUDIO")  # e.g. "http://<studio-ip>:5000"
USER = ""  # os.getenv("DP_USER", "")


def _studio_on() -> bool:
    return bool(STUDIO_URL)


_session: requests.Session | None = None

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
        raise RuntimeError(f"Snapshot failed -> {result}")
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
# -------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
DOT = SCRIPT_DIR / "dotfinder"; DOT.mkdir(exist_ok=True)

_LOGO = cv2.imread(str(SCRIPT_DIR / "youtube_wordmark_200x40.png"), cv2.IMREAD_UNCHANGED)
if _LOGO is None:
    raise FileNotFoundError("youtube_wordmark_200x40.png missing")

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

def _save(img, tag): cv2.imwrite(str(DOT/f"ytfail_{tag}.png"), img)

def has_logo(frame)->bool:
    h,w = frame.shape[:2]
    roi = frame[int(0.02*h):int(0.12*h), int(0.40*w):int(0.60*w)]
    g1  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    g2  = cv2.cvtColor(_LOGO[:,:,:3], cv2.COLOR_BGR2GRAY)
    _,b1= cv2.threshold(g1,180,255,cv2.THRESH_BINARY_INV)
    _,b2= cv2.threshold(g2,180,255,cv2.THRESH_BINARY_INV)
    res = cv2.matchTemplate(b1,b2,cv2.TM_CCOEFF_NORMED)
    return cv2.minMaxLoc(res)[1] >= 0.25

def has_keywords(frame)->bool:
    h,w=frame.shape[:2]
    roi=frame[int(0.20*h):int(0.60*h), int(0.10*w):int(0.90*w)]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    _,bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    txt=pytesseract.image_to_string(bw,config="--oem 3 --psm 6").upper()
    return ("ADD ACCOUNT" in txt) or ("YOUTUBE KIDS" in txt) or ("SWITCH" in txt)

def main():
    if len(sys.argv)<2: print("usage: youtube_splash_screen.py <PortMask>"); sys.exit(1)
    mask=int(sys.argv[1])
    time.sleep(180)          # wait for app launch
    f = grab_current_frame(mask)
    ok = has_logo(f) and has_keywords(f)
    if ok:
        print("[OK] YouTube splash detected"); sys.exit(0)
    _save(f,int(time.time()))
    print("[FAIL] YouTube splash NOT detected"); sys.exit(4)

if __name__=="__main__": main()
