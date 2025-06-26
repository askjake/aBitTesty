#!/usr/bin/env python3
"""
SO_Validate_MiniGuide.py  –  Smart-Object

1) Send one RIGHT-arrow to the device (via UISet “CmdRight”)
2) Poll the DP-Studio “/api/v3/text” endpoint until the phrase
      ‘Mini Guide’
   is found anywhere on screen (or within an optional crop).
3) Success  -> exit 0
   Failure  -> execute_classifyfailure("FAIL", UIStepHistoryID) and exit-code 4
"""

from __future__ import annotations
import sys, time, uuid, json, subprocess
from pathlib import Path
from typing import Any, Optional

import requests   # only std-lib + requests needed

# ──────────────────────────── configuration ──────────────────────────────
STUDIO = "127.0.0.1"                 # change if Studio not local
BASE   = f"http://{STUDIO}:5000"
USER   = ""                          # tester-role user if you wish

SNAP_DIR = r"\\INDEVW-DEVPART03\DataStorage\Snapshots"   # <— keep consistent

SCRIPT_DIR   = Path(__file__).resolve().parent
DOTFINDER_DIR = SCRIPT_DIR / "dotfinder"
DOTFINDER_DIR.mkdir(exist_ok=True)

def _save_dbg(name: str, content: bytes) -> None:
    (DOTFINDER_DIR / name).write_bytes(content)

# ─────────────────────────── helper: REST POST ───────────────────────────
_session: Optional[requests.Session] = None
def _sess() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"Content-Type": "application/json"})
    return _session

def _post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{BASE}{endpoint}"
    rsp = _sess().post(url, data=json.dumps(payload), timeout=30)
    rsp.raise_for_status()
    return rsp.json()

# ─────────────────────────── low-level commands ──────────────────────────


def convertPortViewMask2String(mask: str):
    binary_str = format(mask, '08b')
    reversed_binary_str = binary_str[::-1]

    portViewString = ''.join(
        str(idx + 1) for idx, bit in enumerate(reversed_binary_str) if bit == '1')

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



def text_search(mask: int, phrase: str,
                thresh: int = 80) -> tuple[bool, float]:
    """Return (found?, match-score%)."""
    body = {
        "PortView":           mask,
        "Operation":          1,            # 1 = SearchText
        "SearchText":         phrase,
        "xLeftTopRectangle":  0,
        "yLeftTopRectangle":  0,
        "RectangleWidth":     960,
        "RectangleHeight":    540,
        "Threshold":          thresh
    }
    res = _post("/api/v3/text", body)
    ok   = bool(res.get("ResultFlag"))
    conf = float(res.get("ResultScore", 0))
    return ok, conf

def snapshot(mask: int) -> bytes:
    """Grab a PNG snapshot and return its raw bytes for triage."""
    fname = f"{uuid.uuid4()}.png"
    path  = rf"{SNAP_DIR}\{fname}"
    body  = {
        "PortView":           mask,
        "Operation":          3,     # snapshot
        "OutputFilePath":     path,
        "xLeftTopRectangle":  0,
        "yLeftTopRectangle":  0,
        "RectangleWidth":     960,
        "RectangleHeight":    540,
        "RecordDuration":     0,
        "Threshold":          0,
        "Duration":           0,
        "MonitorDuration":    0,
        "AudioCutOffThreshold": 0
    }
    res = _post("/api/v4/video", body)
    if res.get("ResultCode") != 0:
        raise RuntimeError(f"snapshot failed → {res}")
    return Path(res["ResultFilePath"]).read_bytes()

# ─────────────────────────── classify failure hook ───────────────────────
def execute_classifyfailure(_value: str, uistephistoryid: str) -> None:
    print("execute_classifyfailure")
    exe = Path(r"C:/DPUnified/Tools/ClassifyFailureAgent/ClassifyFailure.exe")
    try:
        subprocess.run([str(exe), _value, uistephistoryid],
                       cwd=exe.parent, check=False, capture_output=True)
    except Exception as e:
        print("ClassifyFailure exception:", e)

# ─────────────────────────── main logic ──────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: SO_Focus_MiniGuide.py <PortMask> [JobRunID UIJobID UITestID UIStepID]")
        sys.exit(1)

    port_mask = int(sys.argv[1])
    UIStepHistoryID: Optional[str] = sys.argv[5] if len(sys.argv) >= 6 else None

    try:
        # 1) Send RIGHT once
        command = "Right"
        send_command_ex(port_mask, command)
        time.sleep(0.5)                       # small settle

        # 2) poll up to ~5 s (10 * 0.5 s)
        FOUND = False
        for _ in range(10):
            ok, score = text_search(port_mask, "Mini Guide", thresh=80)
            print(f"[poll] found={ok} score={score:.0f}")
            if ok:
                FOUND = True
                break
            time.sleep(0.5)

        if not FOUND:
            snap = snapshot(port_mask)
            _save_dbg(f"miniguide_fail_{int(time.time())}.png", snap)
            raise AssertionError("Mini Guide text not detected in time-out")

    except AssertionError as e:
        print("[FAIL]", e)
        if UIStepHistoryID:
            execute_classifyfailure("FAIL", UIStepHistoryID)
        sys.exit(4)

    print("[OK] Mini-Guide detected.")
    sys.exit(0)

# ───────────────────────── entrypoint ────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback, sys
        traceback.print_exc()
        sys.exit(1)
