#!/usr/bin/env python3
"""
normalizer.py  (re‐imagined)

Usage:
   # Shortcut: send a single RESET_USER_SETTINGS to port <PortView> and exit
   python normalizer.py <PortView>

   # SGS mode (any of these flags may be combined; defaults to --reset if none are given):
   python normalizer.py -n <receiver_name> [--dump] [--compare] [--apply] [--reset]

Flags:
  -n, --name     : STB receiver name (must match GROUPS id in sgs_lib.STB)
  -d, --dump     : Query every “data-group” and save <receiver>.current.json under normal/
  -c, --compare  : Diff <receiver>.current.json vs normal/normal.json  (create template if needed)
  -a, --apply    : Push differences so that STB matches normal/normal.json
  -r, --reset    : Immediately push every group in normal/normal.json to the STB

If you specify `-n` but none of `-d,-c,-a,-r`, it will act as if `--reset` was passed.
All JSON files now live under a “normal/” subfolder.
"""

from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
import subprocess
import traceback
# Allow single‐digit invocation: send a RESET_USER_SETTINGS to that port and exit
import dp_lib as gng
if len(sys.argv) == 2 and sys.argv[1].isdigit():
    portview = sys.argv[1]
    gng.send_command(portview, "RESET_USER_SETTINGS")
    print("✓ RESET_USER_SETTINGS command sent to port", portview)
    sys.exit(0)

from sgs_lib import STB, sgs_arg_parse  # must live alongside this script

################################################################################
#  data‐group catalogue  (id → name).  Modify as needed:
################################################################################
GROUPS: Dict[int, str] = {
     1:  "closed_caption_enable",
     2:  "closed_caption",
     3:  "parental_control_enable",
     4:  "parental_controls",
     5:  "parental_control_password",
     6:  "guide",
     7:  "cursor_enable",
     8:  "cursor",
     9:  "channel_preference",
    10:  "multi_channel_swap",
    11:  "multi_channel_recall",
    # (12 omitted)
    13:  "audio_language",
    14:  "audio",
    15:  "timer_defaults",
    16:  "ptat_enable",
    17:  "video_format",
    # (18,19,20 omitted)
    21:  "system_name",
    22:  "tv",
    23:  "tv_format",
    24:  "tv_enhancements_enable",
    25:  "hdmi_cec_enable",
    26:  "network_bridging_enable",
    # (27,28 omitted)
    29:  "whole_home",
    30:  "wjap_name",
    # (31 omitted)
    32:  "inactivity_standby_enable",
    33:  "inactivity_standby",
    34:  "nightly_update_enable",
    35:  "nightly_update",
    36:  "control_4_enable",
    37:  "bluetooth_enable",
    38:  "media_device_pairing_enable",
    39:  "auto_transcode_enable",
    40:  "dvr_sort",
    41:  "media_group_by",
    42:  "wifi",
    # (43,44 omitted)
    45:  "od_popups_on_off",
    46:  "help_overlay_info_popup_on_off",
    47:  "home_media_settings",
    48:  "dvr_schedule",
    49:  "mobile_antenna",
    50:  "guide_appearance",
    51:  "search_netflix",
    52:  "screen_language",
    53:  "ui_theme",
    54:  "cvaa",
    55:  "cvaa_speech_enable",
    56:  "cvaa_magnification_enable",
    57:  "home_screen",
    58:  "search_filter",
    59:  "cvaa_audio_description_enable",
    60:  "whole_home_music_enable",
    #61:  "touchpad_sensitivity",
    62:  "remote_cust_buttons",
    63:  "dish_ip_mode",
    64:  "large_dvr_images_enable",
    65:  "voice_control_mode",
    66:  "coproc_popup",
    67:  "dvr_filter",
    68:  "wifi_wizard",
    69:  "home_native_screen",
}

HERE = Path(__file__).resolve().parent
NORMAL_DIR = HERE / "normal"
NORMAL_DIR.mkdir(exist_ok=True)



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

# ─────────────────────────────── Helpers ────────────────────────────────────
def _json_path(stb: STB, suffix: str) -> Path:
    """
    Returns “normal/<receiver>.<suffix>”, e.g.
      normal/R1234567890.current.json
      normal/R1234567890.normal.json
    (But NOT used by the -r/--reset flag, which will read plain normal/normal.json.)
    """
    base_name = stb.stb.replace("-", "")
    fname = f"{base_name}.{suffix}"
    return NORMAL_DIR / fname


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(f"✓ wrote {path}")


# ─────────────────────────────── Main Logic ─────────────────────────────────
def main() -> None:
    # ─── 1) Set up argparse ───────────────────────────────────────────────
    parser = sgs_arg_parse("SGS settings helper")
    parser.add_argument("-d", "--dump",    action="store_true",
                        help="save <receiver>.current.json under normal/")
    parser.add_argument("-c", "--compare", action="store_true",
                        help="diff <receiver>.current.json vs normal/normal.json")
    parser.add_argument("-a", "--apply",   action="store_true",
                        help="push normal/normal.json differences to STB")
    parser.add_argument("-r", "--reset",   action="store_true",
                        help="immediately apply normal/normal.json to STB and exit")

    # ─── 2) Parse known args + collect any extras ────────────────────────
    args, extra_args = parser.parse_known_args()

    # extra_args now holds *all* trailing positional args
    # e.g. python normalizer.py -n Box123 --dump ID1 ID2 ID3 ...
    # -> extra_args = ['ID1','ID2','ID3', ...]

    # ─── 3) If you expect exactly 4 history IDs at the end, you can do:
    UIStepHistoryID = None
    if len(extra_args) >= 4:
        # grab the last four entries
        JobRunHistoryID, UIJobHistoryID, UITestCaseHistoryID, UIStepHistoryID = extra_args[-4:]
    # (or just keep all of extra_args in a list to handle "multiple" IDs)

    # ─── 4) Default “--reset” behavior ────────────────────────────────────
    if args.name and not (args.dump or args.compare or args.apply or args.reset):
        args.reset = True
    if not (args.dump or args.compare or args.apply or args.reset):
        parser.error("nothing to do – use --dump / --compare / --apply / --reset")

    stb = STB(args)

    current_file        = _json_path(stb, "current.json")
    per_receiver_normal = _json_path(stb, "normal.json")
    BASELINE_FILE       = NORMAL_DIR / "normal.json"

    # ─── 5) Handle --reset immediately ────────────────────────────────────
    if args.reset:
        try:
            baseline = _load_json(BASELINE_FILE)
            if not baseline:
                print("[NOTE] baseline missing, creating from current... ")
                baseline = {}
                for gid, gname in GROUPS.items():
                    reply, _ = stb.sgs_command({
                        "command": "get_stb_settings",
                        "id": gid, "name": gname
                    })
                    if reply and reply.get("result") == 1:
                        baseline[gname] = reply["data"]
                    else:
                        print(f"[WARN] get {gname} → {reply}")
                _dump_json(BASELINE_FILE, baseline)

            before = {}
            for gid, gname in GROUPS.items():
                reply, _ = stb.sgs_command({
                    "command": "get_stb_settings",
                    "id": gid, "name": gname
                })
                before[gname] = reply.get("data", {}) if reply and reply.get("result") == 1 else {}

            print("\n[INFO] Applying baseline settings...\n")
            for gid, gname in GROUPS.items():
                data = baseline.get(gname)
                if data is None:
                    continue
                stb.sgs_command({
                    "command": "set_stb_settings",
                    "id": gid, "name": gname, "data": data
                })
                time.sleep(0.2)

            after = {}
            for gid, gname in GROUPS.items():
                reply, _ = stb.sgs_command({
                    "command": "get_stb_settings",
                    "id": gid, "name": gname
                })
                after[gname] = reply.get("data", {}) if reply and reply.get("result") == 1 else {}

            print("\n[RESULT] Differences (before -> after):\n")
            diffs = [g for g in GROUPS.values() if before.get(g) != after.get(g)]
            if diffs:
                for g in diffs:
                    print(f" - {g}: changed")
            else:
                print("No changes detected.")
            sys.exit(0)

        except Exception:
            traceback.print_exc()
            if UIStepHistoryID:
                execute_classifyfailure("FAIL", UIStepHistoryID)
            sys.exit(4)

    # ─── 6) dump / compare / apply modes ──────────────────────────────────
    if args.dump:
        current = {}
        for gid, gname in GROUPS.items():
            reply, _ = stb.sgs_command({
                "command": "get_stb_settings",
                "id": gid, "name": gname
            })
            if reply and reply.get("result") == 1:
                current[gname] = reply["data"]
            else:
                print(f"[WARN] get {gname} → {reply}")
        _dump_json(current_file, current)

    current       = _load_json(current_file)
    common_normal = _load_json(BASELINE_FILE)

    if args.compare:
        if not common_normal:
            print("[NOTE] normal/normal.json not found; creating template…")
            _dump_json(BASELINE_FILE, current)
            common_normal = current

        print("\n" + "─" * 60)
        diff_cnt = 0
        for grp in sorted(common_normal.keys()):
            if grp not in current:
                print(f"{grp}: missing in current")
                diff_cnt += 1
                continue
            for key, val in common_normal[grp].items():
                cur_val = current[grp].get(key)
                if cur_val != val:
                    print(f"{grp}.{key}: current={cur_val} normal={val}")
                    diff_cnt += 1
        if diff_cnt == 0:
            print("✔ Already normal.")
        print("─" * 60 + "\n")

    if args.apply:
        try:
            if not common_normal:
                print("[ERROR] normal/normal.json missing – cannot apply.")
                sys.exit(1)
            for gid, gname in GROUPS.items():
                desired = common_normal.get(gname)
                if desired is None or desired == current.get(gname):
                    continue
                print(f"[SET ] {gname} … ", end="", flush=True)
                reply, _ = stb.sgs_command({
                    "command": "set_stb_settings",
                    "id": gid, "name": gname, "data": desired
                })
                if not reply or reply.get("result") not in (1, 56):
                    print(f"ERROR → {reply}")
                else:
                    print("async ok" if reply.get("result") == 56 else "ok")
                time.sleep(0.2)
        except Exception:
            if UIStepHistoryID:
                execute_classifyfailure("FAIL", UIStepHistoryID)
            sys.exit(4)

    print("Done.")


if __name__ == "__main__":
    main()
