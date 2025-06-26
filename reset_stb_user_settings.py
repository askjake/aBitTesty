#!/usr/bin/env python3
"""
reset_stb_user_settings.py  (re-imagined)

 --dump       →  query every “data-group” via get_stb_settings and save
                 <stb>_current.json

 --compare    →  list groups/fields that differ between
                 <stb>_current.json  and  <stb>_normal.json

 --apply      →  send set_stb_settings for every differing group to make the
                 STB match <stb>_normal.json

Example:
   python reset_stb_user_settings.py -n hopper --dump --compare --apply
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from typing import Dict, Any

# allow simple portview invocation from t_dvr_001
import dp_lib as gng
if len(sys.argv) == 2 and sys.argv[1].isdigit():
    portview = sys.argv[1]
    gng.send_command(portview, "RESET_USER_SETTINGS")
    print("✓ reset_stb_user_settings command sent to port", portview)
    sys.exit(0)

from sgs_lib import STB, sgs_arg_parse   # same folder
################################################################################
#  data-group catalogue  (id  → name) – extend if you need more
################################################################################
# in reset_stb_user_settings.py

GROUPS: dict[int, str] = {
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
    # 12 omitted per your request
    13:  "audio_language",
    14:  "audio",
    15:  "timer_defaults",
    16:  "ptat_enable",
    17:  "video_format",
    #18:  "remote_codes",
    #19:  "remote",
    #20:  "remote_mode_enable",
    21:  "system_name",
    22:  "tv",
    23:  "tv_format",
    24:  "tv_enhancements_enable",
    25:  "hdmi_cec_enable",
    26:  "network_bridging_enable",
    #27:  "phone",
    #28:  "caller_id_enable",
    29:  "whole_home",
    30:  "wjap_name",
    #31:  "check_switch_alternate_enable",
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
    #43:  "sling_popup",
    #44:  "hdmi_hdcp_enable",
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
    61:  "touchpad_sensitivity",
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

# ───────────────────────── helpers ──────────────────────────────────────────
def _json_path(stb: STB, suffix: str) -> Path:
    """ <script-dir>/<receiver>_<suffix>.json  """
    fname = f"{stb.stb.replace('-', '')}.{suffix}"
    return HERE / fname


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))
    print("✓ wrote", path)


# ───────────────────────── main logic ───────────────────────────────────────
def main() -> None:
    p = sgs_arg_parse("SGS settings helper")
    p.add_argument("-d", "--dump",    action="store_true", help="save <stb>_current.json")
    p.add_argument("-c", "--compare", action="store_true", help="diff  current vs normal")
    p.add_argument("-a", "--apply",   action="store_true", help="push normal settings")
    args = p.parse_args()

    if not (args.dump or args.compare or args.apply):
        p.error("nothing to do – use --dump / --compare / --apply")

    stb = STB(args)
    cur_file = _json_path(stb, "current.json")
    norm_file = _json_path(stb, "normal.json")

    # ---------------------------------------------------------------- dump --
    if args.dump:
        current = {}
        for gid, gname in GROUPS.items():
            payload = {"command": "get_stb_settings",
                       "id": gid, "name": gname}
            reply, _ = stb.sgs_command(payload)
            if reply and reply.get("result") == 1:
                current[gname] = reply.get("data", {})
            else:
                print(f"[WARN] get {gname} (id {gid}) → result={reply}")
        _dump_json(cur_file, current)

    # ---------------------------------------------------------------- load files
    current = _load_json(cur_file)
    normal  = _load_json(norm_file)

    if args.compare:
        if not normal:
            print(f"[NOTE] normal baseline not found – writing template {norm_file}")
            _dump_json(norm_file, current)
            normal = current

        print("\n-------------------------------------------------------------")
        diff_cnt = 0
        for grp in sorted(normal.keys()):
            if grp not in current:
                print(f"{grp}: not in current")
                diff_cnt += 1
                continue
            for k, v in normal[grp].items():
                if current[grp].get(k) != v:
                    print(f"{grp}.{k}:   current={current[grp].get(k)}   normal={v}")
                    diff_cnt += 1
        if diff_cnt == 0:
            print("This settop is NORMAL")
        print("-----------------------------------------------------------\n")

    # ---------------------------------------------------------------- apply --
    if args.apply:
        if not normal:
            print("normal baseline missing – abort apply")
            sys.exit(1)

        for gid, gname in GROUPS.items():
            desired = normal.get(gname)
            if not desired:
                continue
            # only push when different (or --dump not done)
            if desired != current.get(gname):
                print(f"[SET] {gname}")
                payload = {"command": "set_stb_settings",
                           "id": gid, "name": gname,
                           "data": desired}
                reply, _ = stb.sgs_command(payload)
                if not reply or reply.get("result") not in (1, 56):
                    print("   → ERROR", reply)
                else:
                    if reply.get("result") == 56:
                        print("   ✓ saved (async event will follow)")
                    else:
                        print("   ✓ ok")
                time.sleep(0.2)  # avoid flooding

if __name__ == "__main__":
    main()
