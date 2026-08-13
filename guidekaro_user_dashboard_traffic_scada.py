from __future__ import annotations

import html
import io
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError
from streamlit_autorefresh import st_autorefresh


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "uc1_events.db"
FRAME_PATH = BASE_DIR / "assets" / "latest_frame.jpg"

st.set_page_config(
    page_title="GuideKaro User Safety Dashboard",
    page_icon="🚸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_events() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(DB_PATH) as connection:
            return pd.read_sql_query(
                "SELECT * FROM events ORDER BY id DESC LIMIT 500",
                connection,
            )
    except Exception as exc:
        st.error(f"Unable to read GuideKaro event data: {exc}")
        return pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    defaults = {
        "timestamp": "",
        "intersection": "Current Crosswalk",
        "weather": "Unknown",
        "road_condition": "Unknown",
        "visibility": "Unknown",
        "vehicle_count": 0,
        "pedestrian_count": 0,
        "vehicle_speed_kmh": 0.0,
        "distance_to_crosswalk_m": 0.0,
        "pedestrian_vehicle_distance_m": 0.0,
        "confidence": 0.0,
        "risk_score": 0.0,
        "status": "SAFE",
        "response_time_ms": 0.0,
        "fps": 0.0,
        "track_id": None,
        "movement_direction": "UNKNOWN",
        "trajectory_conflict": 0,
        "ttc_seconds": None,
        "notes": "",
    }

    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_columns = [
        "vehicle_count",
        "pedestrian_count",
        "vehicle_speed_kmh",
        "distance_to_crosswalk_m",
        "pedestrian_vehicle_distance_m",
        "confidence",
        "risk_score",
        "response_time_ms",
        "fps",
        "trajectory_conflict",
        "ttc_seconds",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["status"] = df["status"].fillna("SAFE").astype(str).str.upper()
    return df


def safe_image(path: Path):
    if not path.exists():
        return None

    for _ in range(3):
        try:
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                return image.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, PermissionError):
            continue

    return None


def clean_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def alert_profile(status: str, risk: float, ttc) -> dict[str, str]:
    status = str(status).upper()
    ttc_value = None if pd.isna(ttc) else clean_number(ttc)

    if status == "VIOLATION" or risk >= 75 or (
        ttc_value is not None and 0 < ttc_value <= 3
    ):
        return {
            "level": "DANGER",
            "icon": "🚨",
            "headline": "STOP / BRAKE NOW",
            "message": "A pedestrian or vehicle conflict is detected in the crosswalk area.",
            "class_name": "danger-card",
        }

    if status == "WARNING" or risk >= 45 or (
        ttc_value is not None and 0 < ttc_value <= 6
    ):
        return {
            "level": "WARNING",
            "icon": "⚠️",
            "headline": "SLOW DOWN",
            "message": "A pedestrian or potential conflict is approaching the crosswalk.",
            "class_name": "warning-card",
        }

    return {
        "level": "SAFE",
        "icon": "✅",
        "headline": "PROCEED WITH CAUTION",
        "message": "No immediate pedestrian–vehicle conflict is currently detected.",
        "class_name": "safe-card",
    }


def format_ttc(value) -> str:
    if pd.isna(value):
        return "N/A"

    value = clean_number(value)
    if value <= 0:
        return "N/A"

    return f"{value:.1f} s"


def build_reason(latest: pd.Series) -> str:
    notes = str(latest.get("notes", "") or "").strip()
    if notes and notes.lower() not in {"nan", "none"}:
        return notes

    reasons: list[str] = []

    risk = clean_number(latest.get("risk_score", 0))
    speed = clean_number(latest.get("vehicle_speed_kmh", 0))
    distance = clean_number(latest.get("distance_to_crosswalk_m", 0))
    separation = clean_number(latest.get("pedestrian_vehicle_distance_m", 0))
    pedestrians = int(clean_number(latest.get("pedestrian_count", 0)))
    conflict = bool(clean_number(latest.get("trajectory_conflict", 0)))

    if pedestrians > 0:
        reasons.append(f"{pedestrians} pedestrian(s) detected")
    if conflict:
        reasons.append("predicted trajectory conflict")
    if speed >= 35:
        reasons.append("vehicle speed is elevated")
    if 0 < distance <= 20:
        reasons.append("vehicle is close to the crosswalk")
    if 0 < separation <= 15:
        reasons.append("pedestrian–vehicle separation is low")
    if risk >= 45:
        reasons.append("combined risk score is elevated")

    if not reasons:
        return "The system is continuously monitoring the crosswalk for changes."

    return ", ".join(reasons).capitalize() + "."


def render_scada_hmi(latest: pd.Series) -> None:
    """Render a four-way SCADA-style traffic/crosswalk HMI from live GuideKaro values."""
    risk = max(0.0, min(100.0, clean_number(latest.get("risk_score", 0))))
    speed = max(0.0, clean_number(latest.get("vehicle_speed_kmh", 0)))
    distance = max(0.0, clean_number(latest.get("distance_to_crosswalk_m", 0)))
    separation = max(0.0, clean_number(latest.get("pedestrian_vehicle_distance_m", 0)))
    pedestrians = max(0, int(clean_number(latest.get("pedestrian_count", 0))))
    vehicles = max(0, int(clean_number(latest.get("vehicle_count", 0))))
    ttc_raw = latest.get("ttc_seconds")
    ttc = None if pd.isna(ttc_raw) or clean_number(ttc_raw) <= 0 else clean_number(ttc_raw)
    conflict = bool(clean_number(latest.get("trajectory_conflict", 0)))
    direction = str(latest.get("movement_direction", "UNKNOWN") or "UNKNOWN").upper()
    status = str(latest.get("status", "SAFE") or "SAFE").upper()
    track_id = latest.get("track_id")
    track_text = "N/A" if pd.isna(track_id) else str(int(clean_number(track_id)))

    profile = alert_profile(status, risk, ttc_raw)
    level = profile["level"]

    # This is an HMI/advisory representation, not a municipal signal controller.
    # It deliberately ties the displayed signal state to GuideKaro risk:
    # SAFE -> monitored approach green; WARNING -> yellow; DANGER -> all red.
    if level == "DANGER":
        ns_state = "RED"
        ew_state = "RED"
        phase_name = "ALL STOP — RISK INTERLOCK"
        signal_command = "STOP"
    elif level == "WARNING":
        ns_state = "YELLOW"
        ew_state = "RED"
        phase_name = "CAUTION — PREPARE TO STOP"
        signal_command = "SLOW / PREPARE TO STOP"
    else:
        ns_state = "GREEN"
        ew_state = "RED"
        phase_name = "MONITORED APPROACH CLEAR"
        signal_command = "PROCEED WITH CAUTION"

    lamp = {
        "red_on": "#ff453a",
        "red_off": "#4a2020",
        "yellow_on": "#ffd60a",
        "yellow_off": "#4b441d",
        "green_on": "#30d158",
        "green_off": "#173f26",
    }

    ns_red = lamp["red_on"] if ns_state == "RED" else lamp["red_off"]
    ns_yellow = lamp["yellow_on"] if ns_state == "YELLOW" else lamp["yellow_off"]
    ns_green = lamp["green_on"] if ns_state == "GREEN" else lamp["green_off"]
    ew_red = lamp["red_on"] if ew_state == "RED" else lamp["red_off"]
    ew_yellow = lamp["yellow_on"] if ew_state == "YELLOW" else lamp["yellow_off"]
    ew_green = lamp["green_on"] if ew_state == "GREEN" else lamp["green_off"]

    if level == "DANGER":
        state_color = "#ff453a"
        state_soft = "#40191b"
    elif level == "WARNING":
        state_color = "#ffd60a"
        state_soft = "#403917"
    else:
        state_color = "#30d158"
        state_soft = "#153b24"

    # Map 0–20 m to the lower approach lane. 0 m is at the stop line.
    display_distance = min(distance, 20.0)
    vehicle_y = 444 + (display_distance / 20.0) * 112
    vehicle_y = max(444, min(556, vehicle_y))
    toward = direction == "TOWARD_CROSSWALK"
    vehicle_shift = 12 if toward and speed > 1 else 2
    vehicle_period = max(0.75, min(2.8, 2.6 - speed / 40.0))

    pedestrian_visibility = "visible" if pedestrians > 0 else "hidden"
    no_ped_visibility = "hidden" if pedestrians > 0 else "visible"
    conflict_visibility = "visible" if conflict else "hidden"
    normal_path_visibility = "hidden" if conflict else "visible"
    risk_width = max(2.0, risk)
    ttc_text = "N/A" if ttc is None else f"{ttc:.1f} s"
    separation_text = "N/A" if separation >= 90 else f"{separation:.1f} m"
    direction_text = html.escape(direction.replace("_", " ").title())
    reason = html.escape(build_reason(latest))

    pedestrian_signal = "WAIT"
    if pedestrians > 0 and level == "SAFE" and speed < 5:
        pedestrian_signal = "WALK"

    template = r"""
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: #07111f;
            color: #e9f1f8;
            font-family: Inter, Segoe UI, Arial, sans-serif;
        }
        .scada-shell {
            border: 1px solid #34516e;
            border-radius: 16px;
            overflow: hidden;
            background: #081523;
        }
        .scada-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 14px;
            padding: 13px 16px;
            background: #0c2033;
            border-bottom: 1px solid #34516e;
        }
        .scada-title { font-size: 20px; font-weight: 900; letter-spacing: .02em; }
        .scada-subtitle { margin-top: 3px; color: #9ab1c7; font-size: 12px; }
        .state-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            min-width: 190px;
            justify-content: center;
            border-radius: 999px;
            padding: 9px 14px;
            border: 1px solid __STATE_COLOR__;
            background: __STATE_SOFT__;
            color: #fff;
            font-size: 13px;
            font-weight: 900;
        }
        .state-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: __STATE_COLOR__;
            box-shadow: 0 0 14px __STATE_COLOR__;
            animation: signalPulse 1.05s ease-in-out infinite;
        }
        .scada-grid {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(280px, .82fr);
            min-height: 610px;
        }
        .scene { border-right: 1px solid #34516e; background: #0a1724; }
        .operator-panel {
            padding: 13px;
            display: flex;
            flex-direction: column;
            gap: 9px;
            background: #081523;
        }
        .panel-heading {
            color: #9ab1c7;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: .09em;
        }
        .phase-card {
            padding: 11px 12px;
            border: 1px solid __STATE_COLOR__;
            background: __STATE_SOFT__;
            border-radius: 12px;
        }
        .phase-label { color: #d6e3ee; font-size: 10px; font-weight: 900; letter-spacing: .08em; }
        .phase-value { margin-top: 4px; color: #fff; font-size: 16px; font-weight: 900; }
        .risk-card {
            padding: 11px 12px;
            border: 1px solid #34516e;
            background: #0c1c2c;
            border-radius: 12px;
        }
        .risk-row { display:flex; align-items:end; justify-content:space-between; gap:10px; }
        .risk-number { color:#fff; font-size:34px; line-height:1; font-weight:950; }
        .risk-name { color:#9ab1c7; font-size:10px; font-weight:900; letter-spacing:.08em; }
        .risk-track { height:10px; margin-top:10px; border-radius:999px; overflow:hidden; background:#23384d; }
        .risk-fill { width:__RISK_WIDTH__%; height:100%; background:__STATE_COLOR__; box-shadow:0 0 12px __STATE_COLOR__; }
        .kv {
            display:grid;
            grid-template-columns:1fr auto;
            gap:10px;
            align-items:center;
            padding:8px 10px;
            border:1px solid #2b465f;
            border-radius:10px;
            background:#0b1b2a;
        }
        .kv .k { color:#9ab1c7; font-size:11px; }
        .kv .v { color:#fff; font-size:14px; font-weight:900; text-align:right; }
        .reason {
            padding:10px;
            border:1px solid #2b465f;
            border-radius:10px;
            background:#0b1b2a;
            color:#d8e5ef;
            font-size:11px;
            line-height:1.45;
        }
        .disclaimer {
            color:#7f99b2;
            font-size:10px;
            line-height:1.4;
            padding-top:2px;
        }
        svg { display:block; width:100%; height:610px; }
        .tracked-car { animation: carApproach __CAR_PERIOD__s ease-in-out infinite alternate; transform-origin:center; }
        .pedestrian { animation: pedestrianCross 2.7s ease-in-out infinite alternate; }
        .trajectory { stroke-dasharray:10 8; animation: dashFlow 1.0s linear infinite; }
        .signal-active { animation: signalPulse 1.05s ease-in-out infinite; }
        .conflict-ring { animation: conflictPulse 1.0s ease-in-out infinite; transform-origin: 494px 414px; }
        .sensor-wave { animation: sensorWave 1.9s ease-out infinite; transform-origin:430px __VEHICLE_Y__px; }
        @keyframes carApproach { from { transform:translateY(0); } to { transform:translateY(-__CAR_SHIFT__px); } }
        @keyframes pedestrianCross { from { transform:translateX(-38px); } to { transform:translateX(58px); } }
        @keyframes dashFlow { to { stroke-dashoffset:-36; } }
        @keyframes signalPulse { 0%,100% { opacity:.62; } 50% { opacity:1; } }
        @keyframes conflictPulse { 0%,100% { opacity:.55; transform:scale(.78); } 50% { opacity:1; transform:scale(1.22); } }
        @keyframes sensorWave { 0% { opacity:.55; transform:scale(.5); } 100% { opacity:0; transform:scale(1.7); } }
        @media (max-width: 900px) {
            .scada-grid { grid-template-columns:1fr; }
            .scene { border-right:0; border-bottom:1px solid #34516e; }
        }
    </style>

    <div class="scada-shell">
        <div class="scada-topbar">
            <div>
                <div class="scada-title">GuideKaro Four-Way Traffic SCADA HMI</div>
                <div class="scada-subtitle">Risk-driven traffic-light visualization inspired by PLC/SCADA intersection layouts</div>
            </div>
            <div class="state-pill"><span class="state-dot"></span>__LEVEL__ · RISK __RISK__/100</div>
        </div>

        <div class="scada-grid">
            <div class="scene">
                <svg viewBox="0 0 860 610" role="img" aria-label="Four-way intersection SCADA animation with traffic lights, tracked vehicle and pedestrian conflict monitoring">
                    <defs>
                        <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                        <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#ff453a"/></marker>
                        <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#4cc9f0"/></marker>
                    </defs>

                    <!-- Grass / ground -->
                    <rect width="860" height="610" fill="#153422"/>

                    <!-- Four-way roadway -->
                    <rect x="300" y="0" width="260" height="610" fill="#30363c"/>
                    <rect x="0" y="175" width="860" height="260" fill="#30363c"/>
                    <rect x="300" y="175" width="260" height="260" fill="#343b42"/>

                    <!-- Lane markings -->
                    <line x1="430" y1="0" x2="430" y2="155" stroke="#f0c84b" stroke-width="4" stroke-dasharray="24 18" opacity=".9"/>
                    <line x1="430" y1="455" x2="430" y2="610" stroke="#f0c84b" stroke-width="4" stroke-dasharray="24 18" opacity=".9"/>
                    <line x1="0" y1="305" x2="280" y2="305" stroke="#f0c84b" stroke-width="4" stroke-dasharray="24 18" opacity=".9"/>
                    <line x1="580" y1="305" x2="860" y2="305" stroke="#f0c84b" stroke-width="4" stroke-dasharray="24 18" opacity=".9"/>

                    <!-- Direction arrows -->
                    <path d="M382 548 L382 505 M367 520 L382 500 L397 520" fill="none" stroke="#d7dee5" stroke-width="4" opacity=".65"/>
                    <path d="M478 62 L478 105 M463 90 L478 110 L493 90" fill="none" stroke="#d7dee5" stroke-width="4" opacity=".65"/>
                    <path d="M90 260 L135 260 M120 245 L140 260 L120 275" fill="none" stroke="#d7dee5" stroke-width="4" opacity=".65"/>
                    <path d="M770 350 L725 350 M740 335 L720 350 L740 365" fill="none" stroke="#d7dee5" stroke-width="4" opacity=".65"/>

                    <!-- Crosswalks -->
                    <g fill="#eef2f5" opacity=".92">
                        <!-- South -->
                        <rect x="310" y="445" width="34" height="54"/><rect x="353" y="445" width="34" height="54"/><rect x="396" y="445" width="34" height="54"/><rect x="439" y="445" width="34" height="54"/><rect x="482" y="445" width="34" height="54"/><rect x="525" y="445" width="25" height="54"/>
                        <!-- North -->
                        <rect x="310" y="111" width="34" height="54"/><rect x="353" y="111" width="34" height="54"/><rect x="396" y="111" width="34" height="54"/><rect x="439" y="111" width="34" height="54"/><rect x="482" y="111" width="34" height="54"/><rect x="525" y="111" width="25" height="54"/>
                        <!-- West -->
                        <rect x="236" y="185" width="54" height="34"/><rect x="236" y="228" width="54" height="34"/><rect x="236" y="271" width="54" height="34"/><rect x="236" y="314" width="54" height="34"/><rect x="236" y="357" width="54" height="34"/><rect x="236" y="400" width="54" height="25"/>
                        <!-- East -->
                        <rect x="570" y="185" width="54" height="34"/><rect x="570" y="228" width="54" height="34"/><rect x="570" y="271" width="54" height="34"/><rect x="570" y="314" width="54" height="34"/><rect x="570" y="357" width="54" height="34"/><rect x="570" y="400" width="54" height="25"/>
                    </g>

                    <!-- Stop lines -->
                    <line x1="300" y1="435" x2="560" y2="435" stroke="#fff" stroke-width="5"/>
                    <line x1="300" y1="175" x2="560" y2="175" stroke="#fff" stroke-width="5"/>
                    <line x1="290" y1="175" x2="290" y2="435" stroke="#fff" stroke-width="5"/>
                    <line x1="570" y1="175" x2="570" y2="435" stroke="#fff" stroke-width="5"/>

                    <!-- Traffic signal heads: north/south approach follows GuideKaro risk -->
                    <g transform="translate(255 466)">
                        <rect x="0" y="0" width="42" height="108" rx="8" fill="#111820" stroke="#8194a8" stroke-width="2"/>
                        <circle class="__NS_RED_CLASS__" cx="21" cy="22" r="13" fill="__NS_RED__" filter="url(#glow)"/>
                        <circle class="__NS_YELLOW_CLASS__" cx="21" cy="54" r="13" fill="__NS_YELLOW__" filter="url(#glow)"/>
                        <circle class="__NS_GREEN_CLASS__" cx="21" cy="86" r="13" fill="__NS_GREEN__" filter="url(#glow)"/>
                        <text x="21" y="126" text-anchor="middle" fill="#d8e3ed" font-size="11" font-weight="900">SOUTH</text>
                    </g>
                    <g transform="translate(563 36)">
                        <rect x="0" y="0" width="42" height="108" rx="8" fill="#111820" stroke="#8194a8" stroke-width="2"/>
                        <circle class="__NS_RED_CLASS__" cx="21" cy="22" r="13" fill="__NS_RED__" filter="url(#glow)"/>
                        <circle class="__NS_YELLOW_CLASS__" cx="21" cy="54" r="13" fill="__NS_YELLOW__" filter="url(#glow)"/>
                        <circle class="__NS_GREEN_CLASS__" cx="21" cy="86" r="13" fill="__NS_GREEN__" filter="url(#glow)"/>
                        <text x="21" y="126" text-anchor="middle" fill="#d8e3ed" font-size="11" font-weight="900">NORTH</text>
                    </g>

                    <!-- Cross-street signal heads -->
                    <g transform="translate(126 444)">
                        <rect x="0" y="0" width="108" height="42" rx="8" fill="#111820" stroke="#8194a8" stroke-width="2"/>
                        <circle class="__EW_RED_CLASS__" cx="22" cy="21" r="13" fill="__EW_RED__" filter="url(#glow)"/>
                        <circle class="__EW_YELLOW_CLASS__" cx="54" cy="21" r="13" fill="__EW_YELLOW__" filter="url(#glow)"/>
                        <circle class="__EW_GREEN_CLASS__" cx="86" cy="21" r="13" fill="__EW_GREEN__" filter="url(#glow)"/>
                        <text x="54" y="60" text-anchor="middle" fill="#d8e3ed" font-size="11" font-weight="900">WEST</text>
                    </g>
                    <g transform="translate(626 124)">
                        <rect x="0" y="0" width="108" height="42" rx="8" fill="#111820" stroke="#8194a8" stroke-width="2"/>
                        <circle class="__EW_RED_CLASS__" cx="22" cy="21" r="13" fill="__EW_RED__" filter="url(#glow)"/>
                        <circle class="__EW_YELLOW_CLASS__" cx="54" cy="21" r="13" fill="__EW_YELLOW__" filter="url(#glow)"/>
                        <circle class="__EW_GREEN_CLASS__" cx="86" cy="21" r="13" fill="__EW_GREEN__" filter="url(#glow)"/>
                        <text x="54" y="60" text-anchor="middle" fill="#d8e3ed" font-size="11" font-weight="900">EAST</text>
                    </g>

                    <!-- Pedestrian crossing on monitored south crosswalk -->
                    <g class="pedestrian" visibility="__PED_VIS__">
                        <circle cx="385" cy="460" r="9" fill="#fff"/>
                        <line x1="385" y1="470" x2="385" y2="490" stroke="#fff" stroke-width="5" stroke-linecap="round"/>
                        <line x1="385" y1="476" x2="372" y2="485" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
                        <line x1="385" y1="476" x2="398" y2="485" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
                        <line x1="385" y1="490" x2="374" y2="505" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
                        <line x1="385" y1="490" x2="396" y2="505" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
                        <text x="416" y="474" fill="#fff" font-size="12" font-weight="900">PED</text>
                    </g>
                    <text x="322" y="478" fill="#a7b9ca" font-size="12" visibility="__NO_PED_VIS__">No pedestrian detected</text>

                    <!-- Tracked vehicle -->
                    <circle class="sensor-wave" cx="430" cy="__VEHICLE_Y__" r="34" fill="none" stroke="#4cc9f0" stroke-width="3"/>
                    <g class="tracked-car">
                        <rect x="392" y="__CAR_TOP__" width="76" height="54" rx="12" fill="__STATE_COLOR__" stroke="#f8fbff" stroke-width="2" filter="url(#glow)"/>
                        <rect x="404" y="__WINDOW_TOP__" width="52" height="17" rx="6" fill="#0b1520"/>
                        <circle cx="404" cy="__WHEEL_Y__" r="7" fill="#05080c"/><circle cx="456" cy="__WHEEL_Y__" r="7" fill="#05080c"/>
                        <text x="430" y="__ID_Y__" text-anchor="middle" fill="#fff" font-size="12" font-weight="950">ID __TRACK_ID__</text>
                    </g>

                    <!-- Predicted trajectory / conflict -->
                    <line class="trajectory" visibility="__CONFLICT_VIS__" x1="430" y1="__VEHICLE_Y__" x2="494" y2="414" stroke="#ff453a" stroke-width="5" marker-end="url(#arrowRed)"/>
                    <line class="trajectory" visibility="__NORMAL_PATH_VIS__" x1="430" y1="__VEHICLE_Y__" x2="430" y2="438" stroke="#4cc9f0" stroke-width="4" marker-end="url(#arrowBlue)"/>
                    <circle class="conflict-ring" visibility="__CONFLICT_VIS__" cx="494" cy="414" r="23" fill="rgba(255,69,58,.20)" stroke="#ff453a" stroke-width="4"/>
                    <text x="522" y="409" visibility="__CONFLICT_VIS__" fill="#ffd1cd" font-size="12" font-weight="900">PREDICTED</text>
                    <text x="522" y="425" visibility="__CONFLICT_VIS__" fill="#ffd1cd" font-size="12" font-weight="900">CONFLICT</text>

                    <!-- Distance indicator -->
                    <line x1="590" y1="435" x2="590" y2="__VEHICLE_Y__" stroke="#93c5fd" stroke-width="3" stroke-dasharray="6 5"/>
                    <line x1="581" y1="435" x2="599" y2="435" stroke="#93c5fd" stroke-width="3"/>
                    <line x1="581" y1="__VEHICLE_Y__" x2="599" y2="__VEHICLE_Y__" stroke="#93c5fd" stroke-width="3"/>
                    <rect x="604" y="__DIST_LABEL_Y__" width="92" height="34" rx="8" fill="#0b1b2a" stroke="#4b6882"/>
                    <text x="650" y="__DIST_TEXT_Y__" text-anchor="middle" fill="#e8f2fb" font-size="14" font-weight="900">__DISTANCE__ m</text>

                    <!-- HMI tag boxes -->
                    <g>
                        <rect x="16" y="16" width="188" height="118" rx="10" fill="#0b1b2a" stroke="#3a5872"/>
                        <text x="30" y="39" fill="#85a4bf" font-size="10" font-weight="900">MONITORED APPROACH</text>
                        <text x="30" y="62" fill="#fff" font-size="16" font-weight="950">__NS_STATE__</text>
                        <text x="30" y="85" fill="#85a4bf" font-size="10">Signal command</text>
                        <text x="30" y="105" fill="#fff" font-size="12" font-weight="900">__SIGNAL_COMMAND__</text>
                        <text x="30" y="124" fill="#85a4bf" font-size="10">Pedestrian advisory: __PED_SIGNAL__</text>
                    </g>

                    <g>
                        <rect x="656" y="490" width="188" height="101" rx="10" fill="#0b1b2a" stroke="#3a5872"/>
                        <text x="670" y="514" fill="#85a4bf" font-size="10" font-weight="900">LIVE AI TAGS</text>
                        <text x="670" y="537" fill="#fff" font-size="12">Speed: <tspan font-weight="900">__SPEED__ km/h</tspan></text>
                        <text x="670" y="558" fill="#fff" font-size="12">TTC: <tspan font-weight="900">__TTC__</tspan></text>
                        <text x="670" y="579" fill="#fff" font-size="12">Direction: <tspan font-weight="900">__DIRECTION__</tspan></text>
                    </g>

                    <!-- Intersection centre status -->
                    <rect x="326" y="266" width="208" height="78" rx="12" fill="#081523" stroke="__STATE_COLOR__" stroke-width="2"/>
                    <text x="430" y="290" text-anchor="middle" fill="#9ab1c7" font-size="10" font-weight="900">GUIDEKARO SAFETY PHASE</text>
                    <text x="430" y="315" text-anchor="middle" fill="#fff" font-size="15" font-weight="950">__PHASE__</text>
                    <text x="430" y="336" text-anchor="middle" fill="__STATE_COLOR__" font-size="13" font-weight="950">RISK __RISK__/100</text>
                </svg>
            </div>

            <div class="operator-panel">
                <div class="panel-heading">SCADA / HMI STATUS</div>
                <div class="phase-card">
                    <div class="phase-label">CURRENT TRAFFIC-LIGHT PHASE</div>
                    <div class="phase-value">__PHASE__</div>
                </div>
                <div class="risk-card">
                    <div class="risk-row">
                        <div><div class="risk-name">COMBINED AI RISK</div><div class="risk-number">__RISK__<span style="font-size:15px">/100</span></div></div>
                        <div style="font-weight:900;color:__STATE_COLOR__">__LEVEL__</div>
                    </div>
                    <div class="risk-track"><div class="risk-fill"></div></div>
                </div>
                <div class="kv"><div class="k">Tracked vehicle</div><div class="v">ID __TRACK_ID__</div></div>
                <div class="kv"><div class="k">Vehicles detected</div><div class="v">__VEHICLES__</div></div>
                <div class="kv"><div class="k">Pedestrians detected</div><div class="v">__PEDESTRIANS__</div></div>
                <div class="kv"><div class="k">Vehicle speed</div><div class="v">__SPEED__ km/h</div></div>
                <div class="kv"><div class="k">Crosswalk distance</div><div class="v">__DISTANCE__ m</div></div>
                <div class="kv"><div class="k">Pedestrian separation</div><div class="v">__SEPARATION__</div></div>
                <div class="kv"><div class="k">Time to collision</div><div class="v">__TTC__</div></div>
                <div class="kv"><div class="k">Trajectory conflict</div><div class="v">__CONFLICT__</div></div>
                <div class="reason"><b>WHY THIS STATE?</b><br>__REASON__</div>
                <div class="disclaimer">Traffic lamps in this dashboard are a GuideKaro SCADA-style advisory visualization. They do not directly control a municipal traffic signal unless a future approved controller integration is added.</div>
            </div>
        </div>
    </div>
    """

    replacements = {
        "__STATE_COLOR__": state_color,
        "__STATE_SOFT__": state_soft,
        "__LEVEL__": level,
        "__RISK__": f"{risk:.0f}",
        "__RISK_WIDTH__": f"{risk_width:.1f}",
        "__CAR_PERIOD__": f"{vehicle_period:.2f}",
        "__CAR_SHIFT__": f"{vehicle_shift:.0f}",
        "__VEHICLE_Y__": f"{vehicle_y:.0f}",
        "__CAR_TOP__": f"{vehicle_y - 27:.0f}",
        "__WINDOW_TOP__": f"{vehicle_y - 18:.0f}",
        "__WHEEL_Y__": f"{vehicle_y + 28:.0f}",
        "__ID_Y__": f"{vehicle_y + 4:.0f}",
        "__DIST_LABEL_Y__": f"{(435 + vehicle_y) / 2 - 17:.0f}",
        "__DIST_TEXT_Y__": f"{(435 + vehicle_y) / 2 + 5:.0f}",
        "__TRACK_ID__": track_text,
        "__SPEED__": f"{speed:.1f}",
        "__DISTANCE__": f"{distance:.1f}",
        "__SEPARATION__": separation_text,
        "__TTC__": ttc_text,
        "__DIRECTION__": direction_text,
        "__CONFLICT__": "YES" if conflict else "NO",
        "__VEHICLES__": str(vehicles),
        "__PEDESTRIANS__": str(pedestrians),
        "__REASON__": reason,
        "__PHASE__": html.escape(phase_name),
        "__SIGNAL_COMMAND__": html.escape(signal_command),
        "__PED_SIGNAL__": pedestrian_signal,
        "__NS_STATE__": ns_state,
        "__PED_VIS__": pedestrian_visibility,
        "__NO_PED_VIS__": no_ped_visibility,
        "__CONFLICT_VIS__": conflict_visibility,
        "__NORMAL_PATH_VIS__": normal_path_visibility,
        "__NS_RED__": ns_red,
        "__NS_YELLOW__": ns_yellow,
        "__NS_GREEN__": ns_green,
        "__EW_RED__": ew_red,
        "__EW_YELLOW__": ew_yellow,
        "__EW_GREEN__": ew_green,
        "__NS_RED_CLASS__": "signal-active" if ns_state == "RED" else "",
        "__NS_YELLOW_CLASS__": "signal-active" if ns_state == "YELLOW" else "",
        "__NS_GREEN_CLASS__": "signal-active" if ns_state == "GREEN" else "",
        "__EW_RED_CLASS__": "signal-active" if ew_state == "RED" else "",
        "__EW_YELLOW_CLASS__": "signal-active" if ew_state == "YELLOW" else "",
        "__EW_GREEN_CLASS__": "signal-active" if ew_state == "GREEN" else "",
    }

    hmi_html = template
    for token, value in replacements.items():
        hmi_html = hmi_html.replace(token, str(value))

    components.html(hmi_html, height=735, scrolling=False)

def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }

        .guidekaro-brand {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .guidekaro-subtitle {
            font-size: 1rem;
            opacity: 0.75;
            margin-bottom: 1.2rem;
        }

        .location-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 1rem;
        }

        .alert-card {
            border-radius: 22px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.2rem;
            text-align: center;
            border: 2px solid transparent;
        }

        .safe-card {
            background: rgba(46, 160, 67, 0.13);
            border-color: rgba(46, 160, 67, 0.65);
        }

        .warning-card {
            background: rgba(210, 153, 34, 0.15);
            border-color: rgba(210, 153, 34, 0.75);
        }

        .danger-card {
            background: rgba(248, 81, 73, 0.14);
            border-color: rgba(248, 81, 73, 0.78);
        }

        .alert-level {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
        }

        .alert-headline {
            font-size: 2.4rem;
            font-weight: 900;
            line-height: 1.05;
            margin: 0.3rem 0 0.5rem 0;
        }

        .alert-message {
            font-size: 1.08rem;
            opacity: 0.9;
        }

        .section-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            height: 100%;
        }

        .small-label {
            font-size: 0.85rem;
            opacity: 0.72;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            padding: 0.75rem;
            border-radius: 14px;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_user_dashboard(latest: pd.Series) -> None:
    risk = clean_number(latest["risk_score"])
    speed = clean_number(latest["vehicle_speed_kmh"])
    crosswalk_distance = clean_number(latest["distance_to_crosswalk_m"])
    separation = clean_number(latest["pedestrian_vehicle_distance_m"])
    pedestrian_count = int(clean_number(latest["pedestrian_count"]))
    vehicle_count = int(clean_number(latest["vehicle_count"]))
    ttc = latest["ttc_seconds"]

    profile = alert_profile(str(latest["status"]), risk, ttc)

    intersection = str(latest.get("intersection", "Current Crosswalk"))
    if intersection.lower() in {"nan", "none", ""}:
        intersection = "Current Crosswalk"

    timestamp = latest.get("timestamp")
    if pd.isna(timestamp):
        timestamp_text = "Live monitoring"
    else:
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    st.markdown(
        f"""
        <div class="location-card">
            <div class="small-label">CURRENT LOCATION</div>
            <div style="font-size:1.35rem;font-weight:750;">📍 {intersection}</div>
            <div class="small-label">Last update: {timestamp_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="alert-card {profile["class_name"]}">
            <div class="alert-level">{profile["icon"]} {profile["level"]}</div>
            <div class="alert-headline">{profile["headline"]}</div>
            <div class="alert-message">{profile["message"]}</div>
            <div style="margin-top:0.8rem;font-weight:700;">
                Current risk: {risk:.0f}/100
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Vehicle Speed", f"{speed:.1f} km/h")
    metric_columns[1].metric("Crosswalk Distance", f"{crosswalk_distance:.1f} m")
    metric_columns[2].metric("Pedestrian Separation", f"{separation:.1f} m")
    metric_columns[3].metric("Time to Collision", format_ttc(ttc))
    metric_columns[4].metric("Pedestrians Detected", pedestrian_count)

    st.markdown("### Four-Way Traffic SCADA Safety View")
    st.caption(
        "This PLC/SCADA-inspired HMI shows a four-way intersection with animated traffic lights. "
        "The monitored approach changes GREEN → YELLOW → RED according to GuideKaro risk, while vehicle position, TTC, pedestrian presence and predicted conflict update automatically."
    )
    render_scada_hmi(latest)

    st.markdown("### Live Crosswalk Camera")
    camera_col, action_col = st.columns([1.65, 1])

    with camera_col:
        frame = safe_image(FRAME_PATH)
        if frame is None:
            st.info(
                "Waiting for the GuideKaro video processor. "
                "Start enhanced_video_runner.py to display the live crosswalk frame."
            )
        else:
            st.image(
                frame,
                caption="GuideKaro live monitored crosswalk",
                use_container_width=True,
            )

    with action_col:
        st.markdown("#### What should I do?")

        if profile["level"] == "DANGER":
            st.error(
                "Stop or brake immediately if it is safe to do so. "
                "Yield to pedestrians and wait until the crosswalk is clear."
            )
        elif profile["level"] == "WARNING":
            st.warning(
                "Reduce speed, prepare to stop, and watch the crosswalk carefully."
            )
        else:
            st.success(
                "Continue carefully and remain alert for pedestrians entering the crosswalk."
            )

        st.markdown("#### Why am I seeing this alert?")
        st.write(build_reason(latest))

        st.markdown("#### Current Conditions")
        st.write(f"🌦️ **Weather:** {latest.get('weather', 'Unknown')}")
        st.write(f"🛣️ **Road:** {latest.get('road_condition', 'Unknown')}")
        st.write(f"👁️ **Visibility:** {latest.get('visibility', 'Unknown')}")
        st.write(f"🚗 **Vehicles detected:** {vehicle_count}")

    st.divider()

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(
            """
            <div class="section-card">
                <b>🚸 Pedestrian Awareness</b><br><br>
                Always yield to pedestrians already in or entering the crosswalk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col2:
        st.markdown(
            """
            <div class="section-card">
                <b>❄️ Winter Conditions</b><br><br>
                Snow, ice, and reduced visibility may increase stopping distance.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with info_col3:
        st.markdown(
            """
            <div class="section-card">
                <b>🤖 AI Assistance</b><br><br>
                GuideKaro supports driver awareness. The driver remains responsible for safe driving.
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    apply_styles()
    st_autorefresh(interval=2000, key="guidekaro-user-refresh")

    st.markdown(
        '<div class="guidekaro-brand">🚸 GuideKaro</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="guidekaro-subtitle">Real-time Crosswalk Safety Assistant</div>',
        unsafe_allow_html=True,
    )

    df = normalize(load_events())

    if df.empty:
        st.warning(
            "GuideKaro is waiting for crosswalk event data. "
            "Start enhanced_video_runner.py and keep it running."
        )
        st.markdown("### System Flow")
        st.code(
            "Camera / Video → AI Detection → Tracking → Risk Analysis → User Alert",
            language=None,
        )
        return

    latest = df.iloc[0]
    render_user_dashboard(latest)


if __name__ == "__main__":
    main()
