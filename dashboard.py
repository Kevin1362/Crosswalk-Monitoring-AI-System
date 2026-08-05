from __future__ import annotations
from core.config import load_settings
from core.logging_setup import configure_logging

import io
import os
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_autorefresh import st_autorefresh


BASE_DIR = Path(__file__).resolve().parent
SETTINGS = load_settings(BASE_DIR / "config" / "settings.yaml")
LOGGER = configure_logging(BASE_DIR / "logs" / "guidekaro.log")
DB_PATH = BASE_DIR / "database" / "uc1_events.db"
FRAME_PATH = BASE_DIR / "assets" / "latest_frame.jpg"


# Sensitive values are read from environment variables instead of being
# hardcoded in the source code. The dashboard never displays or logs the
# actual DB password.
def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except ValueError:
        LOGGER.warning(
            "Invalid numeric environment variable %s; using default %s",
            name,
            default,
        )
        return float(default)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return int(default)
    try:
        return int(value)
    except ValueError:
        LOGGER.warning(
            "Invalid integer environment variable %s; using default %s",
            name,
            default,
        )
        return int(default)


DEFAULT_FEATURES = (
    "risk_score,vehicle_speed_kmh,distance_to_crosswalk_m,"
    "pedestrian_vehicle_distance_m,confidence,ttc_seconds"
)

FEATURE_NAMES = [
    item.strip()
    for item in os.getenv("FEATURE_NAMES", DEFAULT_FEATURES).split(",")
    if item.strip()
]

MODEL_CONFIDENCE_THRESHOLD = _env_float(
    "MODEL_CONFIDENCE_THRESHOLD",
    SETTINGS.get("detection", {}).get("confidence", 0.35),
)

PIXELS_PER_METER = _env_float(
    "PIXELS_PER_METER",
    SETTINGS.get("tracking", {}).get("pixels_per_meter", 22.0),
)

EXPECTED_ACCURACY = _env_float("EXPECTED_ACCURACY", 0.90)
NUM_EPOCHS = _env_int("NUM_EPOCHS", 50)
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "guidekaro_enhanced")
EXPERIMENT_VERSION = os.getenv("EXPERIMENT_VERSION", "1.0")

# Workshop DB variables. GuideKaro currently stores events in SQLite, but
# these variables demonstrate Docker secrets/environment-variable management.
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

st.set_page_config(
    page_title="GuideKaro Enhanced Dashboard",
    page_icon="🚸",
    layout="wide",
    initial_sidebar_state="expanded",
)

INTERSECTIONS = [
    "GuideKaro Winter Presentation Intersection",
    "GuideKaro Presentation Crosswalk",
    "King St & Victoria St",
    "Homer Watson Blvd & Block Line Rd",
    "Fairway Rd & Wilson Ave",
    "Ottawa St & Fischer-Hallman Rd",
    "University Ave & King St",
    "King St & Frederick St",
    "Weber St & Victoria St",
    "Courtland Ave & Block Line Rd",
    "Highland Rd & Westmount Rd",
    "Hespeler Rd & Pinebush Rd",
]


def load_events() -> pd.DataFrame:
    if not DB_PATH.exists():
        LOGGER.warning("Event database not found at %s", DB_PATH)
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as connection:
            return pd.read_sql_query(
                "SELECT * FROM events ORDER BY id DESC LIMIT 5000",
                connection,
            )
    except Exception as exc:
        LOGGER.exception("Unable to read GuideKaro event database")
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    defaults = {
        "timestamp": "",
        "intersection": "Unknown",
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
    for col, value in defaults.items():
        if col not in df:
            df[col] = value
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric = [
        "vehicle_count", "pedestrian_count", "vehicle_speed_kmh",
        "distance_to_crosswalk_m", "pedestrian_vehicle_distance_m",
        "confidence", "risk_score", "response_time_ms", "fps",
        "trajectory_conflict", "ttc_seconds",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].fillna("SAFE").astype(str).str.upper()
    return df


def safe_image(path: Path):
    if not path.exists():
        return None
    for _ in range(3):
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                return image.convert("RGB").copy()
        except (UnidentifiedImageError, OSError, PermissionError):
            continue
    return None


def status_banner(status: str, risk: float) -> None:
    st.markdown(f"### Current state: **{status}** — Risk **{risk:.1f}/100**")


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    with st.sidebar:
        st.header("GuideKaro controls")
        refresh = st.toggle("Auto-refresh", True)
        if refresh:
            st_autorefresh(interval=3000, key="refresh")
        choices = sorted(set(INTERSECTIONS) | set(df["intersection"].dropna().astype(str)))
        selected = st.multiselect("Intersection", choices)
        states = st.multiselect("Status", ["SAFE", "WARNING", "VIOLATION"], default=["SAFE", "WARNING", "VIOLATION"])
        directions = st.multiselect(
            "Movement direction",
            sorted(df["movement_direction"].dropna().astype(str).unique()),
        )
        st.caption("Configured intersections stay visible even before events are recorded.")

    result = df.copy()
    if selected:
        result = result[result["intersection"].isin(selected)]
    if states:
        result = result[result["status"].isin(states)]
    if directions:
        result = result[result["movement_direction"].isin(directions)]
    return result


def render_live(df: pd.DataFrame) -> None:
    latest = df.iloc[0]
    status_banner(str(latest["status"]), float(latest["risk_score"] or 0))

    cols = st.columns(8)
    cols[0].metric("Risk", f'{latest["risk_score"]:.1f}/100')
    cols[1].metric("Vehicle speed", f'{latest["vehicle_speed_kmh"]:.1f} km/h')
    cols[2].metric("Crosswalk distance", f'{latest["distance_to_crosswalk_m"]:.1f} m')
    cols[3].metric("Pedestrian separation", f'{latest["pedestrian_vehicle_distance_m"]:.1f} m')
    cols[4].metric("Track ID", "N/A" if pd.isna(latest["track_id"]) else int(latest["track_id"]))
    cols[5].metric("TTC", "N/A" if pd.isna(latest["ttc_seconds"]) else f'{latest["ttc_seconds"]:.1f} s')
    cols[6].metric("FPS", f'{latest["fps"]:.1f}')
    cols[7].metric("Latency", f'{latest["response_time_ms"]:.1f} ms')

    left, right = st.columns([1.7, 1])
    with left:
        st.markdown("#### Live tracked frame")
        image = safe_image(FRAME_PATH)
        if image is None:
            st.info("Start enhanced_video_runner.py to generate assets/latest_frame.jpg.")
        else:
            st.image(image, use_container_width=True)
    with right:
        st.markdown("#### Highest-risk tracked object")
        details = {
            "Track ID": latest["track_id"],
            "Object": latest.get("object_class", ""),
            "Movement": latest["movement_direction"],
            "Trajectory conflict": bool(latest["trajectory_conflict"]),
            "Confidence": latest["confidence"],
            "Reasons": latest["notes"],
        }
        st.json(details)

        st.markdown("#### Feature-based decision")
        st.write(
            "The score uses speed, distance to crosswalk, pedestrian–vehicle separation, "
            "movement direction, predicted path conflict, crosswalk occupancy, TTC, confidence, "
            "and environment flags."
        )


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("Traffic behavior analytics")
    work = df.sort_values("timestamp").copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(work, x="timestamp", y="risk_score", color="status", title="Risk score over time")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(
            work,
            x="distance_to_crosswalk_m",
            y="vehicle_speed_kmh",
            color="status",
            size="risk_score",
            hover_data=["track_id", "movement_direction", "ttc_seconds"],
            title="Speed vs distance to crosswalk",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        direction_counts = work["movement_direction"].fillna("UNKNOWN").value_counts().reset_index()
        direction_counts.columns = ["direction", "events"]
        st.plotly_chart(
            px.bar(direction_counts, x="direction", y="events", title="Movement direction distribution"),
            use_container_width=True,
        )
    with c4:
        fig = px.histogram(
            work,
            x="pedestrian_vehicle_distance_m",
            color="status",
            nbins=20,
            title="Pedestrian–vehicle separation distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "track_id" in work:
        top_tracks = (
            work.dropna(subset=["track_id"])
            .groupby("track_id", as_index=False)
            .agg(
                max_risk=("risk_score", "max"),
                max_speed=("vehicle_speed_kmh", "max"),
                min_crosswalk_distance=("distance_to_crosswalk_m", "min"),
                min_separation=("pedestrian_vehicle_distance_m", "min"),
                samples=("id", "count"),
            )
            .sort_values("max_risk", ascending=False)
            .head(20)
        )
        st.markdown("#### Highest-risk tracked objects")
        st.dataframe(top_tracks, hide_index=True, use_container_width=True)


def render_performance(df: pd.DataFrame) -> None:
    st.subheader("System performance")
    cols = st.columns(4)
    cols[0].metric("Average FPS", f'{df["fps"].mean():.1f}')
    cols[1].metric("Average latency", f'{df["response_time_ms"].mean():.1f} ms')
    cols[2].metric("P95 latency", f'{df["response_time_ms"].quantile(0.95):.1f} ms')
    cols[3].metric("Tracked IDs", int(df["track_id"].dropna().nunique()))

    st.plotly_chart(
        px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y=["fps", "response_time_ms"],
            title="Runtime performance trend",
        ),
        use_container_width=True,
    )
    st.info(
        "Precision and recall require labelled ground-truth frames. Run evaluation/evaluate_video.py "
        "with --labels-dir to calculate detection precision/recall. Without annotations, the project "
        "reports runtime FPS and latency only."
    )


def render_failure_cases() -> None:
    st.subheader("Failure-case validation")
    cases = pd.DataFrame(
        [
            ["Poor lighting", "Test night / underexposed video", "Lower confidence, missed pedestrians", "Use varied training/validation data; tune confidence"],
            ["Heavy rain / snow", "Rain or snow obstructs camera", "Blur and false detections", "Weather-aware thresholds; clean lens; robust datasets"],
            ["Occluded pedestrians", "Pedestrian partly hidden by vehicle", "Late detection", "Tracking maintains identity across temporary occlusion"],
            ["Multiple overlapping vehicles", "Dense traffic", "ID switches", "ByteTrack tuning; higher-resolution camera"],
            ["Motion blur", "Fast vehicle / low shutter speed", "Poor boxes and tracking", "Improve camera exposure; use stronger model"],
            ["Camera shake", "Vibration / handheld feed", "False motion and unstable trajectories", "Fixed mounting; stabilization"],
        ],
        columns=["Failure case", "Test", "Expected issue", "Mitigation"],
    )
    st.dataframe(cases, hide_index=True, use_container_width=True)


def render_reports(df: pd.DataFrame) -> None:
    st.subheader("Reporting and export")
    report = df.copy()
    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download event analytics CSV",
        csv,
        "guidekaro_enhanced_events.csv",
        "text/csv",
    )

    summary = {
        "events": len(df),
        "safe": int((df["status"] == "SAFE").sum()),
        "warnings": int((df["status"] == "WARNING").sum()),
        "violations": int((df["status"] == "VIOLATION").sum()),
        "unique_tracks": int(df["track_id"].dropna().nunique()),
        "average_fps": round(float(df["fps"].mean()), 2),
        "average_latency_ms": round(float(df["response_time_ms"].mean()), 2),
        "maximum_risk": round(float(df["risk_score"].max()), 2),
    }
    st.json(summary)



def render_assignment_configuration() -> None:
    """Show safe assignment configuration without exposing secret values."""
    with st.sidebar:
        st.divider()
        st.subheader("Experiment Configuration")
        st.write(f"**Experiment:** {EXPERIMENT_NAME}")
        st.write(f"**Version:** {EXPERIMENT_VERSION}")
        st.write(f"**Expected accuracy target:** {EXPECTED_ACCURACY:.0%}")
        st.write(f"**Training epochs:** {NUM_EPOCHS}")
        st.write(f"**Detection confidence:** {MODEL_CONFIDENCE_THRESHOLD:.2f}")
        st.write(f"**Configured features:** {len(FEATURE_NAMES)}")

        with st.expander("Secrets Management Status"):
            db_user_configured = bool(DB_USER)
            db_password_configured = bool(DB_PASSWORD)

            st.write(
                "DB username: "
                + ("✅ Configured" if db_user_configured else "⚠️ Not configured")
            )
            st.write(
                "DB password: "
                + ("✅ Configured securely" if db_password_configured else "⚠️ Not configured")
            )
            st.write(f"DB host: {DB_HOST}")
            st.write(f"DB port: {DB_PORT}")
            st.caption(
                "Sensitive credentials are read from environment variables. "
                "The password value is never displayed."
            )

        st.caption(
            "Expected accuracy and epoch values are experiment configuration "
            "targets, not measured runtime model accuracy."
        )

def main() -> None:
    st.title("GuideKaro — Enhanced AI Crosswalk Safety Dashboard")
    st.caption(
        "Tracking • speed estimation • trajectory prediction • feature-based risk analysis • analytics • reporting"
    )

    # Log startup once per Streamlit browser session. Only configuration
    # presence is logged; secret values are never written to the log.
    if "assignment9_startup_logged" not in st.session_state:
        LOGGER.info(
            "GuideKaro dashboard started | experiment=%s | version=%s",
            EXPERIMENT_NAME,
            EXPERIMENT_VERSION,
        )
        LOGGER.info(
            "Environment configuration loaded | "
            "db_user_configured=%s | db_password_configured=%s | "
            "db_host=%s | db_port=%s | feature_count=%s | "
            "confidence_threshold=%.2f | expected_accuracy=%.2f | epochs=%s",
            bool(DB_USER),
            bool(DB_PASSWORD),
            DB_HOST,
            DB_PORT,
            len(FEATURE_NAMES),
            MODEL_CONFIDENCE_THRESHOLD,
            EXPECTED_ACCURACY,
            NUM_EPOCHS,
        )
        st.session_state["assignment9_startup_logged"] = True

    render_assignment_configuration()

    df = normalize(load_events())
    if df.empty:
        st.warning(
            "No events found yet. Run enhanced_video_runner.py first. "
            "The dashboard will refresh automatically after data is recorded."
        )
        return

    df = filter_df(df)
    if df.empty:
        st.info("No records match the selected filters.")
        return

    tabs = st.tabs(
        ["Live Monitor", "Analytics", "Performance", "Failure Cases", "Event Log", "Reports"]
    )
    with tabs[0]:
        render_live(df)
    with tabs[1]:
        render_analytics(df)
    with tabs[2]:
        render_performance(df)
    with tabs[3]:
        render_failure_cases()
    with tabs[4]:
        st.dataframe(df, hide_index=True, use_container_width=True, height=620)
    with tabs[5]:
        render_reports(df)


if __name__ == "__main__":
    main()