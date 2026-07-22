from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh


APP_TITLE = "GuideKaro – Intelligent Crosswalk Safety Dashboard"
BASE_DIR = Path(__file__).resolve().parent

INTERSECTION_CATALOG = [
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

st.set_page_config(
    page_title="GuideKaro Dashboard",
    page_icon="🚸",
    layout="wide",
    initial_sidebar_state="expanded",
)


COLUMN_ALIASES = {
    "event_time": "timestamp",
    "time": "timestamp",
    "created_at": "timestamp",
    "datetime": "timestamp",
    "location": "intersection",
    "site": "intersection",
    "camera_location": "intersection",
    "lat": "latitude",
    "lng": "longitude",
    "lon": "longitude",
    "speed": "vehicle_speed_kmh",
    "vehicle_speed": "vehicle_speed_kmh",
    "distance": "distance_to_crosswalk_m",
    "risk": "risk_score",
    "level": "risk_level",
    "event_status": "status",
    "blocked": "crosswalk_blocked",
    "latency": "response_time_ms",
    "fps_latency": "response_time_ms",
    "pedestrians": "pedestrian_count",
    "vehicles": "vehicle_count",
}

DEFAULTS = {
    "timestamp": pd.Timestamp.now(),
    "intersection": "Unknown intersection",
    "latitude": np.nan,
    "longitude": np.nan,
    "weather": "Unknown",
    "road_condition": "Unknown",
    "visibility": "Unknown",
    "vehicle_count": 0,
    "pedestrian_count": 0,
    "vehicle_speed_kmh": 0.0,
    "distance_to_crosswalk_m": 0.0,
    "confidence": 0.0,
    "risk_score": 0.0,
    "risk_level": "SAFE",
    "status": "SAFE",
    "crosswalk_blocked": 0,
    "alert_channel": "None",
    "response_time_ms": 0.0,
    "frame_path": "",
    "notes": "",
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.25);
            padding: 0.75rem;
            border-radius: 0.75rem;
            background: rgba(128,128,128,0.04);
        }
        .gk-banner {
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(128,128,128,0.25);
        }
        .gk-safe {background: rgba(0,150,80,0.12);}
        .gk-warning {background: rgba(255,180,0,0.16);}
        .gk-violation {background: rgba(220,40,40,0.14);}
        .small-note {font-size: 0.86rem; opacity: 0.8;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def db_candidates() -> list[Path]:
    configured = os.getenv("GUIDEKARO_DB_PATH")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            BASE_DIR / "database" / "uc1_events.db",
            BASE_DIR.parent / "database" / "uc1_events.db",
            BASE_DIR / "crosswalk.db",
            BASE_DIR.parent / "crosswalk.db",
        ]
    )
    return candidates


def resolve_db_path() -> Path:
    for candidate in db_candidates():
        if candidate.exists():
            return candidate.resolve()
    return (BASE_DIR / "database" / "uc1_events.db").resolve()


def available_tables(connection: sqlite3.Connection) -> list[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    return pd.read_sql_query(query, connection)["name"].tolist()


def load_database(db_path: Path) -> tuple[pd.DataFrame, str]:
    if not db_path.exists():
        return pd.DataFrame(), "Database file not found"

    try:
        with sqlite3.connect(db_path) as connection:
            tables = available_tables(connection)
            if not tables:
                return pd.DataFrame(), "Database has no tables"

            preferred = ["events", "uc1_events", "detections", "violations"]
            table = next((name for name in preferred if name in tables), tables[0])
            safe_table = table.replace('"', '""')
            df = pd.read_sql_query(
                f'SELECT * FROM "{safe_table}" ORDER BY ROWID DESC LIMIT 5000',
                connection,
            )
            return df, f'Connected to table "{table}"'
    except Exception as exc:
        return pd.DataFrame(), f"Database error: {exc}"


def normalize_data(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    df = raw.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    for column, default in DEFAULTS.items():
        if column not in df.columns:
            df[column] = default

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())

    numeric_cols = [
        "latitude",
        "longitude",
        "vehicle_count",
        "pedestrian_count",
        "vehicle_speed_kmh",
        "distance_to_crosswalk_m",
        "confidence",
        "risk_score",
        "crosswalk_blocked",
        "response_time_ms",
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["risk_level"] = df["risk_level"].astype(str).str.upper().str.strip()

    # Convert common status labels to GuideKaro's three dashboard states.
    status_map = {
        "ALERT": "VIOLATION",
        "HIGH": "VIOLATION",
        "DANGER": "VIOLATION",
        "HOLD": "VIOLATION",
        "WARN": "WARNING",
        "MEDIUM": "WARNING",
        "LOW": "SAFE",
        "NORMAL": "SAFE",
    }
    df["status"] = df["status"].replace(status_map)
    df["risk_level"] = df["risk_level"].replace(status_map)

    inferred_status = np.select(
        [
            (df["crosswalk_blocked"] > 0) | (df["risk_score"] >= 70),
            df["risk_score"].between(40, 69.999),
        ],
        ["VIOLATION", "WARNING"],
        default="SAFE",
    )
    invalid = ~df["status"].isin(["SAFE", "WARNING", "VIOLATION"])
    df.loc[invalid, "status"] = inferred_status[invalid]

    df["risk_level"] = np.where(
        df["risk_level"].isin(["SAFE", "WARNING", "VIOLATION"]),
        df["risk_level"],
        df["status"],
    )

    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def demo_data(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    end = pd.Timestamp.now().floor("min")
    timestamps = pd.date_range(end=end, periods=rows, freq="15min")
    intersections = [
        ("King St & Victoria St", 43.4516, -80.4925),
        ("Homer Watson & Block Line", 43.4202, -80.4752),
        ("Fairway Rd & Wilson Ave", 43.4255, -80.4386),
        ("Ottawa St & Fischer-Hallman", 43.4292, -80.5074),
        ("University Ave & King St", 43.4724, -80.5367),
    ]
    weather_options = ["Clear", "Snow", "Light snow", "Rain", "Fog"]
    road_options = ["Dry", "Wet", "Snow-covered", "Icy"]
    data = []

    for ts in timestamps:
        name, lat, lon = intersections[rng.integers(0, len(intersections))]
        weather = rng.choice(weather_options, p=[0.40, 0.18, 0.18, 0.16, 0.08])
        road = rng.choice(road_options, p=[0.42, 0.22, 0.22, 0.14])
        pedestrians = int(rng.integers(0, 5))
        vehicles = int(rng.integers(1, 12))
        speed = float(np.clip(rng.normal(42, 13), 4, 85))
        distance = float(np.clip(rng.normal(38, 22), 2, 120))
        winter_factor = 18 if road in {"Snow-covered", "Icy"} else 0
        pedestrian_factor = pedestrians * 7
        distance_factor = max(0, 35 - distance) * 0.75
        risk = float(np.clip(speed * 0.7 + winter_factor + pedestrian_factor + distance_factor - 18, 0, 100))
        blocked = int(risk > 78 and rng.random() < 0.55)
        status = "VIOLATION" if blocked or risk >= 70 else "WARNING" if risk >= 40 else "SAFE"

        data.append(
            {
                "id": len(data) + 1,
                "timestamp": ts,
                "intersection": name,
                "latitude": lat + rng.normal(0, 0.0004),
                "longitude": lon + rng.normal(0, 0.0004),
                "weather": weather,
                "road_condition": road,
                "visibility": "Low" if weather in {"Snow", "Fog"} else "Normal",
                "vehicle_count": vehicles,
                "pedestrian_count": pedestrians,
                "vehicle_speed_kmh": round(speed, 1),
                "distance_to_crosswalk_m": round(distance, 1),
                "confidence": round(float(rng.uniform(0.48, 0.97)), 2),
                "risk_score": round(risk, 1),
                "risk_level": status,
                "status": status,
                "crosswalk_blocked": blocked,
                "alert_channel": (
                    "Audio + Visual" if status == "VIOLATION"
                    else "Visual" if status == "WARNING"
                    else "None"
                ),
                "response_time_ms": round(float(np.clip(rng.normal(105, 35), 35, 280)), 1),
                "frame_path": "",
                "notes": "Demo event",
            }
        )
    return pd.DataFrame(data).sort_values("timestamp", ascending=False).reset_index(drop=True)


def filter_data(
    df: pd.DataFrame,
    start_date,
    end_date,
    intersections: Iterable[str],
    statuses: Iterable[str],
    weather: Iterable[str],
    road_conditions: Iterable[str],
) -> pd.DataFrame:
    result = df.copy()
    date_values = result["timestamp"].dt.date
    result = result[(date_values >= start_date) & (date_values <= end_date)]

    if intersections:
        result = result[result["intersection"].isin(intersections)]
    if statuses:
        result = result[result["status"].isin(statuses)]
    if weather:
        result = result[result["weather"].isin(weather)]
    if road_conditions:
        result = result[result["road_condition"].isin(road_conditions)]
    return result


def status_banner(latest: pd.Series) -> None:
    status = str(latest["status"]).upper()
    css_class = {
        "SAFE": "gk-safe",
        "WARNING": "gk-warning",
        "VIOLATION": "gk-violation",
    }.get(status, "gk-warning")

    recommendation = {
        "SAFE": "Crosswalk is currently clear. Continue passive monitoring.",
        "WARNING": "Pedestrian or vehicle risk is rising. Keep the visual warning active and prepare an audio alert.",
        "VIOLATION": "High-risk or crosswalk-blocking event detected. Log the event and keep the visual warning active until the crosswalk clears.",
    }.get(status, "Review the latest event.")

    st.markdown(
        f"""
        <div class="gk-banner {css_class}">
            <h2 style="margin:0;">Current System State: {status}</h2>
            <p style="margin:0.35rem 0 0 0;">{recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_frame(frame_value: str) -> Path | None:
    candidates = []
    if frame_value:
        p = Path(str(frame_value))
        candidates.extend([p, BASE_DIR / p, BASE_DIR.parent / p])
    candidates.extend(
        [
            BASE_DIR / "assets" / "latest_frame.jpg",
            BASE_DIR.parent / "dashboard" / "assets" / "latest_frame.jpg",
            BASE_DIR.parent / "outputs" / "latest_frame.jpg",
        ]
    )
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def safe_metric(label: str, value, suffix: str = "") -> None:
    if isinstance(value, (float, np.floating)):
        formatted = f"{value:,.1f}{suffix}"
    else:
        formatted = f"{value}{suffix}"
    st.metric(label, formatted)


def render_live(df: pd.DataFrame, db_path: Path, source_note: str) -> None:
    st.subheader("Live Crosswalk Monitoring")
    latest = df.iloc[0]
    status_banner(latest)

    metrics = st.columns(5)
    with metrics[0]:
        safe_metric("Risk score", float(latest["risk_score"]), "/100")
    with metrics[1]:
        safe_metric("Vehicle speed", float(latest["vehicle_speed_kmh"]), " km/h")
    with metrics[2]:
        safe_metric("Distance", float(latest["distance_to_crosswalk_m"]), " m")
    with metrics[3]:
        safe_metric("Pedestrians", int(latest["pedestrian_count"]))
    with metrics[4]:
        safe_metric("Processing latency", float(latest["response_time_ms"]), " ms")

    left, right = st.columns([1.8, 1])

    with left:
        st.markdown("#### Latest camera frame")
        frame = resolve_frame(str(latest.get("frame_path", "")))
        if frame:
            st.image(str(frame), use_container_width=True)
        else:
            st.info(
                "No current frame image was found. Save the latest annotated YOLO frame as "
                "`assets/latest_frame.jpg`, or store its path in the `frame_path` database column."
            )
            st.markdown(
                """
                **Expected overlay on the live video**

                Pedestrian and vehicle boxes · crosswalk ROI · confidence · speed · distance · risk score · current status
                """
            )

    with right:
        st.markdown("#### Latest decision")
        details = pd.DataFrame(
            {
                "Field": [
                    "Detected at",
                    "Intersection",
                    "Weather",
                    "Road condition",
                    "Confidence",
                    "Crosswalk blocked",
                    "Alert channel",
                ],
                "Value": [
                    latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    latest["intersection"],
                    latest["weather"],
                    latest["road_condition"],
                    f'{float(latest["confidence"]):.2f}',
                    "Yes" if int(latest["crosswalk_blocked"]) else "No",
                    latest["alert_channel"],
                ],
            }
        )
        st.dataframe(details, hide_index=True, use_container_width=True)

        st.markdown("#### False-alarm controls")
        st.write("✓ Detection must be inside the crosswalk region of interest")
        st.write("✓ Confidence threshold should be at least 0.45")
        st.write("✓ Confirm the event over 3–5 consecutive frames")
        st.write("✓ Track the same object instead of counting it repeatedly")
        st.write("✓ Stop repeated loud audio after a few seconds; keep the visual warning active")

    with st.expander("System connection details"):
        st.code(str(db_path))
        st.write(source_note)
        st.caption(
            "The dashboard refreshes data from the SQLite event database. "
            "The YOLO runner and this dashboard must point to the same database file."
        )


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Safety Overview")

    total = len(df)
    violations = int((df["status"] == "VIOLATION").sum())
    warnings = int((df["status"] == "WARNING").sum())
    blocked = int((df["crosswalk_blocked"] > 0).sum())
    avg_latency = float(df["response_time_ms"].mean()) if total else 0.0

    cols = st.columns(5)
    with cols[0]:
        safe_metric("Total events", total)
    with cols[1]:
        safe_metric("Warnings", warnings)
    with cols[2]:
        safe_metric("Violations", violations)
    with cols[3]:
        safe_metric("Blocked crosswalks", blocked)
    with cols[4]:
        safe_metric("Average latency", avg_latency, " ms")

    left, right = st.columns(2)
    with left:
        trend = (
            df.set_index("timestamp")
            .sort_index()
            .resample("1h")["risk_score"]
            .mean()
            .reset_index()
        )
        fig = px.line(
            trend,
            x="timestamp",
            y="risk_score",
            markers=True,
            title="Average risk score over time",
            labels={"timestamp": "Time", "risk_score": "Risk score"},
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        status_counts = df["status"].value_counts().rename_axis("status").reset_index(name="events")
        fig = px.bar(
            status_counts,
            x="status",
            y="events",
            title="Events by system state",
            labels={"status": "State", "events": "Number of events"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Latest priority events")
    priority = df[df["status"].isin(["WARNING", "VIOLATION"])].head(10)
    columns = [
        "timestamp",
        "intersection",
        "status",
        "risk_score",
        "vehicle_speed_kmh",
        "pedestrian_count",
        "weather",
        "road_condition",
    ]
    st.dataframe(priority[columns], hide_index=True, use_container_width=True)


def render_hotspots(df: pd.DataFrame) -> None:
    st.subheader("Crosswalk Hotspots")

    valid_map = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
        & ~((df["latitude"] == 0) & (df["longitude"] == 0))
    ].copy()

    summary = (
        df.groupby("intersection", as_index=False)
        .agg(
            events=("status", "size"),
            violations=("status", lambda s: int((s == "VIOLATION").sum())),
            warnings=("status", lambda s: int((s == "WARNING").sum())),
            average_risk=("risk_score", "mean"),
            blocked_events=("crosswalk_blocked", "sum"),
        )
        .sort_values(["violations", "average_risk"], ascending=False)
    )
    summary["average_risk"] = summary["average_risk"].round(1)

    if not valid_map.empty:
        locations = (
            valid_map.groupby("intersection", as_index=False)
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                events=("status", "size"),
                violations=("status", lambda s: int((s == "VIOLATION").sum())),
                average_risk=("risk_score", "mean"),
            )
        )
        fig = px.scatter_mapbox(
            locations,
            lat="latitude",
            lon="longitude",
            size="events",
            color="average_risk",
            hover_name="intersection",
            hover_data={"events": True, "violations": True, "average_risk": ":.1f"},
            zoom=10,
            height=520,
            title="Intersection risk map",
        )
        fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 45, "l": 0, "b": 0})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Map data is unavailable. Add `latitude` and `longitude` values to event records. "
            "The hotspot ranking below still works without coordinates."
        )

    st.markdown("#### Ranked intersections")
    st.dataframe(summary, hide_index=True, use_container_width=True)


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("Risk and Detection Analytics")

    left, right = st.columns(2)
    with left:
        weather_summary = (
            df.groupby("weather", as_index=False)
            .agg(average_risk=("risk_score", "mean"), events=("status", "size"))
            .sort_values("average_risk", ascending=False)
        )
        fig = px.bar(
            weather_summary,
            x="weather",
            y="average_risk",
            hover_data=["events"],
            title="Risk by weather condition",
            labels={"weather": "Weather", "average_risk": "Average risk score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        road_summary = (
            df.groupby("road_condition", as_index=False)
            .agg(average_risk=("risk_score", "mean"), violations=("status", lambda s: int((s == "VIOLATION").sum())))
            .sort_values("average_risk", ascending=False)
        )
        fig = px.bar(
            road_summary,
            x="road_condition",
            y="average_risk",
            hover_data=["violations"],
            title="Risk by road condition",
            labels={"road_condition": "Road condition", "average_risk": "Average risk score"},
        )
        st.plotly_chart(fig, use_container_width=True)

    analysis = df.copy()
    analysis["hour"] = analysis["timestamp"].dt.hour
    hourly = (
        analysis.groupby("hour", as_index=False)
        .agg(events=("status", "size"), violations=("status", lambda s: int((s == "VIOLATION").sum())))
    )
    fig = px.line(
        hourly,
        x="hour",
        y=["events", "violations"],
        markers=True,
        title="Event activity by hour of day",
        labels={"hour": "Hour of day", "value": "Events", "variable": "Series"},
    )
    st.plotly_chart(fig, use_container_width=True)

    perf_left, perf_right = st.columns(2)
    with perf_left:
        fig = px.histogram(
            df,
            x="confidence",
            nbins=20,
            title="Detection confidence distribution",
            labels={"confidence": "YOLO confidence"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with perf_right:
        fig = px.scatter(
            df,
            x="vehicle_speed_kmh",
            y="risk_score",
            size="pedestrian_count",
            hover_name="intersection",
            title="Vehicle speed versus risk",
            labels={"vehicle_speed_kmh": "Vehicle speed (km/h)", "risk_score": "Risk score"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_event_log(df: pd.DataFrame) -> None:
    st.subheader("Event Log and Evidence")
    search = st.text_input("Search intersection, status, weather, or notes")
    display = df.copy()

    if search.strip():
        text = search.strip().lower()
        mask = pd.Series(False, index=display.index)
        for column in ["intersection", "status", "weather", "road_condition", "notes"]:
            mask = mask | display[column].astype(str).str.lower().str.contains(text, na=False)
        display = display[mask]

    preferred_columns = [
        "id",
        "timestamp",
        "intersection",
        "status",
        "risk_score",
        "vehicle_speed_kmh",
        "distance_to_crosswalk_m",
        "pedestrian_count",
        "vehicle_count",
        "weather",
        "road_condition",
        "confidence",
        "crosswalk_blocked",
        "alert_channel",
        "response_time_ms",
        "notes",
    ]
    columns = [column for column in preferred_columns if column in display.columns]
    st.dataframe(display[columns], hide_index=True, use_container_width=True, height=520)

    csv = display[columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered event log (CSV)",
        data=csv,
        file_name=f"guidekaro_events_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


def build_report_html(df: pd.DataFrame) -> str:
    total = len(df)
    violations = int((df["status"] == "VIOLATION").sum())
    warnings = int((df["status"] == "WARNING").sum())
    blocked = int((df["crosswalk_blocked"] > 0).sum())
    average_risk = float(df["risk_score"].mean()) if total else 0.0
    top = (
        df.groupby("intersection")
        .agg(events=("status", "size"), violations=("status", lambda s: int((s == "VIOLATION").sum())), average_risk=("risk_score", "mean"))
        .sort_values(["violations", "average_risk"], ascending=False)
        .head(10)
        .round(1)
    )

    return f"""
    <html>
    <head><meta charset="utf-8"><title>GuideKaro Safety Report</title></head>
    <body style="font-family:Arial,sans-serif;max-width:900px;margin:40px auto;line-height:1.5;">
      <h1>GuideKaro Crosswalk Safety Report</h1>
      <p>Generated: {datetime.now():%Y-%m-%d %H:%M}</p>
      <h2>Summary</h2>
      <ul>
        <li>Total recorded events: {total}</li>
        <li>Warnings: {warnings}</li>
        <li>Violations/high-risk events: {violations}</li>
        <li>Crosswalk-blocking events: {blocked}</li>
        <li>Average risk score: {average_risk:.1f}/100</li>
      </ul>
      <h2>Highest-priority intersections</h2>
      {top.to_html()}
      <h2>Interpretation</h2>
      <p>
        Intersections with repeated violations, high average risk, winter road conditions,
        or crosswalk-blocking events should be prioritized for field review and targeted safety action.
      </p>
    </body>
    </html>
    """


def render_reports(df: pd.DataFrame) -> None:
    st.subheader("Reports and Export")
    st.write(
        "Create a filtered city-safety report using the date, intersection, weather, road-condition, "
        "and status filters selected in the sidebar."
    )

    total = len(df)
    violations = int((df["status"] == "VIOLATION").sum())
    violation_rate = (violations / total * 100) if total else 0
    average_risk = float(df["risk_score"].mean()) if total else 0.0

    cols = st.columns(3)
    with cols[0]:
        safe_metric("Events in report", total)
    with cols[1]:
        safe_metric("Violation rate", violation_rate, "%")
    with cols[2]:
        safe_metric("Average risk", average_risk, "/100")

    report_html = build_report_html(df)
    st.download_button(
        "Download city safety report (HTML)",
        data=report_html.encode("utf-8"),
        file_name=f"guidekaro_safety_report_{datetime.now():%Y%m%d_%H%M}.html",
        mime="text/html",
    )

    st.markdown("#### Recommended city follow-up")
    st.write(
        "Prioritize intersections with repeated violations, winter-road risk, high pedestrian activity, "
        "or frequent crosswalk blocking. Review the evidence before changing signal timing, signage, "
        "snow clearing, enforcement, or road design."
    )


def main() -> None:
    inject_css()
    st.title(APP_TITLE)
    st.caption(
        "Camera → YOLO/OpenCV detection → risk decision engine → alerts, event logging, and city analytics"
    )

    db_path = resolve_db_path()

    with st.sidebar:
        st.header("Dashboard controls")
        auto_refresh = st.toggle("Auto-refresh live data", value=True)
        refresh_seconds = st.select_slider(
            "Refresh interval",
            options=[3, 5, 10, 15, 30],
            value=5,
            disabled=not auto_refresh,
        )
        if auto_refresh:
            st_autorefresh(interval=refresh_seconds * 1000, key="guidekaro_refresh")

        if st.button("Refresh now", use_container_width=True):
            st.rerun()

    raw, connection_note = load_database(db_path)
    use_demo = raw.empty
    df = demo_data() if use_demo else normalize_data(raw)

    if df.empty:
        st.error("No usable event data was found.")
        st.stop()

    with st.sidebar:
        if use_demo:
            st.warning("Demo mode: no usable event database was found.")
        else:
            st.success("Live database connected")

        min_date = df["timestamp"].min().date()
        max_date = df["timestamp"].max().date()
        selected_dates = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
        else:
            start_date = end_date = selected_dates

        selected_intersections = st.multiselect(
            "Intersection",
            options=sorted(
                set(INTERSECTION_CATALOG)
                | set(df["intersection"].dropna().astype(str).unique())
            ),
        )
        st.caption(
            "The full location catalogue stays visible. Locations without current "
            "database events will return no records when selected."
        )
        selected_statuses = st.multiselect(
            "System state",
            options=["SAFE", "WARNING", "VIOLATION"],
            default=["SAFE", "WARNING", "VIOLATION"],
        )
        selected_weather = st.multiselect(
            "Weather",
            options=sorted(df["weather"].dropna().astype(str).unique()),
        )
        selected_road = st.multiselect(
            "Road condition",
            options=sorted(df["road_condition"].dropna().astype(str).unique()),
        )

        st.divider()
        st.caption(f"Database: {db_path.name}")
        st.caption(connection_note)

    filtered = filter_data(
        df,
        start_date,
        end_date,
        selected_intersections,
        selected_statuses,
        selected_weather,
        selected_road,
    )

    if filtered.empty:
        st.warning("No events match the current filters.")
        st.stop()

    tabs = st.tabs(
        [
            "Live Monitor",
            "Overview",
            "Hotspots",
            "Analytics",
            "Event Log",
            "Reports",
        ]
    )

    with tabs[0]:
        render_live(filtered, db_path, connection_note)
    with tabs[1]:
        render_overview(filtered)
    with tabs[2]:
        render_hotspots(filtered)
    with tabs[3]:
        render_analytics(filtered)
    with tabs[4]:
        render_event_log(filtered)
    with tabs[5]:
        render_reports(filtered)


if __name__ == "__main__":
    main()