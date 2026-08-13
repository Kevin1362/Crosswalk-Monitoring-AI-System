from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_autorefresh import st_autorefresh

# Use the same risk engine as the video runner whenever it is available.
# The fallback constants exactly match the supplied risk_engine.py/settings.yaml.
try:
    from core.config import load_settings
    from core.risk_engine import RiskEngine, RiskFeatures
except Exception:  # Keeps the dashboard usable when opened as a standalone file.
    load_settings = None
    RiskEngine = None
    RiskFeatures = None


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "uc1_events.db"
FRAME_PATH = BASE_DIR / "assets" / "latest_frame.jpg"
SETTINGS_PATH = BASE_DIR / "config" / "settings.yaml"

st.set_page_config(
    page_title="GuideKaro Enhanced Dashboard",
    page_icon="🚸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentation-friendly styling.
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.35rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    div[data-testid="stMetricLabel"] {font-size: 1rem; font-weight: 650;}
    .risk-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 6px 0 12px 0;
        background: rgba(128,128,128,.055);
    }
    .risk-card h3 {margin-top: 0; margin-bottom: .35rem;}
    .risk-card p {font-size: 1.02rem; margin-bottom: .25rem;}
    .evidence-box {
        border-left: 5px solid #6c757d;
        padding: 10px 14px;
        margin: 7px 0;
        border-radius: 5px;
        background: rgba(128,128,128,.055);
    }
    </style>
    """,
    unsafe_allow_html=True,
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

# Exact values from the provided settings.yaml. These are used only when the
# project's core RiskEngine cannot be imported.
FALLBACK_RISK = {
    "safe_max": 39.0,
    "warning_max": 69.0,
    "reference_speed_kmh": 60.0,
    "reference_crosswalk_distance_m": 20.0,
    "reference_separation_m": 12.0,
    "reference_ttc_s": 5.0,
    "weights": {
        "confidence": 0.08,
        "speed": 0.18,
        "crosswalk_distance": 0.18,
        "separation": 0.20,
        "direction": 0.10,
        "trajectory": 0.12,
        "occupancy": 0.08,
        "ttc": 0.04,
        "environment": 0.02,
    },
}

DISPLAY_NAMES = {
    "confidence": "Detection confidence",
    "speed": "Vehicle speed",
    "crosswalk_distance": "Near crosswalk",
    "separation": "Near pedestrian",
    "direction": "Moving toward crosswalk",
    "trajectory": "Predicted path conflict",
    "occupancy": "Crosswalk occupied",
    "ttc": "Low time-to-collision (TTC)",
    "environment": "Weather / visibility",
}


def load_events() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as connection:
            return pd.read_sql_query(
                "SELECT * FROM events ORDER BY id DESC LIMIT 5000",
                connection,
            )
    except Exception as exc:
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
        "object_class": "",
        "movement_direction": "UNKNOWN",
        "trajectory_conflict": 0,
        "ttc_seconds": None,
        "crosswalk_blocked": 0,
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
        "trajectory_conflict", "ttc_seconds", "crosswalk_blocked",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["status"] = df["status"].fillna("SAFE").astype(str).str.upper()
    df["notes"] = df["notes"].fillna("").astype(str)
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


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def risk_thresholds() -> tuple[float, float]:
    if load_settings is not None and RiskEngine is not None and SETTINGS_PATH.exists():
        try:
            engine = RiskEngine(load_settings(SETTINGS_PATH))
            return float(engine.safe_max), float(engine.warning_max)
        except Exception:
            pass
    return FALLBACK_RISK["safe_max"], FALLBACK_RISK["warning_max"]


def build_risk_breakdown(row: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Recreate the same feature normalization used by core/risk_engine.py.

    The event log stores the final risk score and human-readable reasons. This
    function exposes the individual weighted contributions so a city user can
    see *why* the selected object received that score.
    """
    notes = str(row.get("notes", "") or "")
    road_condition = str(row.get("road_condition", "") or "").lower()
    visibility = str(row.get("visibility", "") or "").lower()

    speed_kmh = safe_float(row.get("vehicle_speed_kmh"))
    crosswalk_distance = safe_float(row.get("distance_to_crosswalk_m"), 99.0)
    separation = safe_float(row.get("pedestrian_vehicle_distance_m"), 99.0)
    confidence = safe_float(row.get("confidence"))
    ttc_raw = row.get("ttc_seconds")
    ttc = None if pd.isna(ttc_raw) else safe_float(ttc_raw)
    toward = str(row.get("movement_direction", "")).upper() == "TOWARD_CROSSWALK"
    conflict = bool(safe_float(row.get("trajectory_conflict")))
    # The event logger stores whether any crosswalk is blocked, while the risk
    # engine's reasons identify whether the selected highest-risk vehicle itself
    # occupied it. Prefer the selected-object reason when available.
    occupied = "crosswalk occupied" in notes.lower()
    adverse_weather = road_condition in {"wet", "icy", "snow", "snow-covered"}
    low_visibility = visibility in {"low", "reduced", "poor"}

    if load_settings is not None and RiskEngine is not None and RiskFeatures is not None and SETTINGS_PATH.exists():
        try:
            engine = RiskEngine(load_settings(SETTINGS_PATH))
            decision = engine.evaluate(
                RiskFeatures(
                    confidence=confidence,
                    speed_kmh=speed_kmh,
                    distance_to_crosswalk_m=crosswalk_distance,
                    pedestrian_vehicle_distance_m=separation,
                    moving_toward_crosswalk=toward,
                    trajectory_conflict=conflict,
                    crosswalk_occupied=occupied,
                    ttc_seconds=ttc,
                    adverse_weather=adverse_weather,
                    low_visibility=low_visibility,
                )
            )
            components = decision.components
            weights = engine.weights
        except Exception:
            components = None
            weights = None
    else:
        components = None
        weights = None

    if components is None or weights is None:
        cfg = FALLBACK_RISK
        components = {
            "confidence": clamp01(confidence),
            "speed": clamp01(speed_kmh / cfg["reference_speed_kmh"]),
            "crosswalk_distance": clamp01(1.0 - crosswalk_distance / cfg["reference_crosswalk_distance_m"]),
            "separation": clamp01(1.0 - separation / cfg["reference_separation_m"]),
            "direction": 1.0 if toward else 0.0,
            "trajectory": 1.0 if conflict else 0.0,
            "occupancy": 1.0 if occupied else 0.0,
            "ttc": 0.0 if ttc is None else clamp01(1.0 - ttc / cfg["reference_ttc_s"]),
            "environment": 0.5 * float(adverse_weather) + 0.5 * float(low_visibility),
        }
        weights = cfg["weights"]

    total_weight = sum(float(weights.get(name, 0.0)) for name in components) or 1.0
    values = {
        "confidence": f"{confidence * 100:.0f}%",
        "speed": f"{speed_kmh:.1f} km/h",
        "crosswalk_distance": f"{crosswalk_distance:.1f} m",
        "separation": f"{separation:.1f} m",
        "direction": "Yes" if toward else "No",
        "trajectory": "Yes" if conflict else "No",
        "occupancy": "Yes" if occupied else "No",
        "ttc": "N/A" if ttc is None else f"{ttc:.1f} s",
        "environment": "Adverse" if adverse_weather or low_visibility else "Normal",
    }

    rows = []
    for name, normalized in components.items():
        weight = float(weights.get(name, 0.0))
        points = 100.0 * float(normalized) * weight / total_weight
        rows.append(
            {
                "Risk factor": DISPLAY_NAMES[name],
                "Current evidence": values[name],
                "Weight": f"{weight * 100:.0f}%",
                "Contribution": round(points, 1),
            }
        )

    breakdown = pd.DataFrame(rows).sort_values("Contribution", ascending=False)
    reasons = [part.strip() for part in notes.split(";") if part.strip()]
    return breakdown, reasons


def status_banner(status: str, risk: float) -> None:
    safe_max, warning_max = risk_thresholds()
    status = status.upper()
    palette = {
        "SAFE": ("#198754", "Routine monitoring"),
        "WARNING": ("#d97706", "Elevated risk — review this object"),
        "VIOLATION": ("#dc3545", "High risk — priority review"),
    }
    color, message = palette.get(status, ("#6c757d", "Review current event"))
    threshold_text = (
        f"SAFE ≤ {safe_max:.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"WARNING {safe_max + 1:.0f}–{warning_max:.0f} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"VIOLATION ≥ {warning_max + 1:.0f}"
    )
    st.markdown(
        f"""
        <div class="risk-card" style="border-left: 8px solid {color};">
            <h3 style="color:{color};">{status} — Risk {risk:.1f}/100</h3>
            <p><b>City interpretation:</b> {message}</p>
            <p style="opacity:.82;"><b>Prototype thresholds:</b> {threshold_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig):
    """Apply large, projector-friendly fonts to every Plotly chart."""
    fig.update_layout(
        font=dict(size=21),
        title=dict(font=dict(size=30), x=0.02, xanchor="left"),
        legend=dict(
            font=dict(size=19),
            title_font=dict(size=20),
            itemsizing="constant",
        ),
        hoverlabel=dict(font_size=18),
        margin=dict(l=85, r=55, t=95, b=85),
    )
    fig.update_xaxes(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(size=22),
        tickfont=dict(size=18),
        automargin=True,
    )
    # This also enlarges text labels/annotations used by bar charts and other plots.
    fig.update_traces(textfont=dict(size=18))
    fig.update_annotations(font=dict(size=18))
    return fig


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
        states = st.multiselect(
            "Risk status",
            ["SAFE", "WARNING", "VIOLATION"],
            default=["SAFE", "WARNING", "VIOLATION"],
        )
        directions = st.multiselect(
            "Movement direction",
            sorted(df["movement_direction"].dropna().astype(str).unique()),
        )
        st.caption("Use these filters to focus on a specific intersection, risk state, or direction.")

    result = df.copy()
    if selected:
        result = result[result["intersection"].isin(selected)]
    if states:
        result = result[result["status"].isin(states)]
    if directions:
        result = result[result["movement_direction"].isin(directions)]
    return result


def render_city_risk_explanation(latest: pd.Series) -> None:
    """Show city authorities exactly why the selected object is considered risky."""
    risk = safe_float(latest.get("risk_score"))
    status = str(latest.get("status", "SAFE")).upper()
    safe_max, warning_max = risk_thresholds()
    breakdown, reasons = build_risk_breakdown(latest)

    st.markdown("### 🏙️ Why is this object considered high risk?")
    st.caption(
        "This panel exposes the evidence behind the Risk Engine. The larger the contribution, "
        "the more that factor pushed the object's risk score upward."
    )

    left, right = st.columns([1.35, 1])

    with left:
        top_for_chart = breakdown[breakdown["Contribution"] > 0].copy()
        if top_for_chart.empty:
            st.info("No active risk factors are contributing to the current score.")
        else:
            top_for_chart = top_for_chart.sort_values("Contribution", ascending=True)
            fig = px.bar(
                top_for_chart,
                x="Contribution",
                y="Risk factor",
                orientation="h",
                text="Contribution",
                title="What is pushing the risk score up?",
                labels={"Contribution": "Risk points added (out of 100)", "Risk factor": ""},
            )
            fig.update_traces(texttemplate="%{text:.1f} pts", textposition="outside", cliponaxis=False)
            fig.update_xaxes(range=[0, max(22, float(top_for_chart["Contribution"].max()) * 1.28)])
            fig = style_plot(fig)
            fig.update_layout(showlegend=False, height=430)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        track_id = latest.get("track_id")
        track_text = "N/A" if pd.isna(track_id) else str(int(track_id))
        object_class = str(latest.get("object_class", "vehicle") or "vehicle")
        st.markdown(
            f"""
            <div class="risk-card">
                <h3>Decision summary</h3>
                <p><b>Tracked object:</b> ID {track_text} ({object_class})</p>
                <p><b>Calculated risk:</b> {risk:.1f}/100</p>
                <p><b>Classification:</b> {status}</p>
                <p><b>Intersection:</b> {latest.get('intersection', 'Unknown')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if status == "VIOLATION":
            st.error(
                f"The score is above the VIOLATION boundary ({warning_max:.0f}). "
                "The dashboard flags this as the highest-priority current object."
            )
        elif status == "WARNING":
            st.warning(
                f"The score is above the SAFE boundary ({safe_max:.0f}) but not above "
                f"the VIOLATION boundary ({warning_max:.0f})."
            )
        else:
            st.success(f"The score remains within the SAFE range (≤ {safe_max:.0f}).")

        st.markdown("#### Evidence recorded by the Risk Engine")
        if reasons:
            for reason in reasons:
                st.markdown(f"- **{reason}**")
        else:
            st.write("No high-risk rule reason was recorded for this event.")

    st.markdown("#### Risk contribution details")
    detail_table = breakdown.rename(columns={"Contribution": "Risk points"}).copy()
    st.dataframe(
        detail_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Risk points": st.column_config.ProgressColumn(
                "Risk points",
                help="Approximate points this factor contributes to the 0–100 prototype risk score.",
                format="%.1f",
                min_value=0.0,
                max_value=20.0,
            )
        },
    )

    # A simple threshold gauge makes the city's decision boundary immediately visible.
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "/100", "font": {"size": 46}},
            title={"text": "Current object risk vs decision thresholds", "font": {"size": 27}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickfont": {"size": 19}},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, safe_max], "color": "rgba(25,135,84,0.20)"},
                    {"range": [safe_max, warning_max], "color": "rgba(217,119,6,0.20)"},
                    {"range": [warning_max, 100], "color": "rgba(220,53,69,0.20)"},
                ],
                "threshold": {
                    "line": {"color": "#dc3545", "width": 4},
                    "thickness": 0.75,
                    "value": warning_max + 1,
                },
            },
        )
    )
    gauge.update_layout(height=330, font=dict(size=21), margin=dict(l=45, r=45, t=85, b=30))
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Prototype decision thresholds: SAFE 0–39, WARNING 40–69, VIOLATION 70–100. "
        "This score is a safety indicator from the project's feature-based Risk Engine; it is not a collision probability."
    )


def render_live(df: pd.DataFrame) -> None:
    latest = df.iloc[0]
    risk = safe_float(latest.get("risk_score"))
    status_banner(str(latest.get("status", "SAFE")), risk)

    # Two rows keep the metrics readable on a laptop or projector.
    row1 = st.columns(4)
    row1[0].metric("Risk score", f"{risk:.1f}/100")
    row1[1].metric("Vehicle speed", f'{safe_float(latest.get("vehicle_speed_kmh")):.1f} km/h')
    row1[2].metric("Distance to crosswalk", f'{safe_float(latest.get("distance_to_crosswalk_m")):.1f} m')
    row1[3].metric("Pedestrian separation", f'{safe_float(latest.get("pedestrian_vehicle_distance_m")):.1f} m')

    row2 = st.columns(4)
    track_id = latest.get("track_id")
    row2[0].metric("Highest-risk Track ID", "N/A" if pd.isna(track_id) else int(track_id))
    row2[1].metric("Time to collision", "N/A" if pd.isna(latest.get("ttc_seconds")) else f'{safe_float(latest.get("ttc_seconds")):.1f} s')
    row2[2].metric("Movement", str(latest.get("movement_direction", "UNKNOWN")).replace("_", " ").title())
    row2[3].metric("Predicted path conflict", "YES" if bool(safe_float(latest.get("trajectory_conflict"))) else "NO")

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
        breakdown, reasons = build_risk_breakdown(latest)
        top_factors = breakdown[breakdown["Contribution"] > 0].head(4)
        st.write(f"**Object:** {latest.get('object_class', '') or 'Vehicle'}")
        st.write(f"**Detection confidence:** {safe_float(latest.get('confidence')) * 100:.0f}%")
        st.write(f"**Road condition:** {latest.get('road_condition', 'Unknown')}")
        st.write(f"**Visibility:** {latest.get('visibility', 'Unknown')}")
        if not top_factors.empty:
            st.markdown("**Strongest score contributors:**")
            for _, factor in top_factors.iterrows():
                st.write(f"• {factor['Risk factor']}: +{factor['Contribution']:.1f} points")
        if reasons:
            st.markdown("**Risk Engine reasons:**")
            for reason in reasons[:5]:
                st.write(f"• {reason}")

    st.divider()
    render_city_risk_explanation(latest)


def render_analytics(df: pd.DataFrame) -> None:
    st.subheader("Traffic behavior analytics")
    st.caption("Use these charts to understand when and why risk is building across recorded events.")
    work = df.sort_values("timestamp").copy()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(work, x="timestamp", y="risk_score", color="status", title="How risk changes over time")
        fig.update_yaxes(range=[0, 100], title="Risk score (0–100)")
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        fig = px.scatter(
            work,
            x="distance_to_crosswalk_m",
            y="vehicle_speed_kmh",
            color="status",
            size="risk_score",
            hover_data=["track_id", "movement_direction", "ttc_seconds"],
            title="Speed compared with distance from crosswalk",
            labels={
                "distance_to_crosswalk_m": "Distance to crosswalk (m)",
                "vehicle_speed_kmh": "Vehicle speed (km/h)",
            },
        )
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        direction_counts = work["movement_direction"].fillna("UNKNOWN").value_counts().reset_index()
        direction_counts.columns = ["direction", "events"]
        fig = px.bar(direction_counts, x="direction", y="events", title="Which movement directions occur most often?")
        fig.update_xaxes(title="Movement direction")
        fig.update_yaxes(title="Number of events")
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c4:
        fig = px.histogram(
            work,
            x="pedestrian_vehicle_distance_m",
            color="status",
            nbins=20,
            title="How close were vehicles and pedestrians?",
            labels={"pedestrian_vehicle_distance_m": "Pedestrian–vehicle separation (m)"},
        )
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
        top_tracks = top_tracks.rename(
            columns={
                "track_id": "Track ID",
                "max_risk": "Highest risk",
                "max_speed": "Highest speed (km/h)",
                "min_crosswalk_distance": "Closest to crosswalk (m)",
                "min_separation": "Closest to pedestrian (m)",
                "samples": "Recorded samples",
            }
        )
        st.markdown("#### Highest-risk tracked objects")
        st.caption("Vehicles are ranked by the highest risk score recorded for their Track ID.")
        st.dataframe(top_tracks, hide_index=True, use_container_width=True)


def render_performance(df: pd.DataFrame) -> None:
    st.subheader("System performance")
    cols = st.columns(4)
    cols[0].metric("Average FPS", f'{df["fps"].mean():.1f}')
    cols[1].metric("Average latency", f'{df["response_time_ms"].mean():.1f} ms')
    cols[2].metric("P95 latency", f'{df["response_time_ms"].quantile(0.95):.1f} ms')
    cols[3].metric("Tracked IDs", int(df["track_id"].dropna().nunique()))

    p1, p2 = st.columns(2)
    with p1:
        fig = px.line(df.sort_values("timestamp"), x="timestamp", y="fps", title="Processing speed over time")
        fig.update_yaxes(title="Frames per second (FPS)")
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with p2:
        fig = px.line(df.sort_values("timestamp"), x="timestamp", y="response_time_ms", title="System response delay over time")
        fig.update_yaxes(title="Response time (ms)")
        fig = style_plot(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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


def main() -> None:
    st.title("GuideKaro — Enhanced AI Crosswalk Safety Dashboard")
    st.caption(
        "For city traffic authorities: live risk evidence • explainable risk scoring • tracking • analytics • reporting"
    )

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
        ["🚦 Live Monitor", "📊 Traffic Analytics", "⚙️ System Performance", "🧪 Failure Cases", "🗂️ Event Log", "📄 Reports"]
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
