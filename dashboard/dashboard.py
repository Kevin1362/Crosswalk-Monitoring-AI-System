import streamlit as st
import pandas as pd
import sqlite3
import time
import os
from datetime import datetime


# ==============================
# Database Configuration
# ==============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_NAME = os.path.join(
    BASE_DIR,
    "crosswalk.db"
)



# ==============================
# Load Detection Data
# ==============================

def load_data():

    if not os.path.exists(DB_NAME):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_NAME)

    query = """
    SELECT *
    FROM detections
    ORDER BY id DESC
    LIMIT 100
    """

    try:
        data = pd.read_sql_query(
            query,
            conn
        )

    except Exception as e:
        st.error(
            f"Database error: {e}"
        )
        data = pd.DataFrame()

    finally:
        conn.close()

    return data



# ==============================
# Streamlit Page Setup
# ==============================

st.set_page_config(
    page_title="Intelligent Crosswalk Monitoring System",
    page_icon="🚦",
    layout="wide"
)



st.title(
    "🚦 Intelligent Crosswalk Monitoring System"
)


st.markdown(
    """
    ### YOLOv5 AI Object Detection Dashboard

    Monitoring:

    - 🚶 Pedestrians
    - 🚗 Vehicles
    - ⚠️ Safety warnings
    - 📊 Detection confidence
    """
)



# ==============================
# Load Database
# ==============================

data = load_data()



if not data.empty:


    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )


    # ==============================
    # Statistics
    # ==============================

    total = len(data)


    pedestrians = len(
        data[
            data["object_type"] == "person"
        ]
    )


    vehicles = len(
        data[
            data["object_type"].isin(
                [
                    "car",
                    "bus",
                    "truck",
                    "motorcycle"
                ]
            )
        ]
    )


    warnings = len(
        data[
            data["status"] == "WARNING"
        ]
    )


    avg_confidence = round(
        data["confidence"].mean() * 100,
        2
    )



    # ==============================
    # Dashboard Metrics
    # ==============================

    col1, col2, col3, col4, col5 = st.columns(5)


    col1.metric(
        "Total Detections",
        total
    )


    col2.metric(
        "Pedestrians",
        pedestrians
    )


    col3.metric(
        "Vehicles",
        vehicles
    )


    col4.metric(
        "Warnings",
        warnings
    )


    col5.metric(
        "Confidence",
        f"{avg_confidence}%"
    )



    st.divider()



    # ==============================
    # Detection Table
    # ==============================

    st.subheader(
        "📋 Latest Detection Events"
    )


    st.dataframe(
        data,
        width="stretch"
    )



    st.divider()



    # ==============================
    # Charts
    # ==============================

    colA, colB = st.columns(2)



    with colA:

        st.subheader(
            "Object Detection Count"
        )


        object_chart = (
            data["object_type"]
            .value_counts()
        )


        st.bar_chart(
            object_chart
        )



    with colB:

        st.subheader(
            "Detection Status"
        )


        status_chart = (
            data["status"]
            .value_counts()
        )


        st.bar_chart(
            status_chart
        )



    st.divider()



    # ==============================
    # Confidence Analysis
    # ==============================

    st.subheader(
        "📈 Confidence Trend"
    )


    confidence_chart = data[
        [
            "timestamp",
            "confidence"
        ]
    ].set_index(
        "timestamp"
    )


    st.line_chart(
        confidence_chart
    )



    st.success(
        f"Database: {DB_NAME}\n\n"
        f"Last update: {datetime.now().strftime('%H:%M:%S')}"
    )



else:

    st.warning(
        "Waiting for YOLO detections..."
    )



# ==============================
# Auto Refresh
# ==============================

time.sleep(3)

st.rerun()