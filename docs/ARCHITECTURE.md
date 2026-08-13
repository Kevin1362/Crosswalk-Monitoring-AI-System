
# GuideKaro Full Project Architecture

The PNG in this folder is the presentation-ready architecture diagram.

## Logical architecture

```mermaid
flowchart LR
    A[Camera / Video / Weather / Signal] --> B[OpenCV Frame Capture]
    B --> C[YOLOv8 Detection]
    C --> D[ByteTrack Tracking]
    D --> E[Speed + Distance + Direction]
    E --> F[Trajectory Prediction]
    F --> G[Feature-Based Risk Engine]
    G --> H{SAFE / WARNING / VIOLATION}
    H --> I[Alert Engine]
    H --> J[SQLite Event Database]
    J --> K[Streamlit Dashboard]
    J --> L[Analytics / Reports]
    I --> K

    subgraph Deployment
      M[Docker Container Platform]
      N[Application Logs + Container Metrics]
      O[Grafana Cloud Monitoring]
    end

    C --> M
    K --> M
    J --> M
    M --> N --> O
```

## Key change from the first prototype

The first prototype mainly used object counts. The enhanced version uses persistent track IDs and features from object motion:

- Track ID
- Detection confidence
- Speed
- Distance to crosswalk
- Pedestrian–vehicle separation
- Movement direction
- Predicted trajectory
- Crosswalk occupancy
- Approximate time-to-collision
- Weather / visibility flags

This makes the risk decision traceable and feature based.
