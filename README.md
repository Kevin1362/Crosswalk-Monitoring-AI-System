Crosswalk Monitoring AI System
 Project Overview

This project is an AI-based traffic safety system designed to monitor crosswalk areas and detect unsafe situations involving vehicles and pedestrians. The main goal is to improve road safety by identifying risks early and supporting better driver decisions using real-time analysis.

 System Purpose

The system focuses on improving pedestrian safety by:

Detecting vehicles and pedestrians near crosswalks
Understanding whether a situation is safe or risky
Warning when a vehicle gets too close to pedestrians
Recording unsafe driving behavior (violations)
Displaying visual alerts in real time
Generating simple traffic analysis reports
🧠 Main System Features (3 Use Cases)
1. Early Warning System

This feature detects when a vehicle is approaching a crosswalk too fast or too close to pedestrians.
If a potential danger is identified, the system generates an early warning before an accident can occur.

2. Safe Decision System

This feature determines the safest action for a vehicle near a crosswalk:

Stop
Wait (Hold position)
Move safely

This decision is based on real-time pedestrian presence and vehicle movement.

3. Violation Tracking System

This feature records unsafe driving behavior such as:

Entering a crosswalk while pedestrians are crossing
Ignoring crosswalk rules

All events are stored for reporting and analysis.
**
 Project Structure**
/datasets → Sample videos and data
/src/detection → Vehicle and pedestrian detection module
/src/risk_analysis → Risk evaluation and scoring
/src/decision_engine → Safe-state decision logic
/src/tracking → Movement tracking system
/logs → Stored violation data
/reports → Generated analytics and summaries
** Technologies Used**
Computer Vision (video analysis)
Object Detection (vehicles & pedestrians)
Risk Scoring
Decision-Making Logic
Event Logging & Reporting
 Current Progress
✔ GitHub repository setup completed
✔ Project structure created
🔄 Detection and risk modules in progress
⏳ Dashboard integration and testing pending
**Project Goal**

To develop an intelligent traffic monitoring system that improves pedestrian safety at crosswalks using AI-based detection, risk analysis, and decision-making
