# AI-Based Crosswalk Monitoring and Driver Alert System

## Project Overview

The AI-Based Crosswalk Monitoring and Driver Alert System is an intelligent traffic safety solution designed to improve pedestrian safety at crosswalks through real-time monitoring, risk assessment, and automated alerts.

The system combines computer vision, machine learning, and decision-making logic to detect pedestrians and vehicles, evaluate potential risks, and generate appropriate warnings before dangerous situations occur.

The project follows a DevOps and MLOps development approach, enabling continuous development, testing, monitoring, and improvement throughout the project lifecycle.

---

## Project Objectives

The main objectives of the system are to:

- Detect pedestrians, vehicles, and crosswalk zones in real time  
- Assess risk levels based on vehicle speed, distance, and pedestrian activity  
- Generate early warnings for potentially unsafe situations  
- Support safe vehicle decision-making near crosswalks  
- Track and record traffic violations for analysis and reporting  
- Provide monitoring and logging capabilities for system evaluation  

---

## System Use Cases

### UC-1: Early Warning System

The system continuously monitors approaching vehicles and pedestrian activity around crosswalks.

When a potentially dangerous situation is detected, an early warning alert is generated to help prevent accidents before they occur.

---

### UC-2: Safe Decision Support System

Based on pedestrian presence, vehicle position, and risk levels, the system determines the safest action:

- Proceed  
- Slow Down  
- Stop  
- Hold Position  

This decision logic helps improve safety at pedestrian crossings.

---

### UC-3: Violation Detection and Tracking

The system identifies unsafe driving behavior, including:

- Failure to yield to pedestrians  
- Entering occupied crosswalks  
- Ignoring crosswalk safety rules  

All detected violations are recorded for future analysis and reporting.

---

## System Architecture

The solution follows a layered architecture consisting of three main components:

### Detection Layer
Responsible for:
- Vehicle detection  
- Pedestrian detection  
- Crosswalk identification  

### Intelligence Layer
Responsible for:
- Risk assessment  
- Distance estimation  
- Speed estimation  
- Decision-making logic  

### Monitoring Layer
Responsible for:
- Alert generation  
- Event logging  
- Performance monitoring  
- Dashboard visualization  

---

## Project Structure
Crosswalk-Monitoring-AI-System/
├── README.md
├── orchestrator.ipynb
├── data-collection/
├── training/
│ └── trained-model-v0.h5
├── dev/
│ └── dev-run-v0.py
└── documentation/

---

## Technologies Used

### Development
- Python  
- OpenCV  
- YOLO  

### DevOps & MLOps
- Azure DevOps  
- GitHub  
- Git Version Control  

### Data Storage
- SQLite  
- CSV Files  

### Monitoring & Reporting
- Logging Frameworks  
- Performance Monitoring  
- Dashboard Visualization  

---

## Sprint Roadmap

### Sprint 0 – Project Setup & Planning
- Azure DevOps configuration  
- GitHub repository setup  
- Literature review  
- Dataset selection  
- MLOps architecture planning  

### Sprint 1 – Detection System Development
- Vehicle detection  
- Pedestrian detection  
- Crosswalk identification  

### Sprint 2 – Risk Analysis Engine
- Speed estimation  
- Distance estimation  
- Risk scoring  

### Sprint 3 – Decision Support & Alerts
- Decision engine  
- Alert generation  
- Visual warning overlays  

### Sprint 4 – DevOps Monitoring & Security
- Centralized monitoring and logging  
- System health monitoring  
- Security and data integrity validation  
- Real-time event tracking  
- Documentation and reporting  

---

## Current Project Status

### Completed
- Repository setup  
- Sprint planning  
- Azure DevOps project structure  
- MLOps architecture design  
- Use case definition  

### In Progress
- Detection module development  
- Risk analysis implementation  
- Decision engine development  

### Planned
- Monitoring dashboard  
- Security validation layer  
- Performance monitoring  
- Final system testing  

---

## Expected Outcomes

The completed system will provide:

- Real-time pedestrian and vehicle monitoring  
- Intelligent risk assessment  
- Automated safety alerts  
- Traffic violation tracking  
- Monitoring and reporting capabilities  

The overall goal is to improve pedestrian safety and support smarter traffic monitoring using AI-driven technologies.

---
## Assignment Evidence
Azure DevOps backlog and GitHub integration evidence added for Assignment 2.
## License

This project is licensed under the MIT License.
