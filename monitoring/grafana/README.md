
# Grafana Cloud Monitoring

GuideKaro is designed so Docker and application logs can be monitored without changing the AI logic.

## What to monitor

### Container health
- container running / stopped
- CPU
- memory
- network
- restart count

### Application
- processing FPS
- frame latency
- risk events
- exceptions
- dashboard availability

### Log sources
- Docker container stdout / stderr
- `logs/guidekaro.log`

## Recommended Grafana panels

1. GuideKaro container health
2. CPU and memory usage
3. Application error count
4. Average latency
5. Average FPS
6. High-risk event count
7. Dashboard availability
8. Docker restarts

## Docker Desktop / Grafana Cloud

If you use the Docker Desktop Grafana integration, connect the running `guidekaro-dashboard` container and select its container metrics / logs in Grafana Cloud.

Do not claim Prometheus or Loki as part of the project unless you actually configured them. The architecture labels this generally as **Docker/Application Logs & Container Metrics → Grafana Cloud Monitoring**.
