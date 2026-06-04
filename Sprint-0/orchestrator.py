import requests
import subprocess
import time

# Start the API server
print("Starting ML Service...")

# NOTE: In real notebook this runs separately
# This is simulation-friendly version

time.sleep(2)

# Send request to API
response = requests.get("http://127.0.0.1:5050/service")

print("Status Code:", response.status_code)
print("Response:", response.json())
