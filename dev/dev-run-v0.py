from ml_service import app

if __name__ == "__main__":
    print("Starting Crosswalk Monitoring Service (v0)...")
    app.run(host="0.0.0.0", port=8085, debug=True)
