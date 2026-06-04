from flask import Flask, jsonify

app = Flask(__name__)

class MLService:
    def get_response(self):
        return {
            "status": "200 OK",
            "message": "ML Service is running successfully"
        }

service = MLService()

@app.route("/service", methods=["GET"])
def service():
    return jsonify(service.get_response())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050)
