from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/service", methods=["GET"])
def service():
    return jsonify({"status": "200 OK"}), 200

if __name__ == "__main__":
    app.run(port=8085)
