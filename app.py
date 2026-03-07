from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ โหลด model สำเร็จ!")

latest_data = {"status": "safe", "message": "รอข้อมูลจาก ESP32...", "current": 0, "score": 0}

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "AmpSafe API Running ✅"})

@app.route('/predict', methods=['POST'])
def predict():
    global latest_data
    data = request.json
    current_value = data['current']
    device_name = data.get('name', 'Machine 1')

    result = model.predict([[current_value]])
    score = model.decision_function([[current_value]])[0]

    if result[0] == 1:
        status = "safe"
        message = "safe"
    elif score >= -0.1:
        status = "warning"
        message = "warning"
    else:
        status = "dangerous"
        message = "dangerous"

    latest_data = {
        "name": device_name,
        "status": status,
        "message": message,
        "current": current_value,
        "score": round(float(score), 4)
    }

    return jsonify(latest_data)

@app.route('/latest', methods=['GET'])
def latest():
    return jsonify(latest_data)

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))