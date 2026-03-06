from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ให้เว็บเพื่อนเรียกได้

# โหลด model ตอนเปิด server
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("✅ โหลด model สำเร็จ!")

# ทดสอบว่า server ทำงานอยู่
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "AmpSafe API Running ✅"})

# รับค่ากระแสแล้วทำนาย
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    current_value = data['current']  # รับค่ากระแสจากเว็บ

    result = model.predict([[current_value]])
    score = model.decision_function([[current_value]])[0]

    if result[0] == 1:
        status = "normal"
        message = "✅ ปกติ"
    else:
        status = "anomaly"
        message = "⚠️ ผิดปกติ! ควรตรวจสอบเครื่องจักร"

    return jsonify({
        "status": status,
        "message": message,
        "current": current_value,
        "score": round(float(score), 4)
    })

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
