import pickle
from sklearn.ensemble import IsolationForest
import numpy as np

# เทรนครั้งเดียว
normal = np.array([[4.8],[5.0],[5.1],[4.9],[5.0],
                   [5.2],[4.7],[5.0],[4.9],[5.1]])
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(normal)

# เซฟลงไฟล์
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ บันทึก model แล้ว!")