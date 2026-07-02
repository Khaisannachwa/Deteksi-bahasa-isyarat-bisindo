from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
import time
import os
import threading

app = Flask(__name__)

# =====================================================
# SYSTEM CONFIGURATION & AUTO-DEBUG
# =====================================================
MODEL_KATA_PATH = "models/model_kata.h5"
LABELS_PATH = "output/labels.npy"

SEQUENCE_LENGTH = 30  
EXPECTED_FEATURES = 126 

print("\n" + "="*50)
print("=== SYSTEM DIAGNOSTIC START ===")
print("="*50)

if os.path.exists(LABELS_PATH):
    labels_kata = np.load(LABELS_PATH)
    print(f"[OK] Label berhasil dimuat. Total kata: {len(labels_kata)}")
else:
    labels_kata = np.array(["Error: labels.npy tidak ditemukan"])
    print(f"[WARNING] File {LABELS_PATH} tidak ditemukan!")

if os.path.exists(MODEL_KATA_PATH):
    try:
        model_kata = load_model(MODEL_KATA_PATH)
        input_shape = model_kata.input_shape
        output_shape = model_kata.output_shape[-1]
        
        SEQUENCE_LENGTH = input_shape[1] if input_shape[1] is not None else 30
        EXPECTED_FEATURES = input_shape[2] if input_shape[2] is not None else 126
        
        print(f"[OK] Model '{MODEL_KATA_PATH}' berhasil dimuat.")
        if len(labels_kata) != output_shape:
            print(f"[CRITICAL] MISMATCH! Label ({len(labels_kata)}) != Output Model ({output_shape})!")
    except Exception as e:
        print(f"[ERROR] Gagal membaca struktur model: {e}")
else:
    print(f"[CRITICAL] File model '{MODEL_KATA_PATH}' tidak ditemukan!")

print("="*50 + "\n")

model_angka = load_model("models/model_angka.h5") if os.path.exists("models/model_angka.h5") else None
model_huruf = load_model("models/model_huruf.h5") if os.path.exists("models/model_huruf.h5") else None

classes_angka = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
classes_huruf = ['A', 'B', 'C', 'D', 'E', 'F']

history = []
latest = {"label": "-", "confidence": 0}

# Lock untuk sinkronisasi thread saat mengakses kamera fisik yang sama
camera_lock = threading.Lock()

def save_history(label, confidence):
    global history
    if label == "-": return
    if len(history) > 0 and history[-1]["hasil"] == label: return

    latest["label"] = label
    latest["confidence"] = round(confidence, 2)
    history.append({
        "waktu": time.strftime("%H:%M:%S"),
        "hasil": label,
        "confidence": round(confidence, 2)
    })
    if len(history) > 20: history.pop(0)

# =====================================================
# MEDIAPIPE INITIALIZATION
# =====================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

def extract_keypoints(results):
    left = np.zeros(63)
    right = np.zeros(63)
    
    if results.multi_hand_landmarks:
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            data = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()
            side = handed.classification[0].label
            if side == "Left":
                left = data
            else:
                right = data
                
    full_hands = np.concatenate([left, right])
    if EXPECTED_FEATURES == 63:
        return right if np.any(right) else left
    return full_hands

# =====================================================
# GENERATOR STREAMS (Safe Camera Lifecycle)
# =====================================================
def generate_kata():
    # Menggunakan Lock agar thread lain tidak membuka kamera bersamaan
    with camera_lock:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        sequence = []
        current_label = "-"
        current_confidence = 0.0
        frame_counter = 0 

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame_counter += 1

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-SEQUENCE_LENGTH:]

                if len(sequence) == SEQUENCE_LENGTH and frame_counter % 3 == 0:
                    input_data = np.expand_dims(sequence, axis=0)
                    prediction = model_kata.predict(input_data, verbose=0)[0]
                    idx = np.argmax(prediction)
                    confidence = prediction[idx]

                    if confidence > 0.85:
                        current_label = labels_kata[idx]
                        current_confidence = confidence
                        save_history(current_label, confidence * 100)
                    else:
                        current_label = "-"
                        current_confidence = confidence
            else:
                sequence.clear()
                current_label = "-"
                current_confidence = 0.0

            cv2.putText(frame, f"Kata : {current_label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Conf : {current_confidence*100:.2f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01) # Mencegah CPU usage loncat mendadak
            
        cap.release()

def generate_angka():
    with camera_lock:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 100:400]
            img = cv2.resize(roi, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)

            if model_angka:
                pred = model_angka.predict(img, verbose=0)[0]
                idx, conf = np.argmax(pred), np.max(pred)
                label = classes_angka[idx] if conf > 0.60 else "-"
                if label != "-": save_history(label, conf * 100)
            else:
                label = "Model Belum Dimuat"

            cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
            cv2.putText(frame, f"Angka: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01)
        cap.release()

def generate_huruf():
    with camera_lock:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        avg = deque(maxlen=10)
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            roi = frame[100:400, 100:400]
            img = cv2.resize(roi, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)

            if model_huruf:
                pred = model_huruf.predict(img, verbose=0)[0]
                avg.append(pred)
                pred_avg = np.mean(avg, axis=0)
                idx, conf = np.argmax(pred_avg), np.max(pred_avg)
                label = classes_huruf[idx] if conf > 0.70 else "-"
                if label != "-": save_history(label, conf * 100)
            else:
                label = "Model Belum Dimuat"

            cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
            cv2.putText(frame, f"Huruf: {label}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            _, buffer = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.01)
        cap.release()

# =====================================================
# FLASK ROUTING
# =====================================================
@app.route("/")
def index(): return render_template("index.html")

@app.route("/angka")
def angka(): return render_template("angka.html")

@app.route("/huruf")
def huruf(): return render_template("huruf.html")

@app.route("/kata")
def kata(): return render_template("kata.html")

@app.route("/video_angka")
def video_angka(): return Response(generate_angka(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_huruf")
def video_huruf(): return Response(generate_huruf(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_kata")
def video_kata(): return Response(generate_kata(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/latest")
def get_latest(): return jsonify(latest)

@app.route("/history")
def get_history(): return jsonify(history[::-1])

if __name__ == "__main__":
    app.run(debug=True, threaded=True)