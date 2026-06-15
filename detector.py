import cv2
import numpy as np
import tensorflow as tf

EMOTION_MODEL_PATH = "best_model.keras"
AGE_MODEL_PATH = "C:\\Users\\mpura\\projects\\emotion_detection_and_age_prediction\\Emotion-Detection-Age-Prediction\\best_age_model.keras"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
age_model = tf.keras.models.load_model(AGE_MODEL_PATH)

def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)          
    resized = cv2.resize(gray, IMG_SIZE)                        
    normalized = resized.astype("float32") / 255.0            
    return np.expand_dims(normalized[..., np.newaxis], axis=0)                      

cap = cv2.VideoCapture(0)   # 0 = default webcam; change to 1 for external

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        pad = int(0.1 * w)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
        face_crop = frame[y1:y2, x1:x2]

        input_tensor = preprocess_face(face_crop)
        preds = emotion_model.predict(input_tensor, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = EMOTION_LABELS[emotion_idx]
        confidence = preds[emotion_idx] * 100

        age_pred = age_model.predict(input_tensor, verbose=0)[0][0]
        age = int(age_pred)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{emotion}, {age}"# ({confidence:.1f}%)"
        label_y = y1 - 10 if y1 - 10 > 10 else y2 + 20
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for i, (emo, prob) in enumerate(zip(EMOTION_LABELS, preds)):
            bar_x = 10
            bar_y = 20 + i * 22
            bar_len = int(prob * 150)
            color = (0, 255, 0) if i == emotion_idx else (100, 100, 100)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 14), color, -1)
            cv2.putText(frame, f"{emo[:3]} {prob:.2f}", (bar_x + 155, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()