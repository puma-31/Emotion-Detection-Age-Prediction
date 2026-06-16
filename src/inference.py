import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMG_SIZE = (48, 48)


def _resolve_model_path(preferred_path: str, fallback_path: str) -> str:
    p = Path(preferred_path)
    if p.exists():
        return str(p)
    fb = Path(fallback_path)
    if fb.exists():
        return str(fb)
    return str(p)


def load_models():
    repo_root = Path(__file__).resolve().parents[1]
    preferred_emotion = repo_root / 'models' / 'best_model.keras'
    fallback_emotion = repo_root / 'Emotion-Detection-Age-Prediction' / 'best_model.keras'

    preferred_age = repo_root / 'models' / 'best_age_model.keras'
    fallback_age = repo_root / 'Emotion-Detection-Age-Prediction' / 'best_age_model.keras'

    em_path = _resolve_model_path(preferred_emotion, fallback_emotion)
    ag_path = _resolve_model_path(preferred_age, fallback_age)

    emotion_model = tf.keras.models.load_model(em_path)
    age_model = tf.keras.models.load_model(ag_path)
    return emotion_model, age_model


def preprocess_face(face_bgr):
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized[..., np.newaxis], axis=0)


def predict_from_image(image_path, emotion_model, age_model):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_path}")
    # naive face detection using OpenCV cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        inp = preprocess_face(face)
        probs = emotion_model.predict(inp, verbose=0)[0]
        emotion = EMOTION_LABELS[int(np.argmax(probs))]
        age = int(age_model.predict(inp, verbose=0)[0][0])
        results.append({'box': (int(x), int(y), int(w), int(h)), 'emotion': emotion, 'probs': probs.tolist(), 'age': age})
    return results, img


def draw_predictions(image, results):
    annotated = image.copy()
    for result in results:
        x, y, w, h = result['box']
        label = f"{result['emotion']}, {result['age']}"
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label_y = y - 10 if y - 10 > 10 else y + h + 20
        cv2.putText(annotated, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return annotated


def run_webcam(emotion_model, age_model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            inp = preprocess_face(face)
            probs = emotion_model.predict(inp, verbose=0)[0]
            emotion = EMOTION_LABELS[int(np.argmax(probs))]
            age = int(age_model.predict(inp, verbose=0)[0][0])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Path to an image file to run inference on')
    parser.add_argument('--webcam', action='store_true', help='Run webcam real-time inference')
    parser.add_argument('--save', type=str, help='Optional path to save annotated image output')
    parser.add_argument('--no-show', action='store_true', help='Do not open image preview window')
    args = parser.parse_args()

    emotion_model, age_model = load_models()

    if args.image:
        res, img = predict_from_image(args.image, emotion_model, age_model)
        annotated = draw_predictions(img, res)
        print(res)

        if args.save:
            cv2.imwrite(args.save, annotated)
            print(f"Saved annotated image to {args.save}")

        if not args.no_show:
            cv2.imshow('Image Inference', annotated)
            print("Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.webcam:
        run_webcam(emotion_model, age_model)
    else:
        print('Specify --image IMAGE or --webcam')


if __name__ == '__main__':
    main()
