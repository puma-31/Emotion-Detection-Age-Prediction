# Emotion Detection and Age Prediction

Real-time facial emotion classification and age estimation from webcam or image input using TensorFlow/Keras and OpenCV.

## Demo Placeholders
- Add a short GIF: `assets/screenshots/demo.gif`
- Add 2-4 screenshots: `assets/screenshots/demo1.png`, `assets/screenshots/demo2.png`
- Optional sample output image: `assets/screenshots/test_image_annotated.jpg`

## Features
- Real-time webcam inference with face detection overlays.
- Single-image inference with annotated output preview.
- Optional save path for annotated image output.
- Separate training notebooks and inference scripts.

## Repository Structure
```text
Emotion-Detection-Age-Prediction/
	assets/
		screenshots/
		test_image.jpg
	demos/
		README.md
	docs/
		RESULTS.md
	models/
		best_model.keras
		best_age_model.keras
		README.md
	notebooks/
		Age_Prediction.ipynb
		Emotion_Detection.ipynb
	src/
		inference.py
	detector.py
	requirements.txt
	.gitignore
	README.md
```

## Setup
From the repository root (`Emotion-Detection-Age-Prediction`):

```powershell
python -m venv tf_env
.\tf_env\Scripts\activate
pip install -r requirements.txt
```

## Run Inference
Webcam:

```powershell
python src/inference.py --webcam
```

Single image (show annotated output window):

```powershell
python src/inference.py --image assets/test_image.jpg
```

Single image (save annotated output):

```powershell
python src/inference.py --image assets/test_image.jpg --save assets/screenshots/test_image_annotated.jpg
```

Single image (save only, no GUI window):

```powershell
python src/inference.py --image assets/test_image.jpg --save assets/screenshots/test_image_annotated.jpg --no-show
```

## Models
- Emotion model: `models/best_model.keras`
- Age model: `models/best_age_model.keras`

`src/inference.py` loads models from `models/` by default.

## Results
See `docs/RESULTS.md` for current metrics and planned plots.
