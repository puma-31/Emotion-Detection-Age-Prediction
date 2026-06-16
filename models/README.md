Models folder

Place curated saved model artifacts here for a tidy repo layout.

Expected files:
- `best_model.keras`       — emotion classifier (Keras/TF saved model)
- `best_age_model.keras`   — age regression model

If these are large, you may prefer to keep them in a release asset or Git LFS. The inference scripts will prefer models in this folder and fall back to the `Emotion-Detection-Age-Prediction/` folder if not present.
