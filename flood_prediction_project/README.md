# AI-Powered Flood Risk Analytics Portal

An end-to-end Machine Learning web application that predicts local flood risk from environmental readings and displays the result through a Flask web interface.

## Tech Stack

- Backend: Flask
- Machine Learning: scikit-learn, pandas, NumPy
- Models: tuned Random Forest, Gradient Boosting, soft-voting ensemble
- Frontend: Bootstrap, FontAwesome, HTML, CSS

## Improved V2 Prediction Signals

- Annual rainfall
- Cloud visibility
- Monsoon rainfall
- River level
- Soil moisture
- Drainage risk score

## Accuracy Improvements Added

- Larger synthetic training dataset with more flood-related variables
- Stratified train/test split
- Standardized preprocessing inside scikit-learn pipelines
- GridSearchCV hyperparameter tuning
- Soft-voting ensemble model
- Saved model accuracy and ROC-AUC metadata
- Probability-based risk levels: Low, Moderate, High, Critical

## Login and Prediction History

- Users enter only a username from the login page.
- Each prediction is saved under the logged-in user.
- Saved predictions can be viewed from the History page.
- History is stored locally in `flood_history.db`, which is created automatically when the app starts.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Retrain the improved model:

```bash
python train_model.py
```

Start the Flask app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:9000
```

## Important

This project uses generated sample data for demonstration. For real flood alerts, always follow official weather and disaster-management warnings.
# rising-waters
