import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "annual_rainfall",
    "cloud_visibility",
    "monsoon_rainfall",
    "river_level",
    "soil_moisture",
    "drainage_score",
]


def generate_data(n_samples=6000, random_state=42):
    rng = np.random.default_rng(random_state)

    annual_rainfall = rng.uniform(500, 3800, n_samples)
    monsoon_ratio = rng.uniform(0.45, 0.86, n_samples)
    monsoon_rainfall = annual_rainfall * monsoon_ratio
    cloud_visibility = rng.uniform(0, 10, n_samples)
    river_level = np.clip(
        18 + monsoon_rainfall / 42 + rng.normal(0, 10, n_samples),
        0,
        100,
    )
    soil_moisture = np.clip(
        20 + monsoon_rainfall / 55 + annual_rainfall / 110 + rng.normal(0, 12, n_samples),
        0,
        100,
    )
    drainage_score = rng.uniform(0, 100, n_samples)

    risk_signal = (
        0.0010 * annual_rainfall
        + 0.0018 * monsoon_rainfall
        + 0.038 * river_level
        + 0.025 * soil_moisture
        + 0.020 * drainage_score
        - 0.18 * cloud_visibility
        + 0.000006 * monsoon_rainfall * river_level
        + rng.normal(0, 0.55, n_samples)
    )

    flood = (risk_signal > np.percentile(risk_signal, 68)).astype(int)

    return pd.DataFrame(
        {
            "annual_rainfall": annual_rainfall,
            "cloud_visibility": cloud_visibility,
            "monsoon_rainfall": monsoon_rainfall,
            "river_level": river_level,
            "soil_moisture": soil_moisture,
            "drainage_score": drainage_score,
            "flood": flood,
        }
    )


def train():
    df = generate_data()
    df.to_csv("historical_weather.csv", index=False)
    print("Created improved historical_weather.csv")

    x = df[FEATURES]
    y = df["flood"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    rf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    search = GridSearchCV(
        rf_pipeline,
        {
            "model__n_estimators": [220, 320],
            "model__max_depth": [10, 14, None],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)

    tuned_rf = search.best_estimator_
    gradient_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42,
                    n_estimators=180,
                    learning_rate=0.06,
                    max_depth=3,
                ),
            ),
        ]
    )

    ensemble = VotingClassifier(
        estimators=[
            ("random_forest", tuned_rf),
            ("gradient_boosting", gradient_model),
        ],
        voting="soft",
        weights=[2, 1],
    )
    ensemble.fit(x_train, y_train)

    predictions = ensemble.predict(x_test)
    probabilities = ensemble.predict_proba(x_test)[:, 1]
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    metadata = {
        "features": FEATURES,
        "accuracy": round(accuracy * 100, 2),
        "roc_auc": round(roc_auc * 100, 2),
        "best_random_forest_params": search.best_params_,
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    with open("flood_model.pkl", "wb") as model_file:
        pickle.dump(ensemble, model_file)

    with open("scaler.pkl", "wb") as metadata_file:
        pickle.dump(metadata, metadata_file)

    print(f"Accuracy: {metadata['accuracy']}%")
    print(f"ROC-AUC: {metadata['roc_auc']}%")
    print(f"Best random forest params: {metadata['best_random_forest_params']}")
    print("Saved flood_model.pkl and scaler.pkl")


if __name__ == "__main__":
    train()
