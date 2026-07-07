import os
import pickle
import sqlite3
from datetime import datetime
from functools import wraps

import pandas as pd
from flask import Flask, redirect, render_template, request, session, url_for


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=template_dir)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "flood-risk-local-dev-key")

model_path = os.path.join(BASE_DIR, "flood_model.pkl")
metadata_path = os.path.join(BASE_DIR, "scaler.pkl")
database_path = os.path.join(BASE_DIR, "flood_history.db")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(metadata_path, "rb") as metadata_file:
    metadata = pickle.load(metadata_file)

LEGACY_FEATURES = ["annual_rainfall", "cloud_visibility", "monsoon_rainfall"]
NEW_FEATURES = [
    "annual_rainfall",
    "cloud_visibility",
    "monsoon_rainfall",
    "river_level",
    "soil_moisture",
    "drainage_score",
]
USING_NEW_MODEL = isinstance(metadata, dict) and "features" in metadata
FEATURES = metadata.get("features", NEW_FEATURES) if USING_NEW_MODEL else LEGACY_FEATURES


def get_db_connection():
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                annual_rainfall REAL NOT NULL,
                cloud_visibility REAL NOT NULL,
                monsoon_rainfall REAL NOT NULL,
                river_level REAL NOT NULL,
                soil_moisture REAL NOT NULL,
                drainage_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                probability REAL NOT NULL,
                result_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        columns = connection.execute("PRAGMA table_info(users)").fetchall()
        column_names = [column["name"] for column in columns]

        if "password_hash" in column_names:
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute(
                """
                CREATE TABLE users_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO users_new (id, username, created_at)
                SELECT id, username, created_at
                FROM users
                """
            )
            connection.execute("DROP TABLE users")
            connection.execute("ALTER TABLE users_new RENAME TO users")
            connection.execute("PRAGMA foreign_keys = ON")


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def save_prediction(form_values, risk_level, probability, result_text):
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO predictions (
                user_id,
                annual_rainfall,
                cloud_visibility,
                monsoon_rainfall,
                river_level,
                soil_moisture,
                drainage_score,
                risk_level,
                probability,
                result_text,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session["user_id"],
                form_values["annual_rainfall"],
                form_values["cloud_visibility"],
                form_values["monsoon_rainfall"],
                form_values["river_level"],
                form_values["soil_moisture"],
                form_values["drainage_score"],
                risk_level,
                round(probability, 2),
                result_text,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ),
        )


init_db()


@app.route("/login", methods=["GET", "POST"])
def login():
    message = None
    message_type = "warning"

    if request.method == "POST":
        username = request.form.get("username", "").strip()

        if not username:
            message = "Please enter your username."
        else:
            with get_db_connection() as connection:
                user = connection.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,),
                ).fetchone()

                if user is None:
                    connection.execute(
                        """
                        INSERT INTO users (username, created_at)
                        VALUES (?, ?)
                        """,
                        (
                            username,
                            datetime.now().strftime("%Y-%m-%d %H:%M"),
                        ),
                    )
                    user = connection.execute(
                        "SELECT * FROM users WHERE username = ?",
                        (username,),
                    ).fetchone()
                user = connection.execute(
                    "SELECT * FROM users WHERE username = ?",
                    (username,),
                ).fetchone()

            session.clear()
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))

    return render_template("login.html", message=message, message_type=message_type)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template(
        "index.html",
        accuracy=metadata.get("accuracy") if USING_NEW_MODEL else None,
        model_version="2.0.0" if USING_NEW_MODEL else "1.0.0",
        username=session.get("username"),
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        all_form_values = {feature: float(request.form[feature]) for feature in NEW_FEATURES}
        input_values = {feature: all_form_values[feature] for feature in FEATURES}
        input_df = pd.DataFrame([input_values], columns=FEATURES)

        model_input = input_df if USING_NEW_MODEL else metadata.transform(input_df)
        prediction = int(model.predict(model_input)[0])
        probability = float(model.predict_proba(model_input)[0][1]) * 100

        if probability >= 85:
            result_text = "CRITICAL RISK: Severe flood conditions are likely. Move valuables up, avoid low-lying roads, and follow official alerts."
            alert_class = "danger"
            risk_level = "Critical"
        elif probability >= 65:
            result_text = "HIGH RISK: Flooding is likely. Prepare safety supplies and keep monitoring local warnings."
            alert_class = "danger"
            risk_level = "High"
        elif probability >= 40 or prediction == 1:
            result_text = "MODERATE RISK: Conditions are elevated. Stay alert and keep drainage paths clear."
            alert_class = "warning"
            risk_level = "Moderate"
        else:
            result_text = "LOW RISK: Weather parameters are currently within safer limits."
            alert_class = "success"
            risk_level = "Low"

        save_prediction(all_form_values, risk_level, probability, result_text)

        return render_template(
            "index.html",
            prediction_text=result_text,
            alert_class=alert_class,
            risk_level=risk_level,
            probability=round(probability, 2),
            accuracy=metadata.get("accuracy") if USING_NEW_MODEL else None,
            roc_auc=metadata.get("roc_auc") if USING_NEW_MODEL else None,
            form_values=all_form_values,
            model_version="2.0.0" if USING_NEW_MODEL else "1.0.0",
            username=session.get("username"),
        )

    except Exception as error:
        return render_template(
            "index.html",
            prediction_text=f"System Error: {error}",
            alert_class="warning",
            accuracy=metadata.get("accuracy") if USING_NEW_MODEL else None,
            model_version="2.0.0" if USING_NEW_MODEL else "1.0.0",
            username=session.get("username"),
        )


@app.route("/history")
@login_required
def history():
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (session["user_id"],),
        ).fetchall()

    return render_template(
        "history.html",
        predictions=rows,
        username=session.get("username"),
    )


if __name__ == "__main__":
    app.run(debug=True, port=9000)
