from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# ── Load trained model ─────────────────────────────────────────────────────────
with open("injury_model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Feature definitions with validation ranges ─────────────────────────────────
FEATURES = [
    {"name": "Age",                "label": "Age",                  "min": 18,   "max": 40,   "step": 1,    "type": "int"},
    {"name": "Gender",             "label": "Gender (0=F, 1=M)",    "min": 0,    "max": 1,    "step": 1,    "type": "int"},
    {"name": "Height_cm",          "label": "Height (cm)",          "min": 150,  "max": 190,  "step": 0.1,  "type": "float"},
    {"name": "Weight_kg",          "label": "Weight (kg)",          "min": 45,   "max": 95,   "step": 0.1,  "type": "float"},
    {"name": "BMI",                "label": "BMI",                  "min": 12,   "max": 34,   "step": 0.01, "type": "float"},
    {"name": "Training_Frequency", "label": "Training Frequency (days/week)", "min": 1, "max": 6, "step": 1, "type": "int"},
    {"name": "Training_Duration",  "label": "Training Duration (min)",       "min": 45, "max": 120, "step": 1, "type": "int"},
    {"name": "Warmup_Time",        "label": "Warmup Time (min)",    "min": 0,    "max": 20,   "step": 1,    "type": "int"},
    {"name": "Sleep_Hours",        "label": "Sleep Hours",          "min": 5,    "max": 9.5,  "step": 0.5,  "type": "float"},
    {"name": "Flexibility_Score",  "label": "Flexibility Score",    "min": 30,   "max": 95,   "step": 0.1,  "type": "float"},
    {"name": "Muscle_Asymmetry",   "label": "Muscle Asymmetry",     "min": 0,    "max": 14.5, "step": 0.1,  "type": "float"},
    {"name": "Recovery_Time",      "label": "Recovery Time (hrs)",  "min": 30,   "max": 119,  "step": 1,    "type": "int"},
    {"name": "Injury_History",     "label": "Injury History (count)","min": 0,   "max": 3,    "step": 1,    "type": "int"},
    {"name": "Stress_Level",       "label": "Stress Level (1–10)",  "min": 1,    "max": 10,   "step": 1,    "type": "int"},
    {"name": "Training_Intensity", "label": "Training Intensity (1–10)", "min": 1, "max": 10, "step": 0.1, "type": "float"},
]


def validate_and_parse(form_data):
    """Parse and validate all form inputs. Returns (features_array, errors)."""
    values = []
    errors = []

    for feat in FEATURES:
        raw = form_data.get(feat["name"], "").strip()
        if raw == "":
            errors.append(f"{feat['label']} is required.")
            continue
        try:
            val = int(raw) if feat["type"] == "int" else float(raw)
        except ValueError:
            errors.append(f"{feat['label']} must be a valid number.")
            continue

        if val < feat["min"] or val > feat["max"]:
            errors.append(
                f"{feat['label']} must be between {feat['min']} and {feat['max']}."
            )
            continue

        values.append(val)

    return (np.array([values]) if not errors else None), errors


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES, result=None, errors=[])


@app.route("/predict", methods=["POST"])
def predict():
    final_features, errors = validate_and_parse(request.form)

    if errors:
        return render_template(
            "index.html",
            features=FEATURES,
            result=None,
            errors=errors,
            form_values=request.form,
        )

    # Prediction
    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0]

    high_risk_pct = round(probability[1] * 100, 1)
    low_risk_pct  = round(probability[0] * 100, 1)

    result = {
        "label":         "HIGH RISK" if prediction == 1 else "LOW RISK",
        "is_high_risk":  prediction == 1,
        "high_risk_pct": high_risk_pct,
        "low_risk_pct":  low_risk_pct,
    }

    return render_template(
        "index.html",
        features=FEATURES,
        result=result,
        errors=[],
        form_values=request.form,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for programmatic access."""
    try:
        data = request.get_json(force=True)
        values = [data[f["name"]] for f in FEATURES]
        arr = np.array([values])
        prediction = int(model.predict(arr)[0])
        probability = model.predict_proba(arr)[0].tolist()
        return jsonify({
            "prediction":    prediction,
            "label":         "High Risk" if prediction == 1 else "Low Risk",
            "probability":   {"low_risk": round(probability[0], 4), "high_risk": round(probability[1], 4)},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)