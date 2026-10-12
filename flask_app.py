
from flask import Flask, request, render_template_string
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

diabetes_data = joblib.load(MODEL_DIR / "diabetes_model.joblib")
housing_data = joblib.load(MODEL_DIR / "california_housing_model.joblib")

DIABETES_INFO = {
    "age": ("Age", 0),
    "sex": ("Sex", 0),
    "bmi": ("BMI", 0),
    "bp": ("Blood Pressure", 0),
    "s1": ("Serum Measurement S1", 0),
    "s2": ("Serum Measurement S2", 0),
    "s3": ("Serum Measurement S3", 0),
    "s4": ("Serum Measurement S4", 0),
    "s5": ("Serum Measurement S5", 0),
    "s6": ("Serum Measurement S6", 0),
}

HOUSING_INFO = {
    "MedInc": ("Median Income", 3.87),
    "HouseAge": ("House Age", 28.6),
    "AveRooms": ("Average Rooms", 5.4),
    "AveBedrms": ("Average Bedrooms", 1.1),
    "Population": ("Population", 1425),
    "AveOccup": ("Average Occupancy", 3.0),
    "Latitude": ("Latitude", 35.6),
    "Longitude": ("Longitude", -119.6),
}


def build_fields(features, info):
    fields = []

    for feature in features:
        details = None

        for key, value in info.items():
            if key.lower() == feature.lower():
                details = value
                break

        if details is None:
            details = (feature.replace("_", " ").title(), 0)

        label, default = details

        fields.append({
            "name": feature,
            "label": label,
            "default": default
        })

    return fields


MODELS = {
    "diabetes": {
        "model": diabetes_data["model"],
        "fields": build_fields(
            diabetes_data["feature_names"], DIABETES_INFO
        ),
        "title": "Diabetes Progression",
        "description": "Predict a numerical disease-progression score.",
        "icon": "✚"
    },
    "housing": {
        "model": housing_data["model"],
        "fields": build_fields(
            housing_data["feature_names"], HOUSING_INFO
        ),
        "title": "California Housing",
        "description": "Estimate a median house value from housing data.",
        "icon": "⌂"
    }
}


# ---------------- SHARED STYLES ----------------

STYLE = """
<style>
:root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --text: #172033;
    --muted: #64748b;
    --primary: #315ce8;
    --border: #e2e8f0;
}
* { box-sizing: border-box; }
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: Inter, "Segoe UI", Arial, sans-serif;
}
.navbar {
    height: 70px;
    padding: 0 6%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: white;
    border-bottom: 1px solid var(--border);
}
.brand {
    font-weight: 800;
    font-size: 19px;
    color: #172554;
}
.brand span { color: var(--primary); }
.nav-tag {
    color: var(--muted);
    font-size: 13px;
}
.hero {
    background: linear-gradient(125deg, #172554, #315ce8);
    color: white;
    padding: 42px 6%;
}
.hero h1 {
    font-size: clamp(28px, 4vw, 38px);
    margin: 0 0 10px;
}
.hero p {
    color: #dbeafe;
    line-height: 1.7;
    margin: 0;
    max-width: 680px;
}
.container {
    max-width: 1080px;
    padding: 30px 20px 55px;
    margin: auto;
}
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 22px;
    box-shadow: 0 6px 24px #0f172a08;
}
.card h2, .card h3 { margin-top: 0; }
.muted { color: var(--muted); line-height: 1.6; }
.model-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 22px;
}
.model-card {
    display: flex;
    flex-direction: column;
    min-height: 225px;
}
.model-icon {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: #eff6ff;
    color: var(--primary);
    font-size: 27px;
    margin-bottom: 20px;
}
.button {
    display: inline-block;
    padding: 12px 17px;
    margin-top: auto;
    background: var(--primary);
    color: white;
    text-decoration: none;
    text-align: center;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
}
.button:hover { background: #2449c7; }
.button.secondary {
    background: #eef2ff;
    color: #2449c7;
}
.eyebrow {
    text-transform: uppercase;
    letter-spacing: 1.3px;
    font-size: 11px;
    font-weight: 800;
    color: var(--primary);
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
    margin-top: 22px;
}
.field label {
    display: block;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
.field input {
    width: 100%;
    min-width: 0;
    border: 1px solid #d5ddeb;
    border-radius: 9px;
    padding: 12px;
    background: #fbfcff;
    font-size: 15px;
    color: var(--text);
    outline: none;
}
.field input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px #315ce81c;
}
.section-heading {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 28px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}
.section-number {
    background: #eff6ff;
    color: var(--primary);
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 12px;
    font-weight: 800;
}
.form-actions {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 28px;
}
.result {
    margin-top: 26px;
    padding: 24px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 13px;
}
.result-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #315ce8;
    font-weight: 800;
}
.result-value {
    font-size: clamp(26px, 4vw, 36px);
    font-weight: 800;
    overflow-wrap: anywhere;
    margin: 10px 0;
}
.result-note { color: #475569; font-size: 13px; line-height: 1.6; }
.error {
    padding: 14px;
    margin-top: 20px;
    border-radius: 9px;
    color: #991b1b;
    background: #fef2f2;
}
.footer {
    text-align: center;
    color: var(--muted);
    font-size: 13px;
    padding: 24px;
}
@media (max-width: 760px) {
    .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .model-grid { grid-template-columns: 1fr; }
    .hero { padding: 32px 22px; }
}
@media (max-width: 460px) {
    .feature-grid { grid-template-columns: 1fr; }
    .card { padding: 19px; }
    .nav-tag { display: none; }
}
</style>
"""


# ---------------- HOME PAGE ----------------

HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Regression Lab | ANN Project</title>
    {{ style|safe }}
</head>
<body>
<nav class="navbar">
    <div class="brand">Regression<span>Lab</span></div>
    <div class="nav-tag">ANN Assignment · Machine Learning</div>
</nav>

<section class="hero">
    <div class="eyebrow" style="color:#bfdbfe">Machine Learning Playground</div>
    <h1>Regression Prediction Studio</h1>
    <p>Explore two trained Linear Regression models.
    Enter numerical features and generate predictions
    through a simple interactive dashboard.</p>
</section>

<main class="container">
    <div class="card">
        <div class="eyebrow">Available models</div>
        <h2 style="margin-top:10px">Choose a prediction task</h2>
        <p class="muted">Select a model to open its organized input form.</p>

        <div class="model-grid">
            <div class="card model-card">
                <div class="model-icon">✚</div>
                <h3>Diabetes Progression</h3>
                <p class="muted">
                    Predict a numerical disease-progression score
                    from 10 standardized medical features.
                </p>
                <a class="button" href="/diabetes">Open Diabetes Model →</a>
            </div>

            <div class="card model-card">
                <div class="model-icon">⌂</div>
                <h3>California Housing</h3>
                <p class="muted">
                    Estimate median house value using 8 housing
                    and location features.
                </p>
                <a class="button" href="/housing">Open Housing Model →</a>
            </div>
        </div>
    </div>

    <div class="card">
        <div class="eyebrow">About the project</div>
        <h2 style="margin-top:10px">Two datasets. One algorithm.</h2>
        <p class="muted">
            Both models use Scikit-learn Linear Regression.
            The models were trained separately and are loaded
            from saved model files.
        </p>
    </div>
</main>

<footer class="footer">
    ANN Assignment · Linear Regression · Flask Interface
</footer>
</body>
</html>
"""


# ---------------- PREDICTION PAGE ----------------

PREDICTION_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ config.title }} | RegressionLab</title>
    {{ style|safe }}
</head>
<body>
<nav class="navbar">
    <a href="/" class="brand" style="text-decoration:none">
        Regression<span>Lab</span>
    </a>
    <a href="/" style="color:#315ce8;text-decoration:none;font-weight:700">
        ← All models
    </a>
</nav>

<section class="hero">
    <div class="eyebrow" style="color:#bfdbfe">Prediction workspace</div>
    <h1>{{ config.title }}</h1>
    <p>{{ config.description }}</p>
</section>

<main class="container">
    <div class="card">
        <div class="eyebrow">Model inputs</div>
        <h2 style="margin-top:10px">Enter feature values</h2>
        <p class="muted">
            Fill in each field below, then select the prediction button.
            The fields are arranged in three columns on larger screens.
        </p>

        {% if model_name == "diabetes" %}
        <div class="section-heading">
            <span class="section-number">01</span>
            <div>
                <strong>Standardized medical features</strong>
                <div class="muted" style="font-size:13px">
                    Use transformed dataset values, not ordinary clinical units.
                </div>
            </div>
        </div>
        {% else %}
        <div class="section-heading">
            <span class="section-number">01</span>
            <div>
                <strong>Housing and location features</strong>
                <div class="muted" style="font-size:13px">
                    Enter values using the dataset's original feature definitions.
                </div>
            </div>
        </div>
        {% endif %}

        <form method="POST">
            <div class="feature-grid">
                {% for field in config.fields %}
                <div class="field">
                    <label for="{{ field.name }}">{{ field.label }}</label>
                    <input
                        type="number"
                        step="any"
                        id="{{ field.name }}"
                        name="{{ field.name }}"
                        value="{{ request.form.get(field.name, field.default) }}"
                        required
                    >
                </div>
                {% endfor %}
            </div>

            <div class="form-actions">
                <button class="button" type="submit">
                    Generate Prediction →
                </button>
                <a class="button secondary" href="{{ request.path }}">
                    Reset values
                </a>
            </div>
        </form>

        {% if result is not none %}
        <div class="result">
            <div class="result-label">Prediction generated</div>
            <div class="result-value">{{ result }}</div>
            <div class="result-note">{{ result_note }}</div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
    </div>

    <div class="card">
        <h3>About this prediction</h3>
        {% if model_name == "diabetes" %}
        <p class="muted">
            The result is a numerical disease-progression score from
            the dataset. It is not a diagnosis or a substitute for medical advice.
        </p>
        {% else %}
        <p class="muted">
            The model predicts median house value in units of $100,000.
            The interface converts the prediction to US dollars.
            It is a model estimate, not a live property valuation.
        </p>
        {% endif %}
    </div>
</main>

<footer class="footer">
    RegressionLab · Scikit-learn Linear Regression
</footer>
</body>
</html>
"""


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template_string(
        HOME_TEMPLATE,
        style=STYLE
    )


@app.route("/<model_name>", methods=["GET", "POST"])
def predict(model_name):
    if model_name not in MODELS:
        return "Model not found", 404

    config = MODELS[model_name]
    result = None
    result_note = ""
    error = None

    if request.method == "POST":
        try:
            values = [
                float(request.form[field["name"]])
                for field in config["fields"]
            ]

            if not all(np.isfinite(values)):
                raise ValueError("Please enter finite numerical values.")

            inputs = np.array(values, dtype=float).reshape(1, -1)
            prediction = float(config["model"].predict(inputs)[0])

            if model_name == "housing":
                result = f"${prediction * 100000:,.0f}"
                result_note = (
                    f"Raw model output: {prediction:.4f} "
                    "units of $100,000. Estimates can be inaccurate "
                    "for inputs outside the training data range."
                )
            else:
                result = f"{prediction:.2f}"
                result_note = (
                    "Predicted disease-progression score. "
                    "This is not a medical diagnosis."
                )

        except (ValueError, KeyError):
            error = "Please enter a valid numerical value in every field."
        except Exception:
            error = "Prediction failed. Please verify your inputs and try again."

    return render_template_string(
        PREDICTION_TEMPLATE,
        style=STYLE,
        config=config,
        model_name=model_name,
        result=result,
        result_note=result_note,
        error=error,
        request=request
    )


if __name__ == "__main__":
    app.run(debug=True)
