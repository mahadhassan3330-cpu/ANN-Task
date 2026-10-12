
import gradio as gr
import joblib
import numpy as np
from pathlib import Path

# ---------------- LOAD MODELS ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

diabetes_data = joblib.load(MODEL_DIR / "diabetes_model.joblib")
housing_data = joblib.load(MODEL_DIR / "california_housing_model.joblib")

diabetes_model = diabetes_data["model"]
diabetes_features = diabetes_data["feature_names"]

housing_model = housing_data["model"]
housing_features = housing_data["feature_names"]


# ---------------- FEATURE INFORMATION ----------------

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


def get_info(feature, info):
    """Match feature names without depending on capitalization."""
    for key, value in info.items():
        if key.lower() == feature.lower():
            return value
    return (feature.replace("_", " ").title(), 0)


# ---------------- PREDICTION FUNCTIONS ----------------

def predict_diabetes(*values):
    try:
        if any(value is None for value in values):
            return "Please complete all input fields."

        inputs = np.array(values, dtype=float).reshape(1, -1)
        prediction = float(diabetes_model.predict(inputs)[0])

        return (
            f"### {prediction:.2f}\n"
            "Predicted disease-progression score\n\n"
            "This is a model estimate, not a medical diagnosis."
        )
    except Exception as error:
        return f"Prediction failed: {error}"


def predict_housing(*values):
    try:
        if any(value is None for value in values):
            return "Please complete all input fields."

        inputs = np.array(values, dtype=float).reshape(1, -1)
        prediction = float(housing_model.predict(inputs)[0])

        if prediction < 0:
            note = "\n\nNote: The model returned a negative estimate."
        else:
            note = ""

        return (
            f"### ${prediction * 100000:,.0f}\n"
            "Estimated median house value\n\n"
            f"Raw model output: {prediction:.4f} "
            "(units of $100,000)"
            f"{note}"
        )
    except Exception as error:
        return f"Prediction failed: {error}"


# ---------------- INTERFACE ----------------

CSS = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}
#hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #172554, #1d4ed8);
    color: white;
    margin-bottom: 16px;
}
#hero h1 { color: white; margin-bottom: 8px; }
#hero p { color: #dbeafe; }
.section-title { margin-top: 8px; }
.result-box {
    border: 1px solid #bfdbfe;
    border-radius: 12px;
}
footer { visibility: hidden; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=CSS,
    title="Regression Lab | ANN Project"
) as demo:

    gr.HTML("""
    <div id="hero">
        <h1>Regression Lab</h1>
        <p>ANN Assignment · Linear Regression</p>
        <p>Explore two trained machine learning models using
        your own numerical inputs.</p>
    </div>
    """)

    with gr.Tabs():

        # -------- DIABETES TAB --------
        with gr.Tab("Diabetes Prediction"):
            gr.Markdown(
                "## Disease Progression Model\n"
                "Enter the 10 standardized features used by "
                "the scikit-learn Diabetes dataset."
            )

            diabetes_inputs = []

            for start in range(0, len(diabetes_features), 3):
                with gr.Row():
                    for feature in diabetes_features[start:start + 3]:
                        label, default = get_info(feature, DIABETES_INFO)

                        field = gr.Number(
                            label=label,
                            value=default,
                            precision=5
                        )
                        diabetes_inputs.append(field)

            diabetes_button = gr.Button(
                "Predict Disease Progression",
                variant="primary",
                size="lg"
            )

            diabetes_output = gr.Markdown(
                "Your prediction will appear here.",
                elem_classes=["result-box"]
            )

            diabetes_button.click(
                predict_diabetes,
                inputs=diabetes_inputs,
                outputs=diabetes_output
            )

        # -------- HOUSING TAB --------
        with gr.Tab("California Housing"):
            gr.Markdown(
                "## House Value Estimator\n"
                "Enter eight housing and location features. "
                "The estimated value is displayed in US dollars."
            )

            housing_inputs = []

            for start in range(0, len(housing_features), 3):
                with gr.Row():
                    for feature in housing_features[start:start + 3]:
                        label, default = get_info(feature, HOUSING_INFO)

                        field = gr.Number(
                            label=label,
                            value=default,
                            precision=4
                        )
                        housing_inputs.append(field)

            housing_button = gr.Button(
                "Estimate House Value",
                variant="primary",
                size="lg"
            )

            housing_output = gr.Markdown(
                "Your prediction will appear here.",
                elem_classes=["result-box"]
            )

            housing_button.click(
                predict_housing,
                inputs=housing_inputs,
                outputs=housing_output
            )

    gr.Markdown(
        "---\n"
        "**Models:** Scikit-learn Linear Regression · "
        "**Interface:** Gradio"
    )


if __name__ == "__main__":
    demo.launch()
