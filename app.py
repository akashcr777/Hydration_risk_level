"""Hydration Risk Level - interactive ML app.

Run with: streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


st.set_page_config(
    page_title="Hydration Risk Level",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "Data" / "Daily_Water_Intake.csv"
TARGET = "Hydration Level"
NUMERIC_FEATURES = ["Age", "Weight (kg)", "Daily Water Intake (liters)"]
CATEGORICAL_FEATURES = ["Gender", "Physical Activity Level", "Weather"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainingResult:
    data: pd.DataFrame
    models: dict[str, Pipeline]
    cv_scores: pd.DataFrame
    test_metrics: pd.DataFrame
    best_name: str
    best_model: Pipeline
    y_test: pd.Series
    y_pred: np.ndarray
    y_probability: np.ndarray
    report: dict[str, Any]


def _load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return data.dropna(subset=FEATURES + [TARGET]).copy()


def _preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def _candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", _preprocessor(scale_numeric=True)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", _preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=180,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                ("preprocessor", _preprocessor()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


@st.cache_resource(show_spinner="Training and comparing models…")
def train_models() -> TrainingResult:
    data = _load_data()
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    models = _candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []
    for name, model in models.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
        cv_rows.append({"Model": name, "CV Macro-F1": scores.mean(), "CV Std": scores.std()})

    cv_scores = pd.DataFrame(cv_rows).sort_values("CV Macro-F1", ascending=False).reset_index(drop=True)
    best_name = str(cv_scores.iloc[0]["Model"])
    for model in models.values():
        model.fit(x_train, y_train)

    metrics_rows = []
    for name, model in models.items():
        prediction = model.predict(x_test)
        metrics_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Macro Precision": precision_score(y_test, prediction, average="macro", zero_division=0),
                "Macro Recall": recall_score(y_test, prediction, average="macro", zero_division=0),
                "Macro F1": f1_score(y_test, prediction, average="macro", zero_division=0),
            }
        )
    test_metrics = pd.DataFrame(metrics_rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    best_model = models[best_name]
    y_pred = best_model.predict(x_test)
    probabilities = best_model.predict_proba(x_test)
    classes = list(best_model.classes_)
    poor_index = classes.index("Poor") if "Poor" in classes else 0
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Store the positive-class probability with a stable meaning for later charts.
    y_probability = probabilities[:, poor_index]
    report["roc_auc"] = roc_auc_score((y_test == "Poor").astype(int), y_probability)
    return TrainingResult(data, models, cv_scores, test_metrics, best_name, best_model, y_test, y_pred, y_probability, report)


def _water_guidance(row: dict[str, Any]) -> tuple[float, float, str]:
    """A simple educational estimate, not medical advice."""
    baseline = float(row["Weight (kg)"]) * 0.035
    activity_adjustment = {"Low": 0.0, "Moderate": 0.35, "High": 0.65}[row["Physical Activity Level"]]
    weather_adjustment = {"Cold": 0.0, "Normal": 0.15, "Hot": 0.4}[row["Weather"]]
    target = baseline + activity_adjustment + weather_adjustment
    current = float(row["Daily Water Intake (liters)"])
    difference = target - current
    if difference > 0.15:
        message = f"Consider roughly {difference:.1f} L more across the day than your current intake."
    elif difference < -0.15:
        message = f"Your entered intake is about {abs(difference):.1f} L above this simple estimate."
    else:
        message = "Your entered intake is close to this simple estimate."
    return target, difference, message


def _risk_badge(label: str, probability: float) -> None:
    if label == "Poor":
        st.error(f"Predicted hydration level: {label} · {probability:.0%} poor-risk probability")
    else:
        st.success(f"Predicted hydration level: {label} · {1 - probability:.0%} good probability")


def main() -> None:
    st.title("💧 Hydration Risk Level")
    st.caption("A machine-learning demo that estimates hydration level from daily habits and conditions.")

    try:
        result = train_models()
    except Exception as exc:
        st.error(f"Could not load or train the model: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Enter your details")
        age = st.slider("Age", 10, 100, 30)
        gender = st.selectbox("Gender", sorted(result.data["Gender"].unique()))
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=70.0, step=0.5)
        intake = st.number_input(
            "Daily water intake (liters)", min_value=0.0, max_value=15.0, value=2.5, step=0.1
        )
        activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])
        weather = st.selectbox("Weather", ["Cold", "Normal", "Hot"])
        user_row = {
            "Age": age,
            "Gender": gender,
            "Weight (kg)": weight,
            "Daily Water Intake (liters)": intake,
            "Physical Activity Level": activity,
            "Weather": weather,
        }

    user_df = pd.DataFrame([user_row], columns=FEATURES)
    label = str(result.best_model.predict(user_df)[0])
    classes = list(result.best_model.classes_)
    probs = result.best_model.predict_proba(user_df)[0]
    poor_probability = float(probs[classes.index("Poor")])
    target, difference, guidance = _water_guidance(user_row)

    st.subheader("Your prediction")
    _risk_badge(label, poor_probability)
    col1, col2, col3 = st.columns(3)
    col1.metric("Poor-risk probability", f"{poor_probability:.1%}")
    col2.metric("Estimated daily target", f"{target:.2f} L")
    col3.metric("Entered intake", f"{intake:.2f} L", delta=f"{intake - target:+.2f} L vs estimate")
    st.info(guidance + " This estimate is for education only and is not medical advice.")

    chart_col, explanation_col = st.columns([1, 1])
    with chart_col:
        st.markdown("**What drives the result?**")
        probability_df = pd.DataFrame({"Probability": probs}, index=classes)
        st.bar_chart(probability_df, y="Probability", color="#2a9d8f")
        st.caption("The model's probabilities for each class. A result near 50% is less certain.")
    with explanation_col:
        st.markdown("**Your intake compared with the estimate**")
        compare_df = pd.DataFrame({"Liters": [intake, target]}, index=["Your intake", "Estimated target"])
        st.bar_chart(compare_df, y="Liters", color="#457b9d")
        st.caption("The target uses a transparent weight/activity/weather heuristic, separate from the ML prediction.")

    with st.expander("Model quality and comparison"):
        st.write(f"Best model selected by 5-fold cross-validated macro-F1: **{result.best_name}**")
        st.dataframe(result.cv_scores.style.format({"CV Macro-F1": "{:.3f}", "CV Std": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.markdown("**Hold-out test-set metrics**")
        st.dataframe(result.test_metrics.style.format({c: "{:.3f}" for c in result.test_metrics.columns[1:]}), use_container_width=True, hide_index=True)
        st.metric("Best-model ROC-AUC", f"{result.report['roc_auc']:.3f}")
        matrix = confusion_matrix(result.y_test, result.y_pred, labels=["Good", "Poor"])
        st.markdown("**Confusion matrix on the test set**")
        st.dataframe(pd.DataFrame(matrix, index=["Actual Good", "Actual Poor"], columns=["Predicted Good", "Predicted Poor"]), use_container_width=True)

    with st.expander("Dataset overview"):
        overview_col, class_col = st.columns(2)
        overview_col.write(f"Rows used: **{len(result.data):,}**")
        overview_col.write(f"Features: **{len(FEATURES)}**")
        class_col.bar_chart(result.data[TARGET].value_counts().rename("Records"), color="#e76f51")
        st.caption("The app compares balanced models so the less common Poor class receives appropriate weight during training.")


if __name__ == "__main__":
    main()
