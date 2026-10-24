# Hydration Risk Level

An interactive Python/Streamlit machine-learning app for estimating whether a user's hydration level is likely to be `Good` or `Poor` from the supplied daily water intake dataset.

## Run locally

```bash
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The app reads `Data/Daily_Water_Intake.csv` at runtime.

## What the app does

- Validates and loads the dataset.
- Compares Logistic Regression, Random Forest, and Extra Trees.
- Uses stratified 5-fold cross-validation and selects the model with the highest macro-F1.
- Uses balanced class weights to account for the `Good`/`Poor` class imbalance.
- Reports cross-validation scores, hold-out metrics, ROC-AUC, and a confusion matrix.
- Gives a user prediction with class probabilities, an intake-versus-estimate graph, and a simple educational water-intake estimate.

The water estimate is a transparent heuristic for understanding the chart and is not medical advice. The ML prediction is based on patterns in this dataset, not a clinical diagnosis.
