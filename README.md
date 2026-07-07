# Diabetes Health Dashboard (Streamlit)

A Streamlit web app that estimates diabetes risk and flags individual health
parameters (glucose, blood pressure, BMI, insulin, etc.) using a Random Forest
model trained on the Pima Indians Diabetes Database.

## Live Demo

🌐 [Live Demo](https://glycoscan-k7jecpfxz98wveubzmfayp.streamlit.app/)

## Folder contents

```
GlycoScan                     # Repository name
├── diabetes_rf_eda.ipynb     # EDA and Model Training notebook  
├── streamlit_app/
|        ├── app.py            # Streamlit app      
|        └── model/
|           ├── rf_diabetes_model.joblib   # Trained Random Forest model
|            ├── scaler.joblib              # Fitted StandardScaler
|            ├── feature_names.joblib       # Feature column order
|            └── diabetes.csv               # Reference dataset (for comparison charts)
|     
└── requirements.txt
```

## How to run locally

1. Install dependencies (Python 3.9+ recommended):

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the app from inside this folder:

   ```bash
   streamlit run app.py
   ```

## What the app includes

- **🔮 Risk Prediction** — enter your health details in the sidebar and get an
  estimated diabetes risk probability with a gauge chart.
- **📊 Parameter Check-up** — each input (glucose, blood pressure, BMI,
  insulin, skin thickness, age, pedigree function) is checked against general
  clinical reference ranges, with plain-language flags telling you which
  values might be worth getting checked, plus a chart comparing your numbers
  to dataset averages for diabetic vs. non-diabetic patients.
- **🌲 Model Insights** — feature importance chart from the trained Random
  Forest, plus a quick look at the underlying dataset.
- **💡 Health Tips** — general, non-personalized lifestyle guidance for
  diabetes awareness (nutrition, activity, sleep, monitoring, etc.).
- **ℹ️ About** — explains what the app does/doesn't do, with a medical
  disclaimer.

## Notes

- The model and scaler are pre-trained (see the companion
  `diabetes_rf_eda.ipynb` notebook) and loaded directly — the app does not
  retrain on startup, so it launches quickly.
- Reference ranges used for parameter flags are general clinical rules of
  thumb for educational purposes; they are **not** a diagnostic tool.
- This app is not a substitute for professional medical advice.
