import streamlit as st
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

# Column mapping for messy dataset headers
col_map = {
    "ï»¿Age of the patient": "age",
    "Gender of the patient": "gender",
    "Total Bilirubin": "total_bilirubin",
    "Direct Bilirubin": "direct_bilirubin",
    "Â Alkphos Alkaline Phosphotase": "alkphos",
    "Â Sgpt Alamine Aminotransferase": "sgpt",
    "Sgot Aspartate Aminotransferase": "sgot",
    "Total Protiens": "total_proteins",
    "Â ALB Albumin": "albumin",
    "A/G Ratio Albumin and Globulin Ratio": "ag_ratio",
    "Result": "dataset"
}

# ================================
# Train the model at app startup
# ================================
@st.cache_resource
def train_model():
    # Load training dataset with encoding fix
    df_train = pd.read_csv(
        "liver_patient_data/Liver Patient Dataset (LPD)_train.csv",
        encoding="latin1"
    )

    # Apply column mapping + normalize names
    df_train = df_train.rename(columns=col_map)
    df_train.columns = df_train.columns.str.strip().str.lower()

    st.write("Cleaned train columns:", df_train.columns.tolist())

    # Encode gender
    df_train["gender"] = LabelEncoder().fit_transform(df_train["gender"])
    df_train["target"] = (df_train["dataset"] == 1).astype(int)

    X = df_train.drop(columns=["dataset", "target"])
    y = df_train["target"]

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_imputed, y)

    return model, imputer


model, imputer = train_model()

# ================================
# Streamlit UI
# ================================
st.title("🧪 Liver Disease Prediction Demo")
st.write("This app retrains the model on Kaggle liver dataset and allows predictions.")

# Sidebar Navigation
page = st.sidebar.radio("Choose a page:", ["Prediction", "Model Performance"])

# ================================
# Prediction Page
# ================================
if page == "Prediction":
    st.header("Enter patient details")

    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tb = st.number_input("Total Bilirubin", min_value=0.0, value=1.0)
    db = st.number_input("Direct Bilirubin", min_value=0.0, value=0.3)
    alkphos = st.number_input("Alkaline Phosphatase", min_value=0.0, value=200.0)
    alt = st.number_input("ALT (SGPT)", min_value=0.0, value=30.0)
    ast = st.number_input("AST (SGOT)", min_value=0.0, value=35.0)
    tp = st.number_input("Total Proteins", min_value=0.0, value=6.8)
    albumin = st.number_input("Albumin", min_value=0.0, value=3.5)
    ag_ratio = st.number_input("A/G Ratio", min_value=0.0, value=1.0)

    # Prepare input
    gender_enc = 1 if gender == "Male" else 0
    features = np.array([[age, gender_enc, tb, db, alkphos, alt, ast, tp, albumin, ag_ratio]])

    # Apply imputer
    features_imputed = imputer.transform(features)

    # Predict
    if st.button("Predict"):
        pred = model.predict(features_imputed)[0]
        prob = model.predict_proba(features_imputed)[0][1]
        if pred == 1:
            st.error(f"⚠️ Likely Liver Disease (Probability: {prob:.2f})")
        else:
            st.success(f"✅ No Liver Disease Detected (Probability: {prob:.2f})")

# ================================
# Model Performance Page
# ================================
elif page == "Model Performance":
    st.header("Model Evaluation on Test Dataset")

    try:
        df_test = pd.read_csv(
            "liver_patient_data/Liver Patient Dataset (LPD)_test.csv",
            encoding="latin1"
        )

        # Apply column mapping + normalize names
        df_test = df_test.rename(columns=col_map)
        df_test.columns = df_test.columns.str.strip().str.lower()

        st.write("Cleaned test columns:", df_test.columns.tolist())

        # Encode gender
        df_test["gender"] = LabelEncoder().fit_transform(df_test["gender"])
        df_test["target"] = (df_test["dataset"] == 1).astype(int)

        X_test = df_test.drop(columns=["dataset", "target"])
        y_test = df_test["target"]

        # Impute missing values
        X_test_imputed = imputer.transform(X_test)

        # Predictions
        y_pred = model.predict(X_test_imputed)
        y_prob = model.predict_proba(X_test_imputed)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        st.write("**Accuracy:**", round(acc, 3))
        st.write("**ROC AUC:**", round(auc, 3))
        st.write("**F1 Score:**", round(f1, 3))

        st.text("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    except Exception as e:
        st.error("Test dataset not found or error during evaluation.")
        st.text(str(e))
