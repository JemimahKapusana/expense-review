# Tools used to load the models, work with paths, and create model input
from pathlib import Path
import joblib
import pandas as pd


# Find the trained_models folder
project_root = Path(__file__).resolve().parents[1]
model_folder = project_root / "trained_models"


# Load the saved models
category_model = joblib.load(
    model_folder / "expense_category_model.joblib"
)

amount_model = joblib.load(
    model_folder / "expense_amount_model.joblib"
)

anomaly_model = joblib.load(
    model_folder / "expense_anomaly_model.joblib"
)


# Function for running all models together
def review_expense(description, amount, date):

    expense_date = pd.to_datetime(date)

    category_input = pd.DataFrame([
        {
            "Description": description,
            "Amount": amount,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    predicted_category = category_model.predict(
        category_input
    )[0]

    amount_input = pd.DataFrame([
        {
            "Description": description,
            "Category": predicted_category,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    expected_amount = amount_model.predict(
        amount_input
    )[0]

    anomaly_input = pd.DataFrame([
        {
            "Description": description,
            "Amount": amount,
            "Category": predicted_category,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    anomaly_prediction = anomaly_model.predict(
        anomaly_input
    )[0]

    is_unusual = anomaly_prediction == -1

    return {
        "suggested_category": predicted_category,
        "expected_amount": round(float(expected_amount), 2),
        "is_unusual": bool(is_unusual)
    }


# Function for running the classification model only
def predict_category(description, amount, date):

    expense_date = pd.to_datetime(date)

    category_input = pd.DataFrame([
        {
            "Description": description,
            "Amount": amount,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    predicted_category = category_model.predict(
        category_input
    )[0]

    return predicted_category


# Function for running the regression model only
def predict_amount(description, category, date):

    expense_date = pd.to_datetime(date)

    amount_input = pd.DataFrame([
        {
            "Description": description,
            "Category": category,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    expected_amount = amount_model.predict(
        amount_input
    )[0]

    return round(float(expected_amount), 2)


# Function for running the anomaly detection model only
def detect_anomaly(description, amount, category, date):

    expense_date = pd.to_datetime(date)

    anomaly_input = pd.DataFrame([
        {
            "Description": description,
            "Amount": amount,
            "Category": category,
            "Year": expense_date.year,
            "Month": expense_date.month,
            "Day": expense_date.day,
            "DayOfWeek": expense_date.day_name(),
            "IsWeekend": expense_date.day_name() in ["Saturday", "Sunday"]
        }
    ])

    anomaly_prediction = anomaly_model.predict(
        anomaly_input
    )[0]

    is_unusual = anomaly_prediction == -1

    return bool(is_unusual)