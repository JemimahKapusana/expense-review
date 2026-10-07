from fastapi import FastAPI
from pydantic import BaseModel
from datetime import date
import logging

from src.inference import (
    review_expense,
    predict_category,
    predict_amount,
    detect_anomaly
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Create the FastAPI application
app = FastAPI()


# Data needed for the combined expense review
class ExpenseRequest(BaseModel):
    description: str
    amount: float
    date: date


# Endpoint for the combined expense review
@app.post("/review-expense")
def review_expense_api(expense: ExpenseRequest):

    # Record that the request reached the ML service
    logger.info("Expense review request received")

    try:
        # Send the expense through the combined ML system
        result = review_expense(
            description=expense.description,
            amount=expense.amount,
            date=expense.date
        )

        # Record that the review finished successfully
        logger.info("Expense review completed successfully")

        return result

    except Exception:

        # Record the error and where it happened
        logger.exception("Expense review failed")

        raise


# Data needed for the classification model
class CategoryRequest(BaseModel):
    description: str
    amount: float
    date: date


# Endpoint for the classification model
@app.post("/predict-category")
def predict_category_api(expense: CategoryRequest):

    # Record that the request reached the ML service
    logger.info("Category prediction request received")

    try:
        # Send the expense through the category model
        predicted_category = predict_category(
            description=expense.description,
            amount=expense.amount,
            date=expense.date
        )

        # Record that the prediction finished successfully
        logger.info("Category prediction completed successfully")

        return {
            "suggested_category": predicted_category
        }

    except Exception:

        # Record the error and where it happened
        logger.exception("Category prediction failed")

        raise


# Data needed for the regression model
class AmountRequest(BaseModel):
    description: str
    category: str
    date: date


# Endpoint for the regression model
@app.post("/predict-amount")
def predict_amount_api(expense: AmountRequest):

    # Record that the request reached the ML service
    logger.info("Amount prediction request received")

    try:
        # Send the expense through the amount model
        expected_amount = predict_amount(
            description=expense.description,
            category=expense.category,
            date=expense.date
        )

        # Record that the prediction finished successfully
        logger.info("Amount prediction completed successfully")

        return {
            "expected_amount": expected_amount
        }

    except Exception:

        # Record the error and where it happened
        logger.exception("Amount prediction failed")

        raise


# Data needed for the anomaly detection model
class AnomalyRequest(BaseModel):
    description: str
    amount: float
    category: str
    date: date


# Endpoint for the anomaly detection model
@app.post("/detect-anomaly")
def detect_anomaly_api(expense: AnomalyRequest):

    # Record that the request reached the ML service
    logger.info("Anomaly detection request received")

    try:
        # Send the expense through the anomaly detection model
        is_unusual = detect_anomaly(
            description=expense.description,
            amount=expense.amount,
            category=expense.category,
            date=expense.date
        )

        # Record that the prediction finished successfully
        logger.info("Anomaly detection completed successfully")

        return {
            "is_unusual": is_unusual
        }

    except Exception:

        # Record the error and where it happened
        logger.exception("Anomaly detection failed")

        raise