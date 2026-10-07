# Import the ML inference functions
from src.inference import (
    review_expense,
    predict_category,
    predict_amount,
    detect_anomaly
)


# Test the combined ML system

# Create a function to test the review_expense() function
def test_review_expense():

    # Send one expense through the combined ML system
    result = review_expense(
        description="Woolworths Ellenbrook",
        amount=92.40,
        date="2026-10-05"
    )

    # Check that all expected results are returned
    assert "suggested_category" in result
    assert "expected_amount" in result
    assert "is_unusual" in result

    # Check that each result has the correct data type
    assert isinstance(result["suggested_category"], str)
    assert isinstance(result["expected_amount"], float)
    assert isinstance(result["is_unusual"], bool)


# Test the classification model

# Create a function to test the predict_category() function
def test_predict_category():

    # Send one expense through the category model
    result = predict_category(
        description="Woolworths Ellenbrook",
        amount=92.40,
        date="2026-10-05"
    )

    # Check that the predicted category is returned as text
    assert isinstance(result, str)


# Test the regression model

# Create a function to test the predict_amount() function
def test_predict_amount():

    # Send one expense through the amount model
    result = predict_amount(
        description="Woolworths Ellenbrook",
        category="Groceries",
        date="2026-10-05"
    )

    # Check that the predicted amount is returned as a decimal number
    assert isinstance(result, float)


# Test the anomaly detection model

# Create a function to test the detect_anomaly() function
def test_detect_anomaly():

    # Send one expense through the anomaly detection model
    result = detect_anomaly(
        description="Woolworths Ellenbrook",
        amount=92.40,
        category="Groceries",
        date="2026-10-05"
    )

    # Check that the anomaly result is returned as True or False
    assert isinstance(result, bool)