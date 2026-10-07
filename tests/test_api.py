# Import FastAPI's testing tool and our API
from fastapi.testclient import TestClient
from src.api import app


# Create a test client for the API
client = TestClient(app)


# Test the combined Expense Review endpoint

# Create a function to test the /review-expense endpoint
def test_review_expense_endpoint():

    # Send an expense to the API
    response = client.post(
        "/review-expense",
        json={
            "description": "Woolworths Ellenbrook",
            "amount": 92.40,
            "date": "2026-10-06"
        }
    )

    # Check that the API request was successful
    assert response.status_code == 200

    # Convert the API response into Python data
    result = response.json()

    # Check that all expected results are returned
    assert "suggested_category" in result
    assert "expected_amount" in result
    assert "is_unusual" in result


# Test the classification endpoint

# Create a function to test the /predict-category endpoint
def test_predict_category_endpoint():

    # Send an expense to the API
    response = client.post(
        "/predict-category",
        json={
            "description": "Woolworths Ellenbrook",
            "amount": 92.40,
            "date": "2026-10-06"
        }
    )

    # Check that the API request was successful
    assert response.status_code == 200


# Test the regression endpoint

# Create a function to test the /predict-amount endpoint
def test_predict_amount_endpoint():

    # Send an expense to the API
    response = client.post(
        "/predict-amount",
        json={
            "description": "Woolworths Ellenbrook",
            "category": "Groceries",
            "date": "2026-10-06"
        }
    )

    # Check that the API request was successful
    assert response.status_code == 200


# Test the anomaly detection endpoint

# Create a function to test the /detect-anomaly endpoint
def test_detect_anomaly_endpoint():

    # Send an expense to the API
    response = client.post(
        "/detect-anomaly",
        json={
            "description": "Woolworths Ellenbrook",
            "amount": 92.40,
            "category": "Groceries",
            "date": "2026-10-06"
        }
    )

    # Check that the API request was successful
    assert response.status_code == 200


# Test that the API rejects missing data

# Create a function to test a request with a missing description
def test_review_expense_missing_description():

    # Send incomplete data to the API
    response = client.post(
        "/review-expense",
        json={
            "amount": 92.40,
            "date": "2026-10-06"
        }
    )

    # Check that FastAPI rejects the invalid request
    assert response.status_code == 422


# Test that the API rejects an invalid date

# Create a function to test a request with an invalid date
def test_review_expense_invalid_date():

    # Send an invalid date to the API
    response = client.post(
        "/review-expense",
        json={
            "description": "Woolworths Ellenbrook",
            "amount": 92.40,
            "date": "banana"
        }
    )

    # Check that FastAPI rejects the invalid request
    assert response.status_code == 422