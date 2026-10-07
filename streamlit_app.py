import streamlit as st
import httpx


categories = [
    "Groceries",
    "Transport",
    "Dining",
    "Bills",
    "Entertainment",
    "Shopping",
    "Health",
    "Education",
    "Personal Care",
    "Travel",
    "Other"
]


selected_page = st.sidebar.selectbox(
    "Choose a feature",
    [
        "Category Prediction",
        "Expected Amount",
        "Unusual Expense Check",
        "Spending Patterns"
    ]
)


if selected_page == "Category Prediction":

    st.title("🏷️ Category Prediction")

    description = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    expense_date = st.date_input("Date")

    predict_button = st.button("Suggest Category")

    if predict_button:

        expense_data = {
            "description": description,
            "amount": amount,
            "date": str(expense_date)
        }

        response = httpx.post(
            "http://127.0.0.1:8000/predict-category",
            json=expense_data,
            timeout=60.0
        )

        result = response.json()

        st.write(
            "Suggested category:",
            result["suggested_category"]
        )


elif selected_page == "Expected Amount":

    st.title("💰 Expected Amount")

    description = st.text_input("Description")
    category = st.selectbox("Category", categories)
    expense_date = st.date_input("Date")

    predict_button = st.button("Predict Amount")

    if predict_button:

        expense_data = {
            "description": description,
            "category": category,
            "date": str(expense_date)
        }

        response = httpx.post(
            "http://127.0.0.1:8000/predict-amount",
            json=expense_data,
            timeout=60.0
        )

        result = response.json()

        st.write(
            "Expected amount:",
            f'${result["expected_amount"]:.2f}'
        )


elif selected_page == "Unusual Expense Check":

    st.title("🚨 Unusual Expense Check")

    description = st.text_input("Description")
    amount = st.number_input("Amount", min_value=0.0, step=0.01)
    category = st.selectbox("Category", categories)
    expense_date = st.date_input("Date")

    check_button = st.button("Check Expense")

    if check_button:

        expense_data = {
            "description": description,
            "amount": amount,
            "category": category,
            "date": str(expense_date)
        }

        response = httpx.post(
            "http://127.0.0.1:8000/detect-anomaly",
            json=expense_data,
            timeout=60.0
        )

        result = response.json()

        if result["is_unusual"]:
            st.warning("🚨 This expense looks unusual.")
        else:
            st.success("✅ This expense looks normal.")


elif selected_page == "Spending Patterns":

    st.title("🫧 Spending Patterns")

    # TODO: Revisit clustering and choose the final model.
    # TODO: Add a graph to display the spending patterns.

    st.write("Spending pattern visualisation coming soon.")