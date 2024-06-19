import streamlit as st
import pandas as pd
import joblib


@st.cache_data(allow_output_mutation=True)
def load_model():
    return joblib.load('pipeline_model.pkl')


def main():
    st.title("Customer Churn Prediction Tool")

    # Sidebar for user input
    st.sidebar.header("Enter Customer Details")
    user_data = {}
    user_data['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    user_data['SeniorCitizen'] = st.sidebar.radio("Senior Citizen", [0, 1])
    user_data['Partner'] = st.sidebar.radio("Has Partner", ["Yes", "No"])
    user_data['Dependents'] = st.sidebar.radio("Has Dependents", ["Yes", "No"])
    user_data['tenure'] = st.sidebar.slider("Months with Company", min_value=0, max_value=72, value=12, step=1)
    user_data['PhoneService'] = st.sidebar.radio("Phone Service", ["Yes", "No"])
    user_data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    user_data['InternetService'] = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    user_data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    user_data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    user_data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    user_data['TechSupport'] = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    user_data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    user_data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    user_data['Contract'] = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    user_data['PaperlessBilling'] = st.sidebar.radio("Paperless Billing", ["Yes", "No"])
    user_data['PaymentMethod'] = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    user_data['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=1.0, value=50.0)
    user_data['TotalCharges'] = st.sidebar.number_input("Total Charges to Date", min_value=0.0, step=1.0, value=500.0)

    # Load the model and perform prediction
    model = load_model()
    if st.button("Predict"):
        user_input_df = pd.DataFrame([user_data])
        prediction = model.predict(user_input_df)
        churn_probability = model.predict_proba(user_input_df)[0][1]
        churn_prediction = 'Yes' if prediction[0] == 1 else 'No'

        # Display prediction results
        st.subheader("Prediction Results")
        st.write(f"The customer will churn: {churn_prediction}")
        st.write(f"Probability of churn: {churn_probability:.4f}")


if __name__ == "__main__":
    main()
