import pandas as pd
import joblib

def collect_user_input():
    """
    Collect user input for the prediction model.

    Returns:
    user_data (dict): Dictionary containing the user's input data.
    """
    user_data = {}
    try:
        user_data['gender'] = input("Enter Gender (Male/Female): ").strip()
        user_data['SeniorCitizen'] = int(input("Are you a Senior Citizen? Enter 1 for Yes and 0 for No: ").strip())
        user_data['Partner'] = input("Do you have a partner? (Yes/No): ").strip()
        user_data['Dependents'] = input("Do you have dependents? (Yes/No): ").strip()
        user_data['tenure'] = int(input("How many months have you been with the company? (e.g., 18): ").strip())
        user_data['PhoneService'] = input("Do you have phone service? (Yes/No): ").strip()
        user_data['MultipleLines'] = input("Do you have multiple phone lines? (Yes/No/No phone service): ").strip()
        user_data['InternetService'] = input("What type of internet service do you have? (DSL/Fiber optic/No): ").strip()
        user_data['OnlineSecurity'] = input("Do you have online security? (Yes/No/No internet service): ").strip()
        user_data['OnlineBackup'] = input("Do you have online backup? (Yes/No/No internet service): ").strip()
        user_data['DeviceProtection'] = input("Do you have device protection? (Yes/No/No internet service): ").strip()
        user_data['TechSupport'] = input("Do you have tech support? (Yes/No/No internet service): ").strip()
        user_data['StreamingTV'] = input("Do you stream TV? (Yes/No/No internet service): ").strip()
        user_data['StreamingMovies'] = input("Do you stream movies? (Yes/No/No internet service): ").strip()
        user_data['Contract'] = input("What is your contract type? (Month-to-month/One year/Two year): ").strip()
        user_data['PaperlessBilling'] = input("Do you use paperless billing? (Yes/No): ").strip()
        user_data['PaymentMethod'] = input("What is your payment method? (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ").strip()
        user_data['MonthlyCharges'] = float(input("What is your monthly charge? (e.g., 50.3): ").strip())
        user_data['TotalCharges'] = float(input("What is your total charge to date? (e.g., 913.3): ").strip())
    except ValueError as e:
        print(f"Invalid input: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while collecting user input: {e}")
        exit()
    return user_data

def main():
    """
    Main function to load the model and make predictions based on user input.
    """
    try:
        print("Welcome to the Customer Churn Prediction Tool")
        user_data = collect_user_input()

        # Load the entire pipeline
        pipeline = joblib.load('pipeline_model.pkl')
        
        # Transform the user input using the pipeline
        user_input_df = pd.DataFrame([user_data])
        prediction = pipeline.predict(user_input_df)
        churn_probability = pipeline.predict_proba(user_input_df)[0][1]

        churn_prediction = 'Yes' if prediction[0] == 1 else 'No'
        print(f"Prediction: The customer will churn: {churn_prediction}")
        print(f"Probability of churn: {churn_probability:.4f}")

    except FileNotFoundError:
        print("Error: Model file not found. Please ensure 'pipeline_model.pkl' is in the current directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
