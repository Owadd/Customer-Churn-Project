import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import joblib

def preprocess_and_train(train_file, model_file):
    """
    Function to preprocess data, train a SVM model, and save the pipeline containing both the preprocessor and the model.
    Print evaluation metrics like accuracy, recall, precision, and F1 score.

    Parameters:
    train_file (str): Path to the training set CSV file.
    model_file (str): Path to save the trained model file using joblib.
    """
    try:
        # Load the training dataset
        data = pd.read_csv(train_file)
        print("Training dataset loaded successfully.")
        
        # Convert TotalCharges to numeric, setting errors='coerce' to convert non-numeric values to NaN
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        # Fill missing values in numeric columns with the median value
        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())
        
        # Define features and target variable
        X = data.drop(columns=['customerID', 'Churn'])
        y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert target to binary
        
        # Define column transformer
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                                'PhoneService', 'MultipleLines', 'InternetService',
                                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies',
                                'Contract', 'PaperlessBilling', 'PaymentMethod']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ])
        
        # Create a pipeline that first transforms the data and then trains the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(kernel='rbf', C=1, gamma='auto', probability=True))
        ])
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        pipeline.fit(X_train, y_train)
        print("Model training completed.")
        
        # Evaluate the model
        y_pred = pipeline.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
    except FileNotFoundError:
        print(f"Error: The file {train_file} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {train_file} is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file {train_file} could not be parsed.")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Define file paths
train_file = 'customer_churn_train.csv'  # Replace with your actual training file path
model_file = 'pipeline_model.pkl'

# Preprocess data and train the model
preprocess_and_train(train_file, model_file)
