import pandas as pd
from sklearn.model_selection import train_test_split
import random

def split_dataset(input_file, train_file, test_file, test_size=0.2):
    """
    Function to split dataset into training and testing sets with a random seed.
    
    Parameters:
    input_file (str): Path to the input CSV file containing the dataset.
    train_file (str): Path to save the training set CSV file.
    test_file (str): Path to save the testing set CSV file.
    test_size (float): Proportion of the dataset to include in the test split. Default is 0.2 (20%).
    """
    try:
        # Load the dataset
        data = pd.read_csv(input_file)
        print("Dataset loaded successfully.")
        
        # Check for necessary columns
        required_columns = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", 
                            "tenure", "PhoneService", "MultipleLines", "InternetService", 
                            "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                            "TechSupport", "StreamingTV", "StreamingMovies", 
                            "Contract", "PaperlessBilling", "PaymentMethod", 
                            "MonthlyCharges", "TotalCharges", "Churn"]
        
        if not all(column in data.columns for column in required_columns):
            raise ValueError("Dataset does not contain all the required columns.")
        
        # Generate a random seed
        random_state = random.randint(0, 10000)
        print(f"Using random seed: {random_state}")
        
        # Split the dataset into training and testing sets
        train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        print("Dataset split into training and testing sets.")
        
        # Save the training and testing sets to CSV files
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)
        print(f"Training set saved to {train_file}")
        print(f"Testing set saved to {test_file}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_file} is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file {input_file} could not be parsed.")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Define file paths
input_file = 'customer_churn.csv'  # Replace with your actual input file path
train_file = 'customer_churn_train.csv'
test_file = 'customer_churn_test.csv'

# Split the dataset
split_dataset(input_file, train_file, test_file)
