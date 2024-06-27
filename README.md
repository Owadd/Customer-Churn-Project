# Credit Risk Assessment

Welcome to the **Credit Risk Assessment** repository! 🏦📊 Here, we're all about predicting whether customers are likely to churn using some fancy machine learning models. Get ready to dive into a world of data, predictions, and a sprinkle of fun! 🎉

## Table of Contents
- [Introduction](#introduction)
- [Files Included](#files-included)
- [Setup Instructions](#setup-instructions)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Introduction

Are you ready to predict customer churn with the power of Python and machine learning? This project provides a complete pipeline from data preprocessing to model training and evaluation. With this toolkit, you'll be able to assess customer credit risk and predict churn with confidence. Let's get started!

## Files Included

Here's a quick rundown of the magic you'll find in this repository:

### 1. `churn_predictor.py`
The heart of our project. This script collects user input, loads the pre-trained model, and predicts customer churn. It's like a crystal ball but more tech-savvy! 🔮

### 2. `churn_predictor_app.py`
Similar to `churn_predictor.py`, but with added awesomeness for deployment as a Streamlit app. Bring your predictions to life on the web! 🌐

### 3. `dataset_splitter.py`
Splits your dataset into training and testing sets with a sprinkle of randomness. Perfect for when you need a fresh split every time. 🎲

### 4. `svm_classifier_train.py`
This script preprocesses data, trains an SVM model, and saves the pipeline. It's the engine room of our machine learning journey. 🚂

### 5. `svm_test.py`
Another version of the training script for testing purposes. Because who doesn't love some extra validation? ✅

### 6. `pipeline_model.pkl`
Our pre-trained model pipeline, ready to make predictions. It's the secret sauce! 🌟

## Setup Instructions

Before you can start predicting churn, you need to set up your environment. Here's how:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/credit-risk-assessment.git
    cd credit-risk-assessment
    ```

2. **Install the required libraries**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure you have the necessary data**:
    Make sure you have the `customer_churn.csv` dataset in the root directory.

## How to Use

### Running the Churn Predictor

To predict customer churn using the command-line interface:

```bash
python churn_predictor.py
```

Follow the prompts to enter customer data and get instant predictions! 📈

### Running the Streamlit App

To launch the web app:

```bash
streamlit run churn_predictor_app.py
```

Open the provided URL in your browser and enjoy the interactive predictions. 🚀

### Splitting the Dataset

To split the dataset into training and testing sets:

```bash
python dataset_splitter.py
```

### Training the Model

To preprocess the data and train the SVM model:

```bash
python svm_classifier_train.py
```

### Testing the Model

To validate the model with additional testing:

```bash
python svm_test.py
```

## Project Structure

Here's a peek into the project's structure:

```
credit-risk-assessment/
│
├── churn_predictor.py
├── churn_predictor_app.py
├── dataset_splitter.py
├── svm_classifier_train.py
├── svm_test.py
├── pipeline_model.pkl
├── customer_churn.csv
├── customer_churn_test.csv
├── customer_churn_train.csv
└── README.md
```

## Contributing

We welcome contributions with open arms! If you have ideas for improvements or new features, feel free to fork the repo, make your changes, and submit a pull request. Let's make this project even more awesome together! 💪


---

Happy predicting! May the churn odds be ever in your favor! 🍀
