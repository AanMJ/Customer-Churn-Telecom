import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Function for Feature Engineering
def feature_engineering(data):
    st.title("Feature Engineering")

    # Drop non-numeric columns
    data_numeric = data.drop(['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService',
                              'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                              'Contract', 'PaperlessBilling', 'PaymentMethod'], axis=1)
    
    # Fill missing values
    data_numeric.fillna(0, inplace=True)

    # Select relevant features for modeling
    selected_features = select_features(data_numeric)
    st.write("Selected Features:", selected_features)

# Function to select features
def select_features(data):
    # Split features and target variable
    X = data.drop('Churn', axis=1)
    y = data['Churn']

    # Use feature importance for feature selection
    model = RandomForestClassifier()
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Select features with importance above threshold
    selected_features = X.columns[feature_importances > 0.01]  # Adjust threshold as needed

    return selected_features

def main():
    # Load Data
    data = pd.read_csv('telecomchurn.csv')
    
    # Perform Feature Engineering
    feature_engineering(data)

if __name__ == "__main__":
    main()