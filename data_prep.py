import streamlit as st
import pandas as pd
import numpy as np

# Function for Data Acquisition and Preprocessing
def data_acquisition():
    st.title("Data Acquisition and Preprocessing")
    
    # Load Data
    @st.cache
    def load_data():
        data = pd.read_csv('telecomchurn.csv')
        return data
    
    data_load_state = st.text('Loading data...')
    data = load_data()
    data_load_state.text("Data loaded successfully!")
    
    # Display first few rows of the dataset
    st.subheader('Data Overview')
    st.write(data.head())
    
    # Check for duplicate customer IDs
    st.subheader('Duplicate Customer IDs')
    unique_customers = data['customerID'].nunique()
    duplicated_customers = data['customerID'].duplicated().sum()
    st.write(f"Total unique customer IDs: {unique_customers}")
    st.write(f"Number of duplicated customer IDs: {duplicated_customers}")
    
    # Check for missing values
    st.subheader('Missing Values')
    missing_values = data.isna().sum()
    st.write(missing_values)

    # Convert 'TotalCharges' to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].replace(' ', np.nan, inplace=True)
    data['TotalCharges'] = data['TotalCharges'].astype(float)

    st.subheader('Data after Preprocessing')
    st.write(data.head())
    
# Main function to run the Streamlit app
def main():
    data_acquisition()

if __name__ == "__main__":
    main()