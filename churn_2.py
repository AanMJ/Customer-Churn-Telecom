import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Load Data
@st.cache_data
def load_data():
    data = pd.read_csv('telecomchurn.csv')
    return data

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text("Data loaded successfully!")

# Custom CSS for improved styling
custom_css = """
<style>
body {
    background-image: url('https://images.unsplash.com/photo-1569982175971-d92b01cf8694?q=80&w=1935&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
    background-position: center;
    background-size: cover;
    background-repeat: no-repeat;
}

.stApp .main{
 background-image:linear-gradient(to right, #0f0c29 0%, #302b63 33%, #24243e 100%);
 transform:translatex(0px) translatey(0px);
 font-family:'Sonsie One', display;
}
.stApp .main{
 background-image:linear-gradient(to right, #0f0c29 0%, #302b63 33%, #24243e 100%);
 transform:translatex(0px) translatey(0px);
 font-family:'Sonsie One', display;
}

/* Span Tag */
#telco-customer-churn-prediction .st-emotion-cache-zt5igj span{
 box-shadow:0px 0px 0px -50px #fafafa inset;
 transform:translatex(0px) translatey(0px);
}

/* Span Tag */
#data-overview .st-emotion-cache-zt5igj span{
 text-shadow:rgb(255, 0, 0) -1px 0px 0px, rgb(0, 255, 255) 1px 0px 0px;
}

/* Glide data editor */
.st-emotion-cache-1n76uvr .gdg-wmyidgi .glideDataEditor{
 background-image:none;
 backdrop-filter:blur(0.24px) hue-rotate(160deg);
 border-top-width:0px;
 box-shadow:0px 0px 8px 0px #fafafa;
}

/* Data frame */
.st-emotion-cache-1n76uvr .st-emotion-cache-9aoz2h .stDataFrame{
 border-top-left-radius:28px;
 border-top-right-radius:18px;
 border-bottom-left-radius:15px;
 border-bottom-right-radius:13px;
}

/* Emotion cache 16txtl3 */
.st-emotion-cache-1cypcdb .st-emotion-cache-6qob1r .st-emotion-cache-16txtl3{
 backdrop-filter:blur(7.26px);
 background-size:contain;
 background-blend-mode:luminosity;
 background-image:linear-gradient(to right, #070000 0%, #4c0001 30%, #070000 100%);
}

/* Emotion cache hc3laj */
.st-emotion-cache-198qupu .stButton .st-emotion-cache-hc3laj{
 background-color:#c118c7;
 backdrop-filter: saturate(4.78);
}
                              
.stBlockqoute {
    background: rgba(255, 255, 255, 0.5);
    border-left: 5px solid #3498db;
}

.stSidebar .sidebar-content {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}

.stSidebar .sidebar-content .widget stButton {
    margin-top: 20px;
}

.stMain {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar Navigation and Input
st.sidebar.title("Customer Churn Prediction")

# Display general information about the dataset
st.sidebar.subheader('Dataset Info')
st.sidebar.write(data.info())

# Prediction Input
st.sidebar.subheader('Make Predictions')

example = {
    'gender': st.sidebar.slider('Gender (Female: 0, Male: 1)', 0, 1),
    'SeniorCitizen': st.sidebar.slider('Senior Citizen (No: 0, Yes: 1)', 0, 1),
    'Partner': st.sidebar.slider('Partner (No: 0, Yes: 1)', 0, 1),
    'Dependents': st.sidebar.slider('Dependents (No: 0, Yes: 1)', 0, 1),
    'tenure': st.sidebar.slider('Tenure', min_value=0, max_value=100, value=0),
    'PhoneService': st.sidebar.slider('Phone Service (No: 0, Yes: 1)', 0, 1),
    'MultipleLines': st.sidebar.slider('Multiple Lines (No: 0, Yes: 1)', 0, 1),
    'InternetService': st.sidebar.slider('Internet Service (No: 0, Yes: 1)', 0, 1),
    'OnlineSecurity': st.sidebar.slider('Online Security (No: 0, Yes: 1)', 0, 1),
    'OnlineBackup': st.sidebar.slider('Online Backup (No: 0, Yes: 1)', 0, 1),
    'DeviceProtection': st.sidebar.slider('Device Protection (No: 0, Yes: 1)', 0, 1),
    'TechSupport': st.sidebar.slider('Tech Support (No: 0, Yes: 1)', 0, 1),
    'StreamingTV': st.sidebar.slider('Streaming TV (No: 0, Yes: 1)', 0, 1),
    'StreamingMovies': st.sidebar.slider('Streaming Movies (No: 0, Yes: 1)', 0, 1),
    'Contract': st.sidebar.slider('Contract (No: 0, Yes: 1)', 0, 1),
    'PaperlessBilling': st.sidebar.slider('Paperless Billing (No: 0, Yes: 1)', 0, 1),
    'PaymentMethod': st.sidebar.slider('Payment Method (No: 0, Yes: 1)', 0, 1),
    'MonthlyCharges': st.sidebar.slider('Monthly Charges', min_value=0.0, max_value=200.0, value=0.0),
    'TotalCharges': st.sidebar.slider('Total Charges', min_value=0.0, max_value=8000.0, value=0.0),
}

# Display first few rows of the dataset
st.title('Telco Customer Churn Prediction')
st.subheader('Data Overview')
st.write(data.head())

# Display general information about the dataset
st.subheader('Data Overview')
st.write(data.info())

# Display title for Data Preprocessing section
st.subheader('Data Preprocessing')

# Preprocessing
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].replace(' ', np.nan, inplace=True)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data['TotalCharges'].fillna(0, inplace=True)

# Encoding categorical variables
cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in cols:
    encoder = OneHotEncoder(sparse=False)
    encoded_col = encoder.fit_transform(data[[col]])
    data[col] = encoded_col

cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
for col in cols:
    encoder = LabelEncoder()
    encoded_col = encoder.fit_transform(data[col])
    data[col] = encoded_col

# Train Test Split
data_x = data[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
data_y = data['Churn']

x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=5)

# Scaling Features
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(x_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
x_train[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaled_train

scaled_test = scaler.transform(x_test[['tenure', 'MonthlyCharges', 'TotalCharges']])
x_test[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaled_test

# Oversampling
sm = SMOTE(sampling_strategy=0.8, k_neighbors=5, random_state=5)
x_resample, y_resample = sm.fit_resample(x_train, y_train)

sm = SMOTE(sampling_strategy=0.66, k_neighbors=5, random_state=5)
x_resample_2, y_resample_2 = sm.fit_resample(x_train, y_train)

# Random Forest Classification without oversample
rf = RandomForestClassifier(random_state=5, criterion='entropy', n_estimators=18, max_depth=12)
rf.fit(x_train, y_train)
prediction = rf.predict(x_test)
st.subheader('Random Forest Classification without Oversampling')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, prediction))
st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction)*100))
st.write("Recall: {:.2f}%".format(recall_score(y_test, prediction)*100))
st.write("Precision: {:.2f}%".format(precision_score(y_test, prediction)*100))

# Random Forest Classification Oversample 55:45
rf.fit(x_resample, y_resample)
prediction = rf.predict(x_test)
st.subheader('Random Forest Classification Oversample 55:45')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, prediction))
st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction)*100))
st.write("Recall: {:.2f}%".format(recall_score(y_test, prediction)*100))
st.write("Precision: {:.2f}%".format(precision_score(y_test, prediction)*100))

# Random Forest Classification Oversample 60:40
rf.fit(x_resample_2, y_resample_2)
prediction = rf.predict(x_test)
st.subheader('Random Forest Classification Oversample 60:40')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, prediction))
st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction)*100))
st.write("Recall: {:.2f}%".format(recall_score(y_test, prediction)*100))
st.write("Precision: {:.2f}%".format(precision_score(y_test, prediction)*100))

# Prediction
st.sidebar.subheader('Make Predictions')
if st.sidebar.button('Predict'):
    example_df = pd.DataFrame(example, index=[0])
    pred = rf.predict(example_df.values)
    if pred[0] == 0:
        st.sidebar.write("No, the customer will not churn according to the predictions.")
    else:
        st.sidebar.write("Yes, the customer will churn according to the predictions.")
