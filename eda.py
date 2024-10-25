import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Function for EDA
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    
    # Load Data
    data = pd.read_csv('telecomchurn.csv')
    
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


    
    # Plot Churn by Senior Citizen
    st.subheader("Churn by Senior Citizen")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='SeniorCitizen', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Senior Citizen")
    plt.xlabel("Senior Citizen")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Plot Churn by Partner
    st.subheader("Churn by Partner")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='Partner', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Partner")
    plt.xlabel("Partner")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Plot Churn by Dependents
    st.subheader("Churn by Dependents")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='Dependents', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Dependents")
    plt.xlabel("Dependents")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Plot Churn by Phone Service
    st.subheader("Churn by Phone Service")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='PhoneService', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Phone Service")
    plt.xlabel("Phone Service")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Plot Churn by Multiple Lines
    st.subheader("Churn by Multiple Lines")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='MultipleLines', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Multiple Lines")
    plt.xlabel("Multiple Lines")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Plot Churn by Internet Service
    st.subheader("Churn by Internet Service")
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='lightyellow')
    sns.countplot(data=data, x='InternetService', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title("Churn by Internet Service")
    plt.xlabel("Internet Service")
    plt.ylabel("Count")
    st.pyplot(fig)
    
    # Continue plotting other features...                                     
              # Plot Churn by Paperless Billing
    st.subheader("Churn by Paperless Billing")
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='lightyellow')
    sns.countplot(data=data, x='PaperlessBilling', hue='Churn', saturation=1, alpha=0.9, palette='rocket')
    plt.title('Churn by Paperless Billing')
    plt.xlabel('Paperless Billing')
    plt.ylabel('Count')
    for p in ax.patches:
       ax.annotate(f'\n{p.get_height()}', (p.get_x() + 0.2, p.get_height()), ha='center', va='top', color='white', size=13)
       st.pyplot(fig)

#

# Plot Churn Proportion
    st.subheader("Churn Proportion")
    fig, ax = plt.subplots(figsize=(8, 7), facecolor='lightyellow')
    plt.pie(data['Churn'].value_counts(), autopct='%.2f%%', pctdistance=1.25, startangle=45,
        textprops={'fontsize': 15}, colors=['indigo', 'darkorange'], shadow=True)
    my_circle = plt.Circle((0, 0), 0.6, color='lightyellow')
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.title('Churn Proportion', fontsize=17, fontweight='bold')
    plt.legend(['No', 'Yes'], bbox_to_anchor=(1, 1), fontsize=12)
    st.pyplot(fig)

# Main function to run the Streamlit app
def main():
    exploratory_data_analysis()

if __name__ == "__main__":
    main()