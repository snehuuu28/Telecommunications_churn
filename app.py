import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# Set page configuration to use full width
st.set_page_config(layout="wide")

# Load the pre-trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Page title
st.title("📊 Customer Churn Prediction")

# Create a three-column layout (Left: Features | Middle: Features | Right: Prediction)
left_col, mid_col, right_col = st.columns([1.5, 1.5, 1])

# Left Side: Feature Inputs (First Half)
with left_col:
    st.subheader("📋 Customer & Account Details")
    account_length = st.slider("Account Length (months)", 1, 500, 100)
    voice_mail_plan = st.selectbox("Voice Mail Plan", [0, 1])
    voice_mail_messages = st.slider("Voice Mail Messages", 0, 100, 5)
    international_plan = st.selectbox("International Plan", [0, 1])
    customer_service_calls = st.slider("Customer Service Calls", 0, 10, 2)
    international_mins = st.slider("International Minutes", 0.0, 100.0, 5.0)
    international_calls = st.slider("International Calls", 0, 20, 3)
    international_charge = st.slider("International Charge", 0.0, 50.0, 5.0)

# Middle Section: Feature Inputs (Second Half)
with mid_col:
    st.subheader("📋 Call & Charge Details")
    day_mins = st.slider("Day Minutes", 0.0, 500.0, 100.0)
    day_calls = st.slider("Day Calls", 0, 200, 50)
    day_charge = st.slider("Day Charge", 0.0, 100.0, 25.0)
    evening_mins = st.slider("Evening Minutes", 0.0, 500.0, 50.0)
    evening_calls = st.slider("Evening Calls", 0, 200, 30)
    evening_charge = st.slider("Evening Charge", 0.0, 100.0, 15.0)
    night_mins = st.slider("Night Minutes", 0.0, 500.0, 20.0)
    night_calls = st.slider("Night Calls", 0, 200, 10)
    night_charge = st.slider("Night Charge", 0.0, 50.0, 5.0)
    total_charge = st.slider("Total Charge", 0.0, 500.0, 100.0)

# Prepare input data for prediction
user_input = np.array([
    account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins,
    international_mins, customer_service_calls, international_plan, day_calls, day_charge, 
    evening_calls, evening_charge, night_calls, night_charge, international_calls, international_charge, 
    total_charge
]).reshape(1, -1)

# Right Side: Prediction & Visualization
with right_col:
    st.subheader("🔮 Prediction")

    # Prediction button
    predict_button = st.button("🚀 Predict Churn", key="predict_button")

    if predict_button:
        with st.spinner("🔄 Analyzing customer data..."):
            time.sleep(2)  # Simulating prediction time
            prediction = model.predict(user_input)

        # Display prediction output
        if prediction == 1:
            st.error("🚨 **The customer is likely to churn!**")
        else:
            st.success("🎉 **The customer is unlikely to churn!**")

        # DataFrame for Visualization
        input_data_df = pd.DataFrame({
            'Feature': [
                'Account Length', 'Voice Mail Plan', 'Voice Mail Messages', 'Day Minutes', 'Evening Minutes', 'Night Minutes', 
                'International Minutes', 'Customer Service Calls', 'International Plan', 'Day Calls', 'Day Charge', 
                'Evening Calls', 'Evening Charge', 'Night Calls', 'Night Charge', 'International Calls', 'International Charge', 
                'Total Charge'
            ],
            'Value': [
                account_length, voice_mail_plan, voice_mail_messages, day_mins, evening_mins, night_mins, international_mins, 
                customer_service_calls, international_plan, day_calls, day_charge, evening_calls, evening_charge, night_calls, 
                night_charge, international_calls, international_charge, total_charge
            ]
        })

        # Bar chart visualization
        st.subheader("📊 Customer Data Visualization")
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(input_data_df['Feature'], input_data_df['Value'], color=plt.cm.viridis(np.linspace(0, 1, len(input_data_df))))
        ax.set_xlabel('Values')
        ax.set_title('Customer Input Data')

        # Add labels for each bar
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 3, bar.get_y() + bar.get_height()/2, f'{width:.1f}', va='center', color='black', fontweight='bold')

        st.pyplot(fig)

        # CSV Download Section
        st.subheader("📥 Download Prediction Data")
        
        # Add prediction result to the DataFrame
        prediction_result = 'Churn Prediction: ' + ('Likely to Churn' if prediction == 1 else 'Unlikely to Churn')
        input_data_df['Prediction'] = prediction_result

        # Convert to CSV
        csv = input_data_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

# Extra Information
st.markdown("---")
st.subheader("ℹ️ About This Tool")
st.info("🔹 This tool predicts whether a customer is likely to churn based on various call and account details.")
st.info("📌 The prediction is based on historical customer data and a trained machine learning model.")
st.info("📊 The visualization helps understand the input data and compare feature values.")
