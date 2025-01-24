import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load reference data (Airline, Source, Destination)
reference_data = pickle.load(open("data.pkl", "rb"))

# Define feature columns based on training
feature_columns = pickle.load(open("feature_columns.pkl", "rb"))

# Set the title and header with an image
st.image("images.jpg", use_container_width=True)  # Replace with your banner image path
st.title("✈️ Airline Price Predictor")
st.markdown("#### Predict the price of your flight based on various factors")
st.markdown("---")

# Load the scaler and model
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("Best_Model.pkl", "rb"))

# Sidebar for user inputs
st.sidebar.header("Enter Flight Details")
st.sidebar.markdown("Provide flight information to get the predicted price.")

# Ensure the required columns exist in reference_data
if not all(col in reference_data.columns for col in ["Airline", "Source", "Destination"]):
    st.error("Reference data does not contain the expected 'Airline', 'Source', or 'Destination' columns.")
else:
    # User inputs
    airline = st.sidebar.selectbox("✈️ Select Airline", reference_data["Airline"].unique())
    source = st.sidebar.selectbox("📍 Select Source", reference_data["Source"].unique())
    destination = st.sidebar.selectbox("📍 Select Destination", reference_data["Destination"].unique())
    total_stops = st.sidebar.number_input("🔄 Total Stops", min_value=0, max_value=4, value=0)
    journey_day = st.sidebar.number_input("📅 Journey Day", min_value=1, max_value=31, value=1)
    journey_month = st.sidebar.number_input("📅 Journey Month", min_value=1, max_value=12, value=1)
    dep_hour = st.sidebar.number_input("🕒 Departure Hour", min_value=0, max_value=23, value=0)
    dep_min = st.sidebar.number_input("🕒 Departure Minute", min_value=0, max_value=59, value=0)
    arrival_hour = st.sidebar.number_input("🕒 Arrival Hour", min_value=0, max_value=23, value=0)
    arrival_min = st.sidebar.number_input("🕒 Arrival Minute", min_value=0, max_value=59, value=0)
    total_duration_minutes = st.sidebar.number_input("⏱ Total Duration (minutes)", min_value=0, value=0)

    # Add a stylish "Predict" button
    predict_button = st.sidebar.button("🚀 Predict Flight Price")

    # Prediction logic
    if predict_button:
        with st.spinner("Predicting the flight price..."):
            # Create input DataFrame
            input_data = pd.DataFrame({
                "Total_Stops": [total_stops],
                "Journey_day": [journey_day],
                "Journey_month": [journey_month],
                "Dep_hour": [dep_hour],
                "Dep_min": [dep_min],
                "Arrival_hour": [arrival_hour],
                "Arrival_min": [arrival_min],
                "Total_Duration_minutes": [total_duration_minutes],
                "Airline": [airline],
                "Source": [source],
                "Destination": [destination],
            })

            # Apply one-hot encoding to categorical columns using get_dummies
            input_encoded = pd.get_dummies(input_data, columns=["Airline", "Source", "Destination"])

            # Align input with feature columns used during training
            input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

            try:
                # Scale the input data
                scaled_input = scaler.transform(input_encoded)

                # Convert the scaled input to DMatrix
                dmatrix_input = xgb.DMatrix(scaled_input)

                # Predict the log-transformed price
                predicted_log_price = model.predict(dmatrix_input)
                predicted_price = np.exp(predicted_log_price[0])  # Reverse log transformation

                # Display the predicted price
                st.success(f"💰 The predicted flight price is ₹{predicted_price:,.2f}")
            except ValueError as e:
                st.error(f"❌ Error in input data: {e}")

# Add footer with a message
st.markdown("---")
st.markdown("🔧 Made with ❤️ by [Prajwal Patel]")
