import streamlit as st
import numpy as np
from keras.models import load_model
import pickle

# Load the pre-trained model
model = load_model('Credit_Card_Fraud_Model.h5')

# Load the pre-fitted StandardScaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Credit Card Fraud Detection System')
st.markdown('Predict the likelihood of a transaction being fraudulent based on input features.')

# Input Section
st.header('Customer Details')

# Customer ID Input
customer_id = st.text_input(
    'Customer ID:',
    placeholder='Enter Customer ID (e.g., 12345)',
    help='Customer ID is used for reference purposes.'
)

# Feature Inputs (A1 to A14)
features = []
st.markdown('### Transaction Features (A1 to A14)')
for i in range(1, 15):
    feature = st.number_input(
        f'Feature {i} (A{i}):',
        step=0.1,
        format="%.4f",
        help=f'Enter the value for feature A{i}.'
    )
    features.append(feature)

# Input for Class (Fraud Label) - Adding Class here
input_class = st.selectbox(
    'Class (Fraud Label):',
    options=[0, 1],
    help='Enter the true class label (0 for Not Fraudulent, 1 for Fraudulent).'
)

# Add the Class feature to the input features list
features_with_class = features + [input_class]

# Prediction Section
if st.button('Check Fraud'):
    try:
        # Ensure all features are entered
        if all(feature is not None for feature in features) and len(features) == 14:
            # Rescale the input features
            scaled_features = scaler.transform([features_with_class])

            # Make prediction using the trained model
            prediction = model.predict(scaled_features)

            # Interpret the result
            fraud_probability = prediction[0][0]
            predicted_class = 1 if fraud_probability > 0.5 else 0

            # Display the result
            st.subheader(f'Result for Customer ID {customer_id}')
            st.write(f'Transaction is **{"Fraudulent" if predicted_class == 1 else "Not Fraudulent"}**')
            st.write(f'Fraud Probability: **{fraud_probability:.2%}**')

            # Add fraud risk commentary
            if fraud_probability > 0.8:
                st.warning("High risk of fraud detected. Immediate action recommended.")
            elif fraud_probability > 0.5:
                st.info("Moderate risk of fraud. Further investigation advised.")
            else:
                st.success("Transaction is likely safe.")

            # Show comparison with the input class
            if input_class == predicted_class:
                st.success(f"The predicted fraud status matches the input class ({'Fraudulent' if input_class == 1 else 'Not Fraudulent'}).")
            else:
                st.error(f"The predicted fraud status does not match the input class. Predicted: {'Fraudulent' if predicted_class == 1 else 'Not Fraudulent'}, Input: {'Fraudulent' if input_class == 1 else 'Not Fraudulent'}")
        else:
            st.warning('Please input all 14 features.')

    except Exception as e:
        st.error(f"An error occurred while processing the input: {e}")

# Footer
st.markdown('---')
st.markdown('© 2024 Credit Card Fraud Detection System')
