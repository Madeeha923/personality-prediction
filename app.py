import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load Saved Files ---
try:
    with open('personality_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    with open('unique_values.pkl', 'rb') as f:
        unique_values = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please run the `save_model.py` script first.")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="",
    layout="centered"
)


# --- User Interface ---
st.title("Personality Prediction")
st.write("Answer the following questions to predict whether you are an Introvert or Extrovert.")

# Define which columns to show in the UI
columns_to_show = ['Time_spent_Alone', 'Stage_fear', 'Drained_after_socializing', 'Going_outside']

# Create a dictionary to hold user inputs
user_input = {}

# Create input fields for ONLY the selected feature columns
for col in columns_to_show:
    question_label = f"How would you rate your '{col.replace('_', ' ').title()}'?"
    
    # Explicitly define which questions are binary (Yes/No)
    if col in ['Stage_fear', 'Drained_after_socializing']:
        user_input[col] = st.selectbox(col,
            
            options=['No', 'Yes']
        )
    # Use sliders for numerical ranges
    elif col in ['Time_spent_Alone', 'Going_outside']:
        options = [float(opt) for opt in unique_values[col] if str(opt).lower() != 'nan']
        if options: # Ensure options list is not empty
            min_val = min(options)
            max_val = max(options)
            # Default slider to the median value of the options
            default_val = int(np.median(options)) 
            user_input[col] = st.slider(
                question_label,
                min_value=int(min_val),
                max_value=int(max_val),
                value=default_val
            )

# --- Prediction Logic ---
if st.button("Predict Personality"):
    # Initialize the dictionary for the model with all required columns
    processed_input = {}

    # 1. Process the user-provided inputs
    for key, value in user_input.items():
        if value == 'Yes':
            processed_input[key] = 1
        elif value == 'No':
            processed_input[key] = 0
        else:
            try:
                # This handles both slider (int) and potential selectbox (str) inputs
                processed_input[key] = float(value)
            except ValueError:
                st.error(f"Invalid input for {key}. Please select a valid option.")
                st.stop()
    
    # 2. Add default values for the columns that were not asked
    for col in model_columns:
        if col not in columns_to_show:
            options = unique_values.get(col, [])
            if len(options) > 2: # It's a numerical-like column, so use the median as a default
                # Filter out 'nan' before calculating median
                numeric_options = [float(v) for v in options if str(v).lower() != 'nan']
                if numeric_options: # Ensure list is not empty after filtering
                    default_value = np.median(numeric_options)
                    processed_input[col] = default_value
                else: # Fallback if all options were 'nan'
                    processed_input[col] = 0 
            else: # It's a binary-like column, default to 0 (No) as it's often the mode
                processed_input[col] = 0


    # Create a DataFrame from the user's input
    input_df = pd.DataFrame([processed_input])

    # Ensure the columns are in the same order as the model was trained on
    input_df = input_df[model_columns]

    # Make a prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)

    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"You are likely an **Extrovert**!")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.info(f"You are likely an **Introvert**!")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")

