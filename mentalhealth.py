import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import os

# --- Configuration ---
MODEL_DIR = 'model'
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl') # Full preprocessing pipeline
IMPUTERS_FILE = os.path.join(MODEL_DIR, 'imputers.pkl') # Separate imputers
FEATURE_INFO_FILE = os.path.join(MODEL_DIR, 'feature_info.pkl') # Contains numerical/categorical features and options
MODEL_FILE = os.path.join(MODEL_DIR, 'dnn_depression_model.pth')

# Hyperparameters for the model
HIDDEN_LAYER_SIZES = [128, 64, 32, 16, 8, 4, 2] 

# --- Define model architecture #
class DepressionPredictor(nn.Module):
    def __init__(self, input_features, hidden_sizes):
        super(DepressionPredictor, self).__init__()
        layers = []
        in_size = input_features
        for h_size in hidden_sizes:
            layers.append(nn.Linear(in_size, h_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_size))
            layers.append(nn.Dropout(0.3))
            in_size = h_size

        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Load model components ---
@st.cache_resource # Cache the model loading for better performance in Streamlit
def load_model_components():
    try:
        preprocessor_pipeline = joblib.load(PREPROCESSOR_FILE)
        cat_imputer, num_imputer = joblib.load(IMPUTERS_FILE)
        feature_info = joblib.load(FEATURE_INFO_FILE)

        numerical_cols_trained = feature_info['numerical_features']
        categorical_cols_trained = feature_info['categorical_features']
        final_feature_columns = feature_info['final_feature_columns']
        loaded_categorical_options = feature_info.get('categorical_options', {})

        # Determine input size for the model dynamically (recreate dummy data)
        dummy_data_for_shape = {}
        for col in numerical_cols_trained:
            dummy_data_for_shape[col] = 0.0
        for col in categorical_cols_trained:
            if col in loaded_categorical_options and loaded_categorical_options[col]:
                dummy_data_for_shape[col] = loaded_categorical_options[col][0]
            else:
                dummy_data_for_shape[col] = 'DefaultCategory'

        dummy_df = pd.DataFrame([dummy_data_for_shape])
        # Ensure dummy_df has the columns expected by the preprocessor's transformers
        dummy_df_ordered = dummy_df[numerical_cols_trained + categorical_cols_trained]

        processed_dummy = preprocessor_pipeline.transform(dummy_df_ordered)
        if hasattr(processed_dummy, 'toarray'):
            processed_dummy = processed_dummy.toarray()
        
        input_features_size = processed_dummy.shape[1]

        model = DepressionPredictor(input_features_size, HIDDEN_LAYER_SIZES)
        model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device("cpu")))
        model.eval()

        return (model, preprocessor_pipeline, cat_imputer, num_imputer,
                numerical_cols_trained, categorical_cols_trained,
                final_feature_columns, loaded_categorical_options)

    except Exception as e:
        st.error(f"Error loading model components or preprocessing info: {e}. "
                 f"Ensure model files are in the '{MODEL_DIR}' directory and are compatible.")
        st.stop() # Stop the app if components can't be loaded

model, preprocessor_pipeline, cat_imputer, num_imputer, numerical_cols_trained, \
    categorical_cols_trained, final_feature_columns, loaded_categorical_options = load_model_components()


# --- UI inputs ---
st.title("Depression Prediction App")

# Dynamically populate selectboxes using loaded options
gender = st.selectbox("Gender", loaded_categorical_options.get('Gender', ["Male", "Female"]))
age = st.number_input("Age", min_value=10, max_value=100, step=1)
role = st.selectbox("Working Professional or Student", loaded_categorical_options.get('Working Professional or Student', ["Working Professional", "Student"]))
academic_pressure = st.number_input("Academic Pressure", min_value=0.0, max_value=10.0, step=0.1)
work_pressure = st.number_input("Work Pressure", min_value=0.0, max_value=10.0, step=0.1)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
study_satisfaction = st.number_input("Study Satisfaction", min_value=0.0, max_value=10.0, step=0.1)
job_satisfaction = st.number_input("Job Satisfaction", min_value=0.0, max_value=10.0, step=0.1)
sleep_duration = st.selectbox("Sleep Duration", loaded_categorical_options.get('Sleep Duration', ["Less than 5 hours", "7-8 hours", "More than 8 hours"]))
diet = st.selectbox("Dietary Habits", loaded_categorical_options.get('Dietary Habits', ["Moderate", "Healthy", "Unhealthy"]))
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts ?", loaded_categorical_options.get('Have you ever had suicidal thoughts ?', ["Yes", "No"]))
work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, step=0.5)
financial_stress = st.number_input("Financial Stress", min_value=0.0, max_value=10.0, step=0.1)
family_history = st.selectbox("Family History of Mental Illness", loaded_categorical_options.get('Family History of Mental Illness', ["Yes", "No"]))

# On Predict
if st.button("Predict"):
    input_dict = {
        'Gender': gender,
        'Age': float(age),
        'Working Professional or Student': role,
        'Academic Pressure': float(academic_pressure),
        'Work Pressure': float(work_pressure),
        'CGPA': float(cgpa),
        'Study Satisfaction': float(study_satisfaction),
        'Job Satisfaction': float(job_satisfaction),
        'Sleep Duration': sleep_duration,
        'Dietary Habits': diet,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Work/Study Hours': float(work_study_hours),
        'Financial Stress': float(financial_stress),
        'Family History of Mental Illness': family_history
    }

    input_df = pd.DataFrame([input_dict])

    # Ensure all numerical inputs are numeric, and handle potential missing values
    for col in numerical_cols_trained:
        if col not in input_df.columns:
            input_df[col] = 0.0 # Default value for numerical

    # Prepare input_df for the preprocessor_pipeline
    # The ColumnTransformer (preprocessor_pipeline) expects specific columns as input
    input_df_for_transform = input_df[numerical_cols_trained + categorical_cols_trained]

    # Apply the full preprocessing pipeline (imputation, encoding, scaling)
    processed_input = preprocessor_pipeline.transform(input_df_for_transform)

    if hasattr(processed_input, 'toarray'):
        processed_input = processed_input.toarray()
    
    # Convert to tensor
    input_tensor = torch.tensor(processed_input, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        output_raw = model(input_tensor) # Model's last layer is Sigmoid, so this is already a probability
        prediction_prob = output_raw.item() # Get the single probability value

    # Convert probability to binary prediction using 0.5 threshold
    pred_class = 1 if prediction_prob > 0.5 else 0

    st.markdown(f"### Prediction: {'Depressed' if pred_class == 1 else 'Not Depressed'}")
    st.markdown(f"*(Confidence: {prediction_prob:.2f})*")