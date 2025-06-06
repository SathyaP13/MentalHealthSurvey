# bias_evaluation.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split # Needed for splitting data for evaluation

print("Starting bias evaluation...")

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'model'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv') 
PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl')
IMPUTERS_FILE = os.path.join(MODEL_DIR, 'imputers.pkl') # For Streamlit's dummy data logic
FEATURE_INFO_FILE = os.path.join(MODEL_DIR, 'feature_info.pkl')
MODEL_FILE = os.path.join(MODEL_DIR, 'dnn_depression_model.pth')

# Hyperparameters for the model 
HIDDEN_LAYER_SIZES = [128, 64, 32, 16, 8, 4, 2] 

# --- Define the Neural Network Model 
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

# --- Load Assets ---
try:
    # Load the full preprocessor pipeline
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    cat_imputer, num_imputer = joblib.load(IMPUTERS_FILE) # Load separate imputers
    feature_info = joblib.load(FEATURE_INFO_FILE)

    numerical_cols_trained = feature_info['numerical_features']
    categorical_cols_trained = feature_info['categorical_features']
    final_feature_columns = feature_info['final_feature_columns'] # All columns after OHE
    loaded_categorical_options = feature_info.get('categorical_options', {})

    # Determine input features size for the model
    # Recreate a dummy DataFrame similar to how it's done in Streamlit
    dummy_data_for_shape = {}
    for col in numerical_cols_trained:
        dummy_data_for_shape[col] = 0.0
    for col in categorical_cols_trained:
        if col in loaded_categorical_options and loaded_categorical_options[col]:
            dummy_data_for_shape[col] = loaded_categorical_options[col][0]
        else:
            dummy_data_for_shape[col] = 'DefaultCategory' # Fallback

    dummy_df = pd.DataFrame([dummy_data_for_shape])
    
    # Ensure dummy_df columns match the order preprocessor expects for initial steps
    dummy_df = dummy_df[numerical_cols_trained + categorical_cols_trained]

    # Transform dummy data using the full preprocessor
    processed_dummy = preprocessor.transform(dummy_df)
    if hasattr(processed_dummy, 'toarray'):
        processed_dummy = processed_dummy.toarray()

    input_features_size = processed_dummy.shape[1]

    # Instantiate the model with the correct class name and parameters
    model = DepressionPredictor(input_features_size, HIDDEN_LAYER_SIZES)
    model.load_state_dict(torch.load(MODEL_FILE, map_location=torch.device('cpu')))
    model.eval()
    print("Preprocessor and Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading assets: {e}. Ensure all required model files are in the '{MODEL_DIR}' directory and '{TRAIN_FILE}' exists.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during asset loading: {e}")
    exit()

# --- Load Original Training Data to get Validation Set with Sensitive Attributes ---
try:
    full_train_df = pd.read_csv(TRAIN_FILE)
    TARGET_COLUMN = 'Depression'
    
    # These columns were dropped in modeltraining.py, so exclude them from X_full
    columns_dropped_in_training = ['id', 'Name', 'City', 'Profession', 'Degree']
    X_full = full_train_df.drop(columns=[col for col in columns_dropped_in_training if col in full_train_df.columns] + [TARGET_COLUMN], errors='ignore')
    y_full = full_train_df[TARGET_COLUMN].astype(int)

    # Split to get the same validation set as in modeltraining.py
    _, X_val_original, _, y_val_original = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print("Original validation data loaded.")
except FileNotFoundError as e:
    print(f"Error loading original train.csv: {e}. Make sure '{TRAIN_FILE}' exists.")
    exit()
except Exception as e:
    print(f"An error occurred during original data loading or splitting: {e}")
    exit()

# --- Identify Sensitive Attributes for Bias Evaluation ---
# Ensure these columns exist in X_val_original
SENSITIVE_ATTRIBUTES = ['Gender', 'Age', 'Working Professional or Student'] 

print(f"\n--- Bias Evaluation for Sensitive Attributes: {SENSITIVE_ATTRIBUTES} ---")

# --- Helper function to evaluate subgroup performance ---
def evaluate_subgroup(group_df, group_labels, model, preprocessor, numerical_features, categorical_features, final_feature_columns):
    if group_df.empty:
        return {'accuracy': np.nan, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan, 'count': 0}

    cols_expected_by_preprocessor = numerical_features + categorical_features
    for col in cols_expected_by_preprocessor:
        if col not in group_df.columns:
            if col in numerical_features:
                group_df[col] = 0.0 # Default numerical value
            else:
                group_df[col] = 'Not_Provided' # Default categorical value (will be handled by OHE's handle_unknown)

    # Transform the subgroup data using the full preprocessor pipeline
    processed_group_data = preprocessor.transform(group_df[cols_expected_by_preprocessor])
    
    # Convert sparse array to dense if applicable
    if hasattr(processed_group_data, 'toarray'):
        processed_group_data = processed_group_data.toarray()

    input_tensor = torch.tensor(processed_group_data, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor) # This output is already a probability (0-1) due to Sigmoid
        predictions = (outputs > 0.5).int().cpu().numpy().flatten() # Binary predictions

    true_labels = group_labels.values

    acc = accuracy_score(true_labels, predictions)
    prec = precision_score(true_labels, predictions, zero_division=0)
    rec = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'count': len(true_labels)
    }

# --- Perform Bias Evaluation for each sensitive attribute ---
bias_results = {}

for attr in SENSITIVE_ATTRIBUTES:
    if attr not in X_val_original.columns:
        print(f"Warning: Sensitive attribute '{attr}' not found in validation data (after initial column drops). Skipping.")
        continue

    print(f"\nEvaluating for '{attr}':")
    bias_results[attr] = {}
    unique_groups = X_val_original[attr].unique()

    for group_value in unique_groups:
        if pd.isna(group_value):
            group_name = "NaN"
            subgroup_X = X_val_original[X_val_original[attr].isna()]
            subgroup_y = y_val_original[X_val_original[attr].isna()]
        else:
            group_name = str(group_value)
            subgroup_X = X_val_original[X_val_original[attr] == group_value]
            subgroup_y = y_val_original[X_val_original[attr] == group_value]

        print(f"  Group: '{group_name}'")
        
        metrics = evaluate_subgroup(
            subgroup_X.copy(), # .copy() to avoid SettingWithCopyWarning
            subgroup_y,
            model,
            preprocessor,
            numerical_cols_trained,
            categorical_cols_trained,
            final_feature_columns
        )
        bias_results[attr][group_name] = metrics

        print(f"    Count: {metrics['count']}, Accuracy: {metrics['accuracy']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
              f"F1-Score: {metrics['f1']:.4f}")

        if metrics['count'] == 0:
            print(f"    Warning: No data for group '{group_name}'. Metrics are NaN.")


# --- Interpret Results ---
print("\n--- Bias Evaluation Summary ---")
for attr, groups_data in bias_results.items():
    print(f"\nAttribute: {attr}")
    metrics_df = pd.DataFrame(groups_data).T
    print(metrics_df)

    f1_scores = metrics_df['f1'].dropna()
    if not f1_scores.empty:
        f1_min = f1_scores.min()
        f1_max = f1_scores.max()
        f1_diff = f1_max - f1_min
        print(f"  F1-Score Range: {f1_min:.4f} to {f1_max:.4f} (Difference: {f1_diff:.4f})")
        if f1_diff > 0.1:
            print(f"  **Potential bias detected: F1-score varies significantly across groups.**")
        else:
            print(f"  F1-score seems relatively consistent across groups.")
    else:
        print("  No F1-scores available for comparison.")

    recall_scores = metrics_df['recall'].dropna()
    if not recall_scores.empty:
        recall_min = recall_scores.min()
        recall_max = recall_scores.max()
        recall_diff = recall_max - recall_min
        print(f"  Recall Range: {recall_min:.4f} to {recall_max:.4f} (Difference: {recall_diff:.4f})")
        if recall_diff > 0.1:
            print(f"  **Potential bias detected: Recall (Equal Opportunity) varies significantly across groups.**")
        else:
            print(f"  Recall seems relatively consistent across groups.")
    else:
        print("  No Recall scores available for comparison.")

print("\nBias evaluation complete. Review the metrics above for disparities.")
print("If significant disparities are found, consider data augmentation, re-sampling, or using fairness-aware algorithms during training.")