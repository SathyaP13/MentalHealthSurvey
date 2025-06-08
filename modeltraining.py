import pandas as pd
import torch
from torch import nn, optim
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os

print("Starting model training and artifact generation...")

# --- Configuration ---
DATA_DIR = 'data'
MODEL_DIR = 'model'
TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl') # Full preprocessing pipeline
IMPUTERS_FILE = os.path.join(MODEL_DIR, 'imputers.pkl') # Separate imputers for specific use cases
FEATURE_INFO_FILE = os.path.join(MODEL_DIR, 'feature_info.pkl') # Stores column names, options, etc.
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'dnn_depression_model.pth')
TEST_PREDICTIONS_FILE = 'test_predictions.csv' # Saved at project root

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Define the Neural Network Model ---
# This class must be IDENTICAL across modeltraining.py, biasevaluation.py, and mentalhealth.py
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
        layers.append(nn.Sigmoid()) # Output a probability between 0 and 1

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# --- Load Training Dataset ---
try:
    df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded training data from {TRAIN_FILE}")
except FileNotFoundError:
    print(f"Error: {TRAIN_FILE} not found. Please ensure the '{DATA_DIR}' folder exists and contains 'train.csv'.")
    exit()

# Clean column names
df.columns = df.columns.str.strip()

# Store original categorical column names and their unique options before dropping irrelevant columns
# This is for feature_info.pkl and Streamlit selectboxes
original_categorical_cols_for_options = df.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_options_map = {
    col: df[col].dropna().unique().tolist()
    for col in original_categorical_cols_for_options
    if col in df.columns # Ensure column exists
}


# Drop irrelevant columns 
columns_to_drop = ['id', 'Name', 'City', 'Profession', 'Degree']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

# Define target column
target_col = 'Depression'

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

# Identify actual categorical and numerical columns in X *after* dropping
current_categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
current_numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- Create Preprocessing Pipelines ---
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, current_numerical_cols),
        ('cat', categorical_transformer, current_categorical_cols)
    ],
    remainder='passthrough'
)

# Fit preprocessor on X and transform
X_transformed = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding for consistent column order
ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(current_categorical_cols)
final_feature_columns_list = current_numerical_cols + ohe_feature_names.tolist()

# Convert sparse array to dense if it is sparse (from OneHotEncoder)
if hasattr(X_transformed, 'toarray'):
    X_transformed = X_transformed.toarray()

# Convert to DataFrame to ensure consistent column order and names for split
X_processed_df = pd.DataFrame(X_transformed, columns=final_feature_columns_list)

# Train-validation split
X_train_processed, X_val_processed, y_train, y_val = train_test_split(
    X_processed_df, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_processed.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) # Unsqueeze for BCELoss
X_val_tensor = torch.tensor(X_val_processed.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1) # Unsqueeze for BCELoss


# --- Initialize model, loss function, and optimizer ---
input_size = X_train_processed.shape[1]
hidden_size_layers = [128, 64, 32, 16, 8, 4, 2] # Hidden layer sizes for DepressionPredictor

model = DepressionPredictor(input_size, hidden_size_layers)
criterion = nn.BCELoss() # Use BCELoss because model outputs are sigmoid (probabilities)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
num_epochs = 1000
best_val_f1 = -1.0 # Initialize for saving best model

print("\nStarting model training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

            # Convert probabilities to binary predictions for metrics (threshold at 0.5)
            val_preds = (val_outputs > 0.5).int().cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()

            val_acc = accuracy_score(y_val_np, val_preds)
            val_prec = precision_score(y_val_np, val_preds, zero_division=0)
            val_rec = recall_score(y_val_np, val_preds, zero_division=0)
            val_f1 = f1_score(y_val_np, val_preds, zero_division=0)

            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {loss.item():.4f} - "
                  f"Val Loss: {val_loss.item():.4f} - "
                  f"Val Acc: {val_acc:.4f} - "
                  f"Val Precision: {val_prec:.4f} - "
                  f"Val Recall: {val_rec:.4f} - "
                  f"Val F1: {val_f1:.4f}")

            # Save the best model based on F1-score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"  --> Model state saved to {MODEL_SAVE_PATH}! New best F1: {best_val_f1:.4f}")

# --- Final Evaluation on Test Dataset ---
if os.path.exists(TEST_FILE):
    try:
        test_df = pd.read_csv(TEST_FILE)
        print(f"\nLoaded test data from {TEST_FILE}")
    except Exception as e:
        print(f"Error loading test.csv: {e}. Skipping test set evaluation.")
        test_df = None
else:
    print(f"Warning: {TEST_FILE} not found. Skipping test set evaluation.")
    test_df = None


if test_df is not None:
    test_df.columns = test_df.columns.str.strip()
    test_df_original_cols = test_df.columns.tolist() # to keep track of original test_df columns

    # Drop the same irrelevant columns as training data
    test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], errors='ignore')

    for col in current_numerical_cols + current_categorical_cols:
        if col not in test_df.columns:
            if col in current_numerical_cols:
                test_df[col] = X[col].mean() # Impute with training mean
            else:
                test_df[col] = X[col].mode()[0] # Impute with training mode

    test_df_for_transform = test_df[current_numerical_cols + current_categorical_cols]

    # Apply the same preprocessor fitted on the training data
    X_test_processed = preprocessor.transform(test_df_for_transform)
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()

    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_preds_binary = (test_outputs > 0.5).int().cpu().numpy().flatten()

    print("\nTest predictions (binary class indices):")
    print(test_preds_binary)

    submission = pd.DataFrame({'prediction': test_preds_binary})
    submission.to_csv(TEST_PREDICTIONS_FILE, index=False)
    print(f"Test predictions saved to {TEST_PREDICTIONS_FILE}")


# --- Save Preprocessor and Feature Information ---
try:
    # Save the full preprocessor pipeline
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    print(f"Preprocessor pipeline saved to {PREPROCESSOR_FILE}")

    # Extract scaler and imputers for specific use in Streamlit app's dummy data logic
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    cat_imputer = preprocessor.named_transformers_['cat'].named_steps['imputer']
    num_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
    joblib.dump((cat_imputer, num_imputer), IMPUTERS_FILE)
    print(f"Imputers saved to {IMPUTERS_FILE}")

    # Prepare and save feature information
    feature_info = {
        'numerical_features': current_numerical_cols, # Numerical columns used in X (before OHE)
        'categorical_features': current_categorical_cols, # Categorical columns used in X (before OHE)
        'final_feature_columns': final_feature_columns_list, # All columns after OHE, in correct order
        'categorical_options': categorical_options_map # Unique options for Streamlit selectboxes
    }
    joblib.dump(feature_info, FEATURE_INFO_FILE)
    print(f"Feature info saved to {FEATURE_INFO_FILE}")

except Exception as e:
    print(f"Error saving preprocessing objects or feature info: {e}")

print("\nModel training and artifact saving complete.")