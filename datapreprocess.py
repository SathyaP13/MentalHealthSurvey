import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import joblib
import os

print("Starting data preprocessing...")

# --- Configuration --- #
DATA_DIR = 'data'
MODEL_DIR = 'model'

TRAIN_FILE = os.path.join(DATA_DIR, 'train.csv')
TEST_FILE = os.path.join(DATA_DIR, 'test.csv')

PREPROCESSOR_FILE = os.path.join(MODEL_DIR, 'preprocessor.pkl')
TRAIN_PROCESSED_FILE = os.path.join(MODEL_DIR, 'train_processed.pt')
VAL_PROCESSED_FILE = os.path.join(MODEL_DIR, 'val_processed.pt')
TEST_PROCESSED_FILE = os.path.join(MODEL_DIR, 'test_processed.pt')
FEATURE_INFO_FILE = os.path.join(MODEL_DIR, 'feature_info.pkl') # File to save feature names and categories
TEST_IDS_FILE = os.path.join(MODEL_DIR, 'test_ids.pkl') # File to save test IDs
TARGET_COLUMN = 'Depression'
ID_COLUMNS = ['id', 'Name'] # Columns to be dropped from features

# Explicitly define features that *must* be treated as categorical, even if they contain numbers #
FORCE_CATEGORICAL_FEATURES = ['Degree', 'Dietary Habits', 'Sleep Duration', 'Gender',
                              'Working Professional or Student', 'Suicidal Thoughts', 'Family History of Mental Illness',
                              'City', 'Profession']

# Create necessary directories #
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Data --- #
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    print(f"Loaded {TRAIN_FILE} with shape: {train_df.shape}")
    print(f"Loaded {TEST_FILE} with shape: {test_df.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure '{TRAIN_FILE}' and '{TEST_FILE}' exist in the correct path relative to this script.")
    exit()

X = train_df.drop(columns=[col for col in ID_COLUMNS if col in train_df.columns] + [TARGET_COLUMN], errors='ignore')
y = train_df[TARGET_COLUMN]

test_ids = test_df['id'] # Keep test IDs for submission
X_test = test_df.drop(columns=[col for col in ID_COLUMNS if col in test_df.columns], errors='ignore')


# --- Identify Numerical and Categorical Features --- #
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Force certain numerical-looking columns to be categorical #
for col in FORCE_CATEGORICAL_FEATURES:
    if col in numerical_features:
        numerical_features.remove(col)
        categorical_features.append(col)
    elif col not in categorical_features and col in X.columns:
        categorical_features.append(col)

# Ensure no duplicates and maintain order #
numerical_features = sorted(list(set(numerical_features)))
categorical_features = sorted(list(set(categorical_features))) # Sort for consistent order

print(f"Final Numerical features: {numerical_features}")
print(f"Final Categorical features: {categorical_features}")

# --- Create Preprocessing Pipelines --- #
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # 'ignore' handles unseen categories gracefully
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) as they are.
)

# --- Fit Preprocessor on Training Data and Transform --- #
print("Fitting preprocessor and transforming data...")
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# --- Ensure all processed arrays are dense NumPy arrays --- #
if hasattr(X_processed, 'toarray'):
    X_processed = X_processed.toarray()
if hasattr(X_test_processed, 'toarray'):
    X_test_processed = X_test_processed.toarray()

print(f"Shape of X_processed (after toarray conversion): {X_processed.shape}")
print(f"Shape of X_test_processed (after toarray conversion): {X_test_processed.shape}")

# --- Extract categories from the fitted OneHotEncoder for Streamlit --- #
categorical_options_map = {}
if 'cat' in preprocessor.named_transformers_ and hasattr(preprocessor.named_transformers_['cat'], 'named_steps') and 'onehot' in preprocessor.named_transformers_['cat'].named_steps:
    onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    for i, col_name in enumerate(categorical_features):
        if i < len(onehot_encoder.categories_):
            categorical_options_map[col_name] = onehot_encoder.categories_[i].tolist()
else:
    print("Warning: 'cat' transformer or 'onehot' step not found in preprocessor. Categorical options will be empty.")

# --- Split Training Data into Train and Validation Sets --- #
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test_processed shape: {X_test_processed.shape}")

# --- Convert to PyTorch Tensors --- #
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

# --- Save Preprocessor and Processed Data --- #
print("Saving preprocessor and processed data...")
joblib.dump(preprocessor, PREPROCESSOR_FILE)

# Save feature information including categories for Streamlit app #
joblib.dump({
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'categorical_options': categorical_options_map # Actual categories learned by OHE
}, FEATURE_INFO_FILE)

torch.save({'X_train': X_train_tensor, 'y_train': y_train_tensor}, TRAIN_PROCESSED_FILE)
torch.save({'X_val': X_val_tensor, 'y_val': y_val_tensor}, VAL_PROCESSED_FILE)
torch.save(X_test_tensor, TEST_PROCESSED_FILE)
joblib.dump(test_ids, TEST_IDS_FILE)

print("Data preprocessing complete.")
print(f"Preprocessor saved to: {PREPROCESSOR_FILE}")
print(f"Feature info (names & categories) saved to: {FEATURE_INFO_FILE}")
print(f"Processed training data saved to: {TRAIN_PROCESSED_FILE}")
print(f"Processed validation data saved to: {VAL_PROCESSED_FILE}")
print(f"Processed test data saved to: {TEST_PROCESSED_FILE}")