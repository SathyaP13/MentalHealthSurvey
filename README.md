# Depression Prediction using Deep Learning

## Project Overview

- The **Depression Prediction** project aims to develop a robust deep learning model to predict the likelihood of an individual experiencing depression.
- By analyzing various factors such as demographic information, lifestyle choices, and medical history, this model seeks to identify patterns that indicate a higher risk of depression.
- A key focus of this solution is to address the complexities of healthcare data, including potential biases, to ensure equitable and fair predictions for people from diverse backgrounds.

## Approach

The project follows a structured approach encompassing data processing, model development, and deployment using streamlit:

### 1. Data Preprocessing

* **Loading:** Training and test datasets are loaded from specified CSV/Excel files.
* **Cleaning:** Missing values are handled using imputation strategies (e.g., median for numerical, most frequent for categorical).
* **Encoding:** Categorical variables are converted into numerical formats using One-Hot Encoding.
* **Normalization:** Continuous numerical features are scaled using `StandardScaler` to ensure consistent ranges.
* **Splitting:** The training dataset is split into training and validation sets for robust model evaluation.

### 2. Model Development

* **Architecture:** A custom Deep Learning architecture, specifically a Multilayer Perceptron (MLP), is built using PyTorch.
* **Layers:** The model incorporates several linear layers with ReLU activation, Batch Normalization for stability, and Dropout for regularization.
* **Output:** The final output layer uses a Sigmoid activation function, suitable for binary classification (predicting the probability of depression).

### 3. Pipeline Concept

* **Data Pipeline:** Ensures a seamless flow of data from raw input through preprocessing, and make it ready for model training and testing.
* **Model Training Pipeline:** Automates the iterative process of training, evaluating, and refining the deep learning model.

### 4. Evaluation Metrics

Model performance is evaluated using:

* **Accuracy:** Overall percentage of correct predictions.
* **Precision:** Proportion of true positive predictions among all positive predictions.
* **Recall:** Proportion of true positive predictions among all actual positive cases.
* **F1-Score:** The harmonic mean of Precision and Recall, providing a balance between the two.
* **Bias Evaluation:** A critical aspect involves assessing the model's fairness across different demographic groups (e.g., age, gender, race) to ensure equitable predictions and prevent discriminatory outcomes.

### 5. Model Deployment

* **Streamlit Application:** A user-friendly web application is developed using Streamlit, allowing users to input their data and receive real-time depression predictions.

## Technology stack
### Programming Language:
- Python: The core language for all development (data processing, model training, application).

### Deep Learning Framework:
- PyTorch: For building, training, and managing the neural network models.

### Data Handling & Preprocessing:
- Pandas: For data manipulation and analysis (loading CSVs, DataFrame operations).
- NumPy: For numerical operations, especially with arrays.
- Scikit-learn: For data preprocessing utilities (e.g., StandardScaler, OneHotEncoder, SimpleImputer, train_test_split).
- Joblib: For saving and loading Python objects (like the ColumnTransformer preprocessor).

### Model Deployment & Web Application:
- Streamlit: For creating the interactive web application user interface.

* **`datapreprocess.py`:** Python script for cleaning, transforming, and preparing the raw data.
* **`modeltraining.py`:** Python script for building, training, and evaluating the deep learning model.
* **`mentalhealth.py`:** Python script for the interactive web application.
* **`model/` directory:**
    * `preprocessor.pkl`: The fitted scikit-learn preprocessor object.
    * `depression_model.pth`: The trained PyTorch model's state dictionary.
    * `feature_info.pkl`: Information about processed features, including categories for Streamlit.
    * `train_processed.pt`, `val_processed.pt`, `test_processed.pt`: Preprocessed data in PyTorch tensor format.

## Setup and Running the Project

### Prerequisites

* Python 3.x
* `pip` package manager

### Installation

1.  **Create a virtual environment (recommended):**
    ```
    python -m venv venv
    source venv/bin/activate
    # On Windows: `venv\Scripts\activate` 
    ```
2.  **Install dependencies:**
    ```
    pip install pandas numpy scikit-learn torch joblib streamlit
    ```

### Running the Project

1.  **Place your dataset:**
    Ensure your `train.csv` and `test.csv` (or equivalent data files) are placed in a directory named `data` at the root of your project:

2.  **Run Data Preprocessing:**
    ```bash
    python datapreprocess.py
    ```
    This script will process the raw data and save preprocessed files and the preprocessor object in the `model/` directory.

3.  **Run Model Training:**
    ```bash
    python modeltraining.py
    ```
    This script will train the deep learning model using the preprocessed data and save the trained model in the `model/` directory.

4.  **Run Streamlit Application Locally:**
    ```bash
    streamlit run mentalhealth.py
    ```
    This will launch the web application in your default browser, typically at `http://localhost:8501`.

## Support and Contribution

Contributions are welcome! Please feel free to open issues or submit pull requests.

**Disclaimer:** This model is intended for research and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any questions you may have regarding a medical condition or mental health concerns.
