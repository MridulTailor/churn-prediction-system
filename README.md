# Customer Churn Prediction System

A comprehensive machine learning system for predicting customer churn using the IBM Telco Customer Churn dataset. This project includes exploratory data analysis, model training, evaluation, and a web interface for making predictions.

## Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis with visualizations
- **Multiple ML Models**: Gradient Boosting and Logistic Regression
- **Model Evaluation**: Detailed performance metrics and comparisons
- **Web Interface**: Flask-based web application for real-time predictions
- **API Endpoints**: RESTful API for integration with other systems
- **Automated Pipeline**: Complete end-to-end machine learning pipeline

## Dataset

This project uses the IBM Telco Customer Churn dataset with 33 columns including:

- Customer demographics (Gender, Senior Citizen, Partner, Dependents)
- Geographic information (Country, State, City, Lat/Long)
- Service information (Phone, Internet, Streaming services, etc.)
- Account information (Contract, Payment Method, Charges)
- Churn information (Churn Label, Churn Value, Churn Score, CLTV)

## Project Structure

```
churn-prediction-system/
├── data/
│   └── telco_churn.csv         # Dataset
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preprocessing
│   ├── eda.py                 # Exploratory data analysis
│   ├── model_training.py      # Model training and selection
│   ├── model_evaluation.py    # Model evaluation and metrics
│   └── api.py                 # Flask API server
├── models/                    # Saved models and preprocessors
├── static/plots/             # Generated visualizations
├── templates/
│   └── index.html            # Web interface
├── main.py                   # Main script with interactive menu
├── test_imports.py           # Test script for validation
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd churn-prediction-system
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**

   - Download the IBM Telco Customer Churn dataset from Kaggle
   - Place it as `data/telco_churn.csv`

4. **Test the installation:**
   ```bash
   python test_imports.py
   ```

## Usage

### Interactive Menu

Run the main script for an interactive menu:

```bash
python main.py
```

Options available:

1. Run complete pipeline (EDA + Training + Evaluation)
2. Run only EDA
3. Run only model training
4. Run only model evaluation
5. Start Flask API server
6. View dataset information

### Web Interface

Start the web server and visit `http://localhost:8000`:

```bash
python main.py
# Choose option 5
```

### API Usage

Send POST requests to `http://localhost:8000/predict` with customer data:

```python
import requests

data = {
    "gender": "Male",
    "SeniorCitizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35
}

response = requests.post('http://localhost:8000/predict', json=data)
print(response.json())
```

## Key Features

### Smart Data Handling

- Automatic field mapping from web form to dataset columns
- Missing value imputation
- Feature engineering (service counts, charge ratios)
- Handles optional fields (Total Charges calculated if missing)

### Robust Preprocessing

- Label encoding for categorical variables
- Standard scaling for numerical features
- Proper train/test splitting with stratification
- Feature exclusion for non-predictive columns

### Comprehensive Evaluation

- Multiple performance metrics
- Visual comparisons between models
- Feature importance analysis
- Confusion matrices and classification reports
