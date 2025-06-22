from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import json

template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Global variables for models
model = None
preprocessor = None
feature_names = None

def load_models():
    """Load models and preprocessor"""
    global model, preprocessor, feature_names
    
    try:
        model = joblib.load('models/best_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        
        if hasattr(preprocessor, 'feature_names_'):
            feature_names = preprocessor.feature_names_
        else:
            try:
                with open('models/feature_names.json', 'r') as f:
                    feature_names = json.load(f)
                    preprocessor.feature_names_ = feature_names
            except FileNotFoundError:
                print("⚠️  Feature names file not found.")
        
        print("✅ Models loaded successfully")
        if feature_names:
            print(f"📊 Feature names loaded: {len(feature_names)} features")
            print(f"Expected features: {feature_names[:5]}...")
        
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    if model is None or preprocessor is None:
        return jsonify({
            'error': 'Models not loaded. Please run model training first.',
            'details': 'Run: python src/model_training.py'
        }), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        sample_customer = create_sample_customer(data)
        input_df = pd.DataFrame([sample_customer])
        print(f"Input data shape: {input_df.shape}")
        print(f"Input columns: {list(input_df.columns)}")
        processed_df = process_prediction_data(input_df)
        
        prediction = model.predict(processed_df)[0]
        probability = model.predict_proba(processed_df)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': {
                'no_churn': float(probability[0]),
                'churn': float(probability[1])
            },
            'churn_probability': float(probability[1]),
            'churn_risk': 'High' if probability[1] > 0.7 else 'Medium' if probability[1] > 0.3 else 'Low'
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': str(e),
            'suggestion': 'Check that all required fields are provided and values are valid'
        }), 400

def create_sample_customer(input_data):
    """Create a complete customer record with defaults for missing fields"""
    
    field_mapping = {
        'gender': 'Gender',
        'SeniorCitizen': 'Senior Citizen',
        'Partner': 'Partner',
        'Dependents': 'Dependents',
        'tenure': 'Tenure Months',
        'PhoneService': 'Phone Service',
        'MultipleLines': 'Multiple Lines',
        'InternetService': 'Internet Service',
        'OnlineSecurity': 'Online Security',
        'OnlineBackup': 'Online Backup',
        'DeviceProtection': 'Device Protection',
        'TechSupport': 'Tech Support',
        'StreamingTV': 'Streaming TV',
        'StreamingMovies': 'Streaming Movies',
        'Contract': 'Contract',
        'PaperlessBilling': 'Paperless Billing',
        'PaymentMethod': 'Payment Method',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges'
    }
    
    customer = {
        'CustomerID': 'API_CUSTOMER_001',
        'Count': 1,
        'Country': 'United States',
        'State': 'California',
        'City': 'Los Angeles',
        'Zip Code': 90210,
        'Lat Long': '34.0522,-118.2437',
        'Latitude': 34.0522,
        'Longitude': -118.2437,
        'Gender': 'Male',
        'Senior Citizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'Tenure Months': 12,
        'Phone Service': 'Yes',
        'Multiple Lines': 'No',
        'Internet Service': 'DSL',
        'Online Security': 'No',
        'Online Backup': 'No',
        'Device Protection': 'No',
        'Tech Support': 'No',
        'Streaming TV': 'No',
        'Streaming Movies': 'No',
        'Contract': 'Month-to-month',
        'Paperless Billing': 'No',
        'Payment Method': 'Electronic check',
        'Monthly Charges': 50.0,
        'Total Charges': 600.0,
        'Churn Score': 50,
        'CLTV': 1200.0
    }
    
    for api_field, dataset_field in field_mapping.items():
        if api_field in input_data and input_data[api_field] is not None:
            customer[dataset_field] = input_data[api_field]
    
    
    # If no phone service, set multiple lines accordingly
    if customer['Phone Service'] == 'No':
        customer['Multiple Lines'] = 'No phone service'
    
    # If no internet service, set internet-dependent services accordingly
    if customer['Internet Service'] == 'No':
        for service in ['Online Security', 'Online Backup', 'Device Protection', 
                       'Tech Support', 'Streaming TV', 'Streaming Movies']:
            customer[service] = 'No internet service'
    
    # Calculate Total Charges if not provided
    if 'TotalCharges' not in input_data and 'MonthlyCharges' in input_data and 'tenure' in input_data:
        customer['Total Charges'] = customer['Monthly Charges'] * customer['Tenure Months']
    
    customer['CLTV'] = customer['Monthly Charges'] * customer['Tenure Months'] * 2
    
    return customer

def process_prediction_data(df):
    """Process data through the same pipeline as training"""
    try:
        df = preprocessor.clean_data(df)
        
        df = preprocessor.feature_engineering(df)
        
        exclude_cols = ['CustomerID', 'Churn Value', 'Churn Label', 'Lat Long', 'City', 'Churn Reason']
        df = df.drop([col for col in exclude_cols if col in df.columns], axis=1)
        
        print(f"After cleaning and feature engineering: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        df = preprocessor.encode_categorical_features(df, fit=False)
        
        print(f"After encoding: {df.shape}")
        
        # Handle feature alignment if we have the expected feature names
        if feature_names:
            current_features = set(df.columns)
            expected_features = set(feature_names)
            
            # Add missing features with default values (0)
            missing_features = expected_features - current_features
            for feature in missing_features:
                df[feature] = 0
                print(f"Added missing feature: {feature}")
            
            # Remove extra features
            extra_features = current_features - expected_features
            if extra_features:
                df = df.drop(columns=list(extra_features))
                print(f"Removed extra features: {extra_features}")
            
            # Reorder columns to match training order
            df = df.reindex(columns=feature_names, fill_value=0)
            
            print(f"After feature alignment: {df.shape}")
        
        df_scaled = preprocessor.scale_features(df, fit=False)
        
        print(f"Final processed shape: {df_scaled.shape}")
        
        return df_scaled
        
    except Exception as e:
        print(f"Error in process_prediction_data: {e}")
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'preprocessor_loaded': preprocessor is not None,
            'feature_names_loaded': feature_names is not None,
            'expected_features_count': len(feature_names) if feature_names else 0,
            'model_type': type(model).__name__ if model else None
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'model_loaded': model is not None,
            'model_type': type(model).__name__ if model else None,
            'preprocessor_loaded': preprocessor is not None,
            'feature_names_available': feature_names is not None,
            'expected_features_count': len(feature_names) if feature_names else 0
        }
        
        if feature_names:
            info['sample_features'] = feature_names[:10]  # First 10 features
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reload_models', methods=['POST'])
def reload_models():
    """Reload models without restarting the server"""
    success = load_models()
    if success:
        return jsonify({'status': 'Models reloaded successfully'})
    else:
        return jsonify({'error': 'Failed to reload models'}), 500

if __name__ == '__main__':
    print("🌐 Starting Flask API server...")
    print("📱 Web interface: http://localhost:8000")
    print("🔗 API endpoint: http://localhost:8000/predict")
    print("🏥 Health check: http://localhost:8000/health")
    print("📊 Model info: http://localhost:8000/model_info")
    print("Press Ctrl+C to stop the server")
    
    if model is None:
        print("⚠️  Models not loaded. Run: python src/model_training.py")
    
    app.run(debug=True, host='0.0.0.0', port=8000)