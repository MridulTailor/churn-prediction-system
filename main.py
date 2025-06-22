#!/usr/bin/env python3
"""
Customer Churn Prediction System - Main Script
"""

import os
import sys
from pathlib import Path

sys.path.append('src')

from data_preprocessing import DataPreprocessor
from eda import EDAAnalyzer
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

def create_project_structure():
    """Create necessary directories"""
    directories = ['data', 'src', 'models', 'static/plots', 'templates']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("Project structure created successfully!")

def check_data_file():
    """Check if the dataset file exists and validate its structure"""
    data_file = 'data/telco_churn.csv'
    if not os.path.exists(data_file):
        print(f"❌ Dataset not found at {data_file}")
        print("Please download the IBM Telco Customer Churn dataset from Kaggle and place it in the data/ directory")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"✅ Dataset found at {data_file}")
        print(f"📊 Dataset shape: {df.shape}")
        
        # Check for required columns
        required_cols = ['Churn Value', 'Churn Label', 'Monthly Charges', 'Tenure Months']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"⚠️  Warning: Missing required columns: {missing_cols}")
            print("Dataset might not be in the expected format")
        else:
            print("✅ Dataset structure validated")
        
        return True
    except Exception as e:
        print(f"❌ Error reading dataset: {str(e)}")
        return False

def run_complete_pipeline():
    """Run the complete machine learning pipeline"""
    data_file = 'data/telco_churn.csv'
    
    print("🚀 Starting Customer Churn Prediction Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: EDA
        print("📊 Step 1: Exploratory Data Analysis")
        analyzer = EDAAnalyzer()
        df, correlations = analyzer.run_complete_eda(data_file)
        print("✅ EDA completed")
        
        # Step 2: Model Training
        print("\n🤖 Step 2: Model Training")
        trainer = ModelTrainer()
        best_model, feature_names, X_test, y_test = trainer.train_all_models(data_file)
        print("✅ Model training completed")
        
        # Step 3: Model Evaluation
        print("\n📈 Step 3: Model Evaluation")
        evaluator = ModelEvaluator()
        
        # Load and evaluate all models
        import joblib
        
        if os.path.exists('models/gradient_boosting_model.pkl'):
            gb_model = joblib.load('models/gradient_boosting_model.pkl')
            evaluator.evaluate_model(gb_model, X_test, y_test, feature_names, 'Gradient Boosting')
        
        if os.path.exists('models/logistic_regression_model.pkl'):
            lr_model = joblib.load('models/logistic_regression_model.pkl')
            evaluator.evaluate_model(lr_model, X_test, y_test, feature_names, 'Logistic Regression')
        
        evaluator.compare_all_models()
        print("✅ Model evaluation completed")
        
        print("\n🎉 Pipeline completed successfully!")
        print("📁 Check the following directories for outputs:")
        print("   - models/: Trained models")
        print("   - static/plots/: Visualizations")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in pipeline: {str(e)}")
        print("Please check your data file and try again.")
        return False

def main():
    """Main function"""
    print("Customer Churn Prediction System")
    print("=" * 40)
    
    create_project_structure()
    
    # Check if data file exists
    if not check_data_file():
        return
    
    print("\nChoose an option:")
    print("1. Run complete pipeline (EDA + Training + Evaluation)")
    print("2. Run only EDA")
    print("3. Run only model training")
    print("4. Run only model evaluation")
    print("5. Start Flask API server")
    print("6. View dataset information")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        run_complete_pipeline()
    elif choice == '2':
        try:
            analyzer = EDAAnalyzer()
            analyzer.run_complete_eda('data/telco_churn.csv')
            print("✅ EDA completed successfully!")
        except Exception as e:
            print(f"❌ Error in EDA: {str(e)}")
    elif choice == '3':
        try:
            trainer = ModelTrainer()
            trainer.train_all_models('data/telco_churn.csv')
            print("✅ Model training completed successfully!")
        except Exception as e:
            print(f"❌ Error in model training: {str(e)}")
    elif choice == '4':
        # Check if models exist
        if not os.path.exists('models/best_model.pkl'):
            print("❌ No trained models found. Please run training first.")
            return
        
        try:
            from model_evaluation import ModelEvaluator
            import joblib
            
            # Load preprocessor and prepare test data
            preprocessor = joblib.load('models/preprocessor.pkl')
            X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
                'data/telco_churn.csv'
            )
            
            evaluator = ModelEvaluator()
            best_model = joblib.load('models/best_model.pkl')
            evaluator.evaluate_model(best_model, X_test, y_test, feature_names, 'Best Model')
            print("✅ Model evaluation completed successfully!")
        except Exception as e:
            print(f"❌ Error in model evaluation: {str(e)}")
    elif choice == '5':
        # Check if models exist
        if not os.path.exists('models/best_model.pkl'):
            print("❌ No trained models found. Please run training first.")
            return
        
        try:
            print("🌐 Starting Flask API server...")
            print("📱 Web interface will be available at: http://localhost:8000")
            print("🔗 API endpoint: http://localhost:8000/predict")
            print("Press Ctrl+C to stop the server")
            
            from api import app
            app.run(debug=True, host='0.0.0.0', port=8000)
        except Exception as e:
            print(f"❌ Error starting server: {str(e)}")
    elif choice == '6':
        try:
            import pandas as pd
            df = pd.read_csv('data/telco_churn.csv')
            print("\n📊 Dataset Information:")
            print("=" * 40)
            print(f"Shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")
            print("\nColumn Names:")
            for i, col in enumerate(df.columns, 1):
                print(f"{i:2d}. {col}")
            print(f"\nChurn Distribution:")
            if 'Churn Label' in df.columns:
                print(df['Churn Label'].value_counts())
            elif 'Churn Value' in df.columns:
                print(df['Churn Value'].value_counts())
            print(f"\nMissing Values:")
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                print(missing_values[missing_values > 0])
            else:
                print("No missing values found")
        except Exception as e:
            print(f"❌ Error reading dataset: {str(e)}")
    else:
        print("❌ Invalid choice. Please select 1-6.")

if __name__ == "__main__":
    main()