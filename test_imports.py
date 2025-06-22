#!/usr/bin/env python3
"""
Test script to validate all imports and basic functionality
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append('src')

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import DataPreprocessor: {e}")
        return False
    
    try:
        from eda import EDAAnalyzer
        print("✅ EDAAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import EDAAnalyzer: {e}")
        return False
    
    try:
        from model_training import ModelTrainer
        print("✅ ModelTrainer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ModelTrainer: {e}")
        return False
    
    try:
        from model_evaluation import ModelEvaluator
        print("✅ ModelEvaluator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ModelEvaluator: {e}")
        return False
    
    try:
        from api import app
        print("✅ Flask API imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Flask API: {e}")
        return False
    
    return True

def test_data_file():
    """Test if data file exists and can be loaded"""
    print("\n🧪 Testing data file...")
    
    data_file = 'data/telco_churn.csv'
    if not os.path.exists(data_file):
        print(f"❌ Dataset not found at {data_file}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(data_file)
        print(f"✅ Dataset loaded successfully: {df.shape}")
        
        # Check key columns
        key_columns = ['Churn Value', 'Monthly Charges', 'Tenure Months']
        missing = [col for col in key_columns if col not in df.columns]
        if missing:
            print(f"⚠️  Missing key columns: {missing}")
        else:
            print("✅ Key columns found")
        
        return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n🧪 Testing directories...")
    
    required_dirs = ['data', 'src', 'models', 'static/plots', 'templates']
    all_exist = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/ exists")
        else:
            print(f"❌ {directory}/ missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("Customer Churn Prediction System - Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test directories
    dirs_ok = test_directories()
    
    # Test data file
    data_ok = test_data_file()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    
    if imports_ok and dirs_ok and data_ok:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' to start the interactive menu")
        print("2. Or run individual components directly")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        if not imports_ok:
            print("   - Check that all required packages are installed")
        if not dirs_ok:
            print("   - Create missing directories")
        if not data_ok:
            print("   - Download and place the dataset in data/telco_churn.csv")
        return False

if __name__ == "__main__":
    main()
