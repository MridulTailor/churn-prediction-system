import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Load the dataset"""
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        # Convert Total Charges to numeric (it might be stored as string)
        if 'Total Charges' in df.columns:
            df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
            # Fill missing values with median
            df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)
        
        return df
    
    def feature_engineering(self, df):
        """Create new features"""
        services = ['Phone Service', 'Multiple Lines', 'Internet Service', 'Online Security',
                   'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
        
        for service in services:
            if service in df.columns:
                df[f'{service.replace(" ", "")}_Yes'] = (df[service] == 'Yes').astype(int)
        
        # Monthly charges per tenure
        df['MonthlyCharges_per_tenure'] = df['Monthly Charges'] / (df['Tenure Months'] + 1)
        
        # Total charges per monthly charges ratio (if Total Charges exists)
        if 'Total Charges' in df.columns:
            df['TotalCharges_per_MonthlyCharges'] = df['Total Charges'] / (df['Monthly Charges'] + 1)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable if present
        if 'Churn Value' in categorical_cols:
            categorical_cols.remove('Churn Value')
        if 'Churn Label' in categorical_cols:
            categorical_cols.remove('Churn Label')
        
        # Remove columns that shouldn't be encoded
        exclude_cols = ['CustomerID', 'Lat Long', 'City', 'Churn Reason']
        for col in exclude_cols:
            if col in categorical_cols:
                categorical_cols.remove(col)
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """Scale numerical features"""
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            if X_test is not None:
                X_test_scaled = self.scaler.transform(X_test)
                return X_train_scaled, X_test_scaled
            return X_train_scaled
        else:
            return self.scaler.transform(X_train)
    
    def prepare_data(self, file_path, test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""

        df = self.load_data(file_path)
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        
        # Separate features and target
        # Exclude non-predictive columns
        exclude_cols = ['Churn Value', 'Churn Label', 'CustomerID', 'Lat Long', 'City', 'Churn Reason']
        X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
        y = df['Churn Value']
        
        # Encode categorical features
        X = self.encode_categorical_features(X, fit=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
        
        # Store feature names for later use
        self.feature_names_ = X.columns.tolist()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
        'data/telco_churn.csv'
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")