import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
import os
from data_preprocessing import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.preprocessor = DataPreprocessor()
        
    def create_models_directory(self):
        """Create directory for saving models"""
        os.makedirs('models', exist_ok=True)
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting Classifier with hyperparameter tuning"""
        print("Training Gradient Boosting Classifier...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        gb_classifier = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            gb_classifier, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_gb = grid_search.best_estimator_
        
        y_pred = best_gb.predict(X_test)
        y_pred_proba = best_gb.predict_proba(X_test)[:, 1]
        
        self.models['gradient_boosting'] = {
            'model': best_gb,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Best GB parameters: {grid_search.best_params_}")
        print(f"Best GB CV score: {grid_search.best_score_:.4f}")
        print(f"GB Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return best_gb
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression with hyperparameter tuning"""
        print("Training Logistic Regression...")
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
        
        grid_search = GridSearchCV(
            lr_classifier, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        y_pred = best_lr.predict(X_test)
        y_pred_proba = best_lr.predict_proba(X_test)[:, 1]
        
        self.models['logistic_regression'] = {
            'model': best_lr,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"Best LR parameters: {grid_search.best_params_}")
        print(f"Best LR CV score: {grid_search.best_score_:.4f}")
        print(f"LR Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        return best_lr
    
    def compare_models(self, y_test):
        """Compare model performances"""
        print("\nModel Comparison:")
        print("-" * 50)
        
        best_score = 0
        best_model_name = None
        
        for name, model_info in self.models.items():
            score = model_info['roc_auc']
            print(f"{name.upper()}: ROC-AUC = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = name
        
        print(f"\nBest Model: {best_model_name.upper()} with ROC-AUC = {best_score:.4f}")
        self.best_model = self.models[best_model_name]
        
        return best_model_name, self.best_model
    
    def save_models(self):
        """Save trained models"""
        self.create_models_directory()
        
        for name, model_info in self.models.items():
            joblib.dump(model_info['model'], f'models/{name}_model.pkl')
            print(f"Saved {name} model")
        
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        print("Saved preprocessor")
        
        if self.best_model:
            joblib.dump(self.best_model['model'], 'models/best_model.pkl')
            print("Saved best model")
            
        if hasattr(self.preprocessor, 'feature_names_'):
            import json
            with open('models/feature_names.json', 'w') as f:
                json.dump(self.preprocessor.feature_names_, f)
            print("Saved feature names")
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model_info in self.models.items():
            fpr, tpr, _ = roc_curve(y_test, model_info['y_pred_proba'])
            auc_score = model_info['roc_auc']
            plt.plot(fpr, tpr, label=f'{name.upper()} (AUC = {auc_score:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('static/plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_all_models(self, file_path):
        """Train all models and return the best one"""

        X_train, X_test, y_train, y_test, feature_names = self.preprocessor.prepare_data(file_path)
        
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        
        best_model_name, best_model = self.compare_models(y_test)
        
        self.plot_roc_curves(y_test)
        
        self.save_models()
        
        return best_model, feature_names, X_test, y_test

if __name__ == "__main__":
    trainer = ModelTrainer()
    best_model, feature_names, X_test, y_test = trainer.train_all_models(
        'data/telco_churn.csv'
    )
    print("Model training completed!")