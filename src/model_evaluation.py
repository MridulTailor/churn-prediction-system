import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import joblib

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.create_plots_directory()
    
    def create_plots_directory(self):
        """Create directory for saving plots"""
        import os
        os.makedirs('static/plots', exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba, model_name):
        """Calculate all evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'static/plots/confusion_matrix_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_feature_importance(self, model, feature_names, model_name):
        """Plot feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            print(f"Feature importance not available for {model_name}")
            return
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance_df.head(15)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top 15 Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'static/plots/feature_importance_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def generate_classification_report(self, y_true, y_pred, model_name):
        """Generate and display classification report"""
        report = classification_report(y_true, y_pred, 
                                     target_names=['No Churn', 'Churn'],
                                     output_dict=True)
        
        print(f"\nClassification Report - {model_name}")
        print("-" * 50)
        print(classification_report(y_true, y_pred, 
                                  target_names=['No Churn', 'Churn']))
        
        return report
    
    def evaluate_model(self, model, X_test, y_test, feature_names, model_name):
        """Complete model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
        
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        self.plot_feature_importance(model, feature_names, model_name)
        report = self.generate_classification_report(y_test, y_pred, model_name)
        
        print(f"\nMetrics Summary - {model_name}")
        print("-" * 30)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics, report
    
    def compare_all_models(self):
        """Compare all evaluated models"""
        if not self.metrics:
            print("No models have been evaluated yet.")
            return
        
        comparison_df = pd.DataFrame(self.metrics).T
        
        print("\nModel Comparison Summary")
        print("=" * 60)
        print(comparison_df.round(4))
        
        plt.figure(figsize=(12, 8))
        comparison_df.plot(kind='bar', ax=plt.gca())
        plt.title('Model Performance Comparison')
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df

if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    import joblib
    import os
    
    # Check if models exist
    if not os.path.exists('models/preprocessor.pkl'):
        print("❌ Preprocessor not found. Please run model training first.")
        exit(1)
    
    if not os.path.exists('models/gradient_boosting_model.pkl'):
        print("❌ Gradient Boosting model not found. Please run model training first.")
        exit(1)
        
    if not os.path.exists('models/logistic_regression_model.pkl'):
        print("❌ Logistic Regression model not found. Please run model training first.")
        exit(1)
    
    preprocessor = joblib.load('models/preprocessor.pkl')
    
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(
        'data/telco_churn.csv'
    )
    
    evaluator = ModelEvaluator()
    
    print("📊 Evaluating Gradient Boosting Model...")
    gb_model = joblib.load('models/gradient_boosting_model.pkl')
    evaluator.evaluate_model(gb_model, X_test, y_test, feature_names, 'Gradient Boosting')
    
    print("\n📊 Evaluating Logistic Regression Model...")
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    evaluator.evaluate_model(lr_model, X_test, y_test, feature_names, 'Logistic Regression')
    
    print("\n📊 Comparing All Models...")
    evaluator.compare_all_models()
    
    print("\n✅ Model evaluation completed successfully!")
    print("📁 Check static/plots/ directory for visualizations.")