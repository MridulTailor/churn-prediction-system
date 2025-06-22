import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
import os

class EDAAnalyzer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        
    def create_plots_directory(self):
        """Create directory for saving plots"""
        os.makedirs('static/plots', exist_ok=True)
    
    def basic_statistics(self, df):
        """Generate basic statistics"""
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nChurn Distribution:")
        print(df['Churn Label'].value_counts())
        print(f"Churn Rate: {df['Churn Value'].mean():.2%}")
        print(f"\nChurn Score Range: {df['Churn Score'].min()} - {df['Churn Score'].max()}")
        print(f"Average CLTV: ${df['CLTV'].mean():.2f}")
        print(f"\nChurn Reason (non-null): {df['Churn Reason'].notna().sum()} out of {len(df)}")
        
    def plot_churn_distribution(self, df):
        """Plot churn distribution"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        df['Churn Label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Churn Distribution (Count)')
        plt.xlabel('Churn')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
        
        plt.subplot(1, 2, 2)
        df['Churn Label'].value_counts(normalize=True).plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Churn Distribution (Percentage)')
        plt.xlabel('Churn')
        plt.ylabel('Percentage')
        plt.xticks([0, 1], ['No', 'Yes'], rotation=0)
        
        plt.tight_layout()
        plt.savefig('static/plots/churn_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_numerical_features(self, df):
        """Plot distribution of numerical features"""
        # Include all relevant numerical columns
        numerical_cols = ['Tenure Months', 'Monthly Charges', 'Churn Score', 'CLTV']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(numerical_cols):
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
            axes[i+4].boxplot([df[df['Churn Value']==0][col], df[df['Churn Value']==1][col]], 
                             tick_labels=['No Churn', 'Churn'])
            axes[i+4].set_title(f'{col} by Churn Status')
            axes[i+4].set_ylabel(col)
        
        plt.tight_layout()
        plt.savefig('static/plots/numerical_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_categorical_features(self, df):
        """Plot categorical features vs churn"""
        categorical_cols = ['Gender', 'Senior Citizen', 'Partner', 'Dependents', 
                           'Contract', 'Payment Method', 'Internet Service', 'Paperless Billing']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(categorical_cols):
            churn_by_category = df.groupby([col, 'Churn Label']).size().unstack(fill_value=0)
            churn_rate = churn_by_category.div(churn_by_category.sum(axis=1), axis=0)
            
            if 'Yes' in churn_rate.columns:
                churn_rate['Yes'].plot(kind='bar', ax=axes[i], color='coral')
            else:
                churn_rate.iloc[:, -1].plot(kind='bar', ax=axes[i], color='coral')
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('static/plots/categorical_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_service_features(self, df):
        """Plot service-related features vs churn"""
        service_cols = ['Phone Service', 'Multiple Lines', 'Online Security', 'Online Backup',
                       'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(service_cols):
            churn_by_category = df.groupby([col, 'Churn Label']).size().unstack(fill_value=0)
            churn_rate = churn_by_category.div(churn_by_category.sum(axis=1), axis=0)
            
            if 'Yes' in churn_rate.columns:
                churn_rate['Yes'].plot(kind='bar', ax=axes[i], color='lightcoral')
            else:
                churn_rate.iloc[:, -1].plot(kind='bar', ax=axes[i], color='lightcoral')
            axes[i].set_title(f'Churn Rate by {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('static/plots/service_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def correlation_heatmap(self, df):
        """Create correlation heatmap"""
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numerical_df.corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.tight_layout()
        plt.savefig('static/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_importance_analysis(self, df):
        """Analyze feature importance using correlation with target"""
        # Convert categorical variables to numerical for correlation analysis
        df_encoded = df.copy()
        
        # Columns to exclude from correlation analysis
        exclude_cols = ['CustomerID', 'Lat Long', 'City']
        
        # Remove excluded columns
        for col in exclude_cols:
            if col in df_encoded.columns:
                df_encoded = df_encoded.drop(col, axis=1)
        
        # Encode categorical variables
        for col in df_encoded.select_dtypes(include=['object']).columns:
            df_encoded[col] = pd.Categorical(df_encoded[col]).codes
        
        # Calculate correlation with churn
        correlations = df_encoded.corr()['Churn Value'].abs().sort_values(ascending=False)
        correlations = correlations.drop('Churn Value')  # Remove self-correlation
        
        plt.figure(figsize=(10, 8))
        correlations.head(15).plot(kind='barh', color='steelblue')
        plt.title('Top 15 Features by Correlation with Churn')
        plt.xlabel('Absolute Correlation')
        plt.tight_layout()
        plt.savefig('static/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlations
    
    def plot_geographic_features(self, df):
        """Plot geographic features vs churn"""
        geo_cols = ['Country', 'State']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, col in enumerate(geo_cols):
            # Get top 10 locations by count
            top_locations = df[col].value_counts().head(10).index
            df_filtered = df[df[col].isin(top_locations)]
            
            churn_by_category = df_filtered.groupby([col, 'Churn Label']).size().unstack(fill_value=0)
            churn_rate = churn_by_category.div(churn_by_category.sum(axis=1), axis=0)
            
            if 'Yes' in churn_rate.columns:
                churn_rate['Yes'].plot(kind='bar', ax=axes[i], color='darkorange')
            else:
                churn_rate.iloc[:, -1].plot(kind='bar', ax=axes[i], color='darkorange')
            axes[i].set_title(f'Churn Rate by {col} (Top 10)')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Churn Rate')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('static/plots/geographic_features.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tenure_and_charges_analysis(self, df):
        """Plot detailed analysis of tenure and charges"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0, 0].hist([df[df['Churn Value']==0]['Tenure Months'], 
                        df[df['Churn Value']==1]['Tenure Months']], 
                       bins=30, alpha=0.7, label=['No Churn', 'Churn'], 
                       color=['skyblue', 'salmon'])
        axes[0, 0].set_title('Tenure Distribution by Churn')
        axes[0, 0].set_xlabel('Tenure Months')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Monthly charges distribution by churn
        axes[0, 1].hist([df[df['Churn Value']==0]['Monthly Charges'], 
                        df[df['Churn Value']==1]['Monthly Charges']], 
                       bins=30, alpha=0.7, label=['No Churn', 'Churn'],
                       color=['skyblue', 'salmon'])
        axes[0, 1].set_title('Monthly Charges Distribution by Churn')
        axes[0, 1].set_xlabel('Monthly Charges')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Scatter plot: Tenure vs Monthly Charges colored by churn
        scatter = axes[1, 0].scatter(df['Tenure Months'], df['Monthly Charges'], 
                                   c=df['Churn Value'], cmap='RdYlBu', alpha=0.6)
        axes[1, 0].set_title('Tenure vs Monthly Charges (Colored by Churn)')
        axes[1, 0].set_xlabel('Tenure Months')
        axes[1, 0].set_ylabel('Monthly Charges')
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # CLTV vs Churn Score
        scatter2 = axes[1, 1].scatter(df['CLTV'], df['Churn Score'], 
                                    c=df['Churn Value'], cmap='RdYlBu', alpha=0.6)
        axes[1, 1].set_title('CLTV vs Churn Score (Colored by Churn)')
        axes[1, 1].set_xlabel('CLTV')
        axes[1, 1].set_ylabel('Churn Score')
        plt.colorbar(scatter2, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('static/plots/tenure_charges_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_eda(self, file_path):
        """Run complete EDA analysis"""
        self.create_plots_directory()
        
        df = self.preprocessor.load_data(file_path)
        df = self.preprocessor.clean_data(df)
        
        self.basic_statistics(df)
        
        self.plot_churn_distribution(df)
        self.plot_numerical_features(df)
        self.plot_categorical_features(df)
        self.plot_service_features(df)
        self.correlation_heatmap(df)
        correlations = self.feature_importance_analysis(df)
        self.plot_geographic_features(df)
        self.plot_tenure_and_charges_analysis(df)
        
        return df, correlations

if __name__ == "__main__":
    analyzer = EDAAnalyzer()
    df, correlations = analyzer.run_complete_eda('data/telco_churn.csv')
    print("EDA completed. Check the static/plots/ directory for visualizations.")