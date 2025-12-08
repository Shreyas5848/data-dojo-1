"""
Notebook Template Engine - Clean Version
Generates Jupyter notebooks from data profiling results
"""

import pandas as pd
from typing import Dict, Optional, Optional
from datetime import datetime
import nbformat as nbf


class NotebookTemplateEngine:
    """Generate Jupyter notebook templates from data profiling results."""
    
    def __init__(self):
        self.templates = {
            'exploratory_data_analysis': self._create_eda_template,
            'data_cleaning': self._create_cleaning_template,
            'classification_analysis': self._create_classification_template,
            'regression_analysis': self._create_regression_template,
            'time_series_analysis': self._create_timeseries_template,
            'clustering_analysis': self._create_clustering_template,
            'dimensionality_reduction': self._create_dimensionality_template,
            'feature_engineering': self._create_feature_engineering_template
        }
    
    def generate_notebook(self, profile_results: Dict, template_type: str, dataset_name: str = "dataset", dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Generate a notebook from profiling results.
        
        Args:
            profile_results: Data profiling results
            template_type: Type of template to generate
            dataset_name: Name of the dataset
            dataset_path: Path to the dataset file (relative or absolute)
        """
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        return self.templates[template_type](profile_results, dataset_name, dataset_path)
    
    def _create_eda_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Exploratory Data Analysis template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title
        cells.append(nbf.v4.new_markdown_cell(f"""# 📊 Exploratory Data Analysis: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Dataset:** {dataset_name}  
**Profile Results:** {profile_results.get('total_rows', 'Unknown')} rows, {profile_results.get('total_columns', 'Unknown')} columns

This notebook provides comprehensive analysis of your dataset based on DataDojo profiling results."""))
        
        # Imports
        cells.append(nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline"""))
        
        # Load data
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset
df = pd.read_csv('{data_load_path}')

print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"Columns: {list(df.columns)}")"""))
        
        # Overview
        cells.append(nbf.v4.new_markdown_cell("## 2. 🔍 Data Overview"))
        cells.append(nbf.v4.new_code_cell("""# Basic dataset information
print("=== DATASET OVERVIEW ===")
print(f"Shape: {df.shape}")
print(f"Data types:")
print(df.dtypes)

print("\\n=== MISSING VALUES ===")
missing_summary = df.isnull().sum()
print(missing_summary[missing_summary > 0])

print("\\n=== BASIC STATISTICS ===")
display(df.describe())"""))
        
        # Numeric analysis
        numeric_cols = profile_results.get('numeric_columns', [])
        if numeric_cols:
            cells.append(nbf.v4.new_markdown_cell("## 3. 📊 Numeric Columns Analysis"))
            # Create code without f-string conflicts
            code_lines = [
                f"# Analyze numeric columns: {numeric_cols}",
                f"numeric_cols = {numeric_cols}",
                "",
                "if len(numeric_cols) > 0:",
                '    print("=== NUMERIC STATISTICS ===")',
                "    display(df[numeric_cols].describe())",
                "    ",
                "    # Create distribution plots",
                "    n_cols = min(len(numeric_cols), 4)",
                "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))",
                "    axes = axes.ravel()",
                "    ",
                "    for i in range(n_cols):",
                "        col = numeric_cols[i]",
                "        df[col].hist(bins=30, ax=axes[i], alpha=0.7)",
                "        axes[i].set_title(f'Distribution of {col}')",
                "        axes[i].set_xlabel(col)",
                "    ",
                "    plt.tight_layout()",
                "    plt.show()"
            ]
            cells.append(nbf.v4.new_code_cell("\n".join(code_lines)))
        
        # Categorical analysis  
        categorical_cols = profile_results.get('categorical_columns', [])
        if categorical_cols:
            cells.append(nbf.v4.new_markdown_cell("## 4. 📋 Categorical Columns Analysis"))
            code_lines = [
                f"# Analyze categorical columns: {categorical_cols}",
                f"categorical_cols = {categorical_cols}",
                "",
                "if len(categorical_cols) > 0:",
                '    print("=== CATEGORICAL ANALYSIS ===")',
                "    ",
                "    for col in categorical_cols[:3]:  # Show first 3 columns",
                '        print(f"\\n{col}:")',
                "        print(df[col].value_counts().head())",
                "        ",
                "        # Create bar plot",
                "        plt.figure(figsize=(10, 6))",
                "        df[col].value_counts().head(10).plot(kind='bar')",
                "        plt.title(f'Top Values in {col}')",
                "        plt.xticks(rotation=45)",
                "        plt.tight_layout()",
                "        plt.show()"
            ]
            cells.append(nbf.v4.new_code_cell("\n".join(code_lines)))
        
        # Correlation
        if len(numeric_cols) > 1:
            cells.append(nbf.v4.new_markdown_cell("## 5. 🔗 Correlation Analysis"))
            cells.append(nbf.v4.new_code_cell("""# Correlation analysis for numeric columns
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) > 1:
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Find high correlations
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_val:.3f}")
    
    if high_corr:
        print("High correlations (|r| > 0.7):")
        for item in high_corr:
            print(item)"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 6. ✅ Summary & Next Steps"))
        quality_score = profile_results.get('overall_quality_score', 85.0)
        code_lines = [
            f"# Data quality assessment",
            f"quality_score = {quality_score}",
            "",
            'print("=== FINAL SUMMARY ===")',
            'print(f"Overall Quality Score: {quality_score}%")',
            'print(f"Dataset Shape: {df.shape}")',
            'print(f"Missing Values: {df.isnull().sum().sum()}")',
            'print(f"Duplicates: {df.duplicated().sum()}")',
            "",
            'print("\\n=== RECOMMENDED NEXT STEPS ===")',
            'print("1. Handle missing values (if any)")',
            'print("2. Check for outliers in numeric columns")',
            'print("3. Consider feature engineering")',
            'print("4. Select target variable for machine learning")',
            'print("5. Prepare data for modeling")'
        ]
        cells.append(nbf.v4.new_code_cell("\n".join(code_lines)))
        
        nb.cells = cells
        return nb
    
    def _create_cleaning_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Data Cleaning template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        cells.append(nbf.v4.new_markdown_cell(f"""# 🧹 Data Cleaning Workflow: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Purpose:** Systematic data cleaning and preparation

This notebook provides a comprehensive data cleaning workflow."""))
        
        cells.append(nbf.v4.new_code_cell(f"""# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('{data_load_path}')
df_original = df.copy()  # Keep backup

print(f"Original shape: {{df.shape}}")
print("Starting data cleaning workflow...")"""))
        
        # Missing values
        cells.append(nbf.v4.new_markdown_cell("## 1. 🕳️ Handle Missing Values"))
        cells.append(nbf.v4.new_code_cell("""# Check for missing values
missing_summary = df.isnull().sum()
missing_percentage = (missing_summary / len(df)) * 100

print("=== MISSING VALUES SUMMARY ===")
for col in missing_summary[missing_summary > 0].index:
    count = missing_summary[col]
    percentage = missing_percentage[col]
    print(f"{col}: {count} missing ({percentage:.1f}%)")

# Handle missing values
# Option 1: Drop columns with >50% missing
high_missing_cols = missing_percentage[missing_percentage > 50].index
if len(high_missing_cols) > 0:
    print(f"\\nDropping columns with >50% missing: {list(high_missing_cols)}")
    df = df.drop(columns=high_missing_cols)

# Option 2: Fill numeric columns with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Option 3: Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"\\nAfter cleaning - Missing values: {df.isnull().sum().sum()}")"""))
        
        # Duplicates
        cells.append(nbf.v4.new_markdown_cell("## 2. 🗑️ Remove Duplicates"))
        cells.append(nbf.v4.new_code_cell("""# Handle duplicates
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates()

print(f"Duplicates removed: {duplicates_before}")
print(f"New shape: {df.shape}")"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 3. ✅ Cleaning Summary"))
        cells.append(nbf.v4.new_code_cell("""# Final summary
print("=== CLEANING SUMMARY ===")
print(f"Original shape: {df_original.shape}")
print(f"Final shape: {df.shape}")
print(f"Rows removed: {len(df_original) - len(df)}")
print(f"Missing values remaining: {df.isnull().sum().sum()}")

# Save cleaned data
df.to_csv('cleaned_dataset.csv', index=False)
print("\\nCleaned dataset saved as 'cleaned_dataset.csv'")"""))
        
        nb.cells = cells
        return nb
    
    def _create_classification_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Classification template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title and overview
        cells.append(nbf.v4.new_markdown_cell(f"""# 🎯 Classification Analysis: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Classification Modeling  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides a complete classification modeling workflow including data exploration, preprocessing, model training, and evaluation.

## 📋 Workflow Steps
1. **Data Loading & Exploration**
2. **Target Variable Analysis** 
3. **Feature Engineering & Preprocessing**
4. **Model Training & Selection**
5. **Model Evaluation & Metrics**
6. **Feature Importance Analysis**
7. **Predictions & Results**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("✅ All libraries imported successfully!")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Initial Exploration"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset - REPLACE '{data_load_path}' with your actual file path
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {{df.shape}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")

print("\\n=== BASIC INFO ===")
print(df.info())

print("\\n=== FIRST 5 ROWS ===")
display(df.head())

print("\\n=== MISSING VALUES ===")
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(missing_data[missing_data > 0])
else:
    print("No missing values found!")
    
print("\\n=== DUPLICATE ROWS ===")
duplicates = df.duplicated().sum()
print(f"Duplicate rows: {duplicates}")"""))
        
        # Target variable analysis
        cells.append(nbf.v4.new_markdown_cell("## 2. 🎯 Target Variable Analysis"))
        cells.append(nbf.v4.new_code_cell("""# IMPORTANT: Define your target variable here
# REPLACE 'target_column' with your actual target column name
target_column = 'target_column'  # ⚠️ UPDATE THIS WITH YOUR TARGET COLUMN

# Check if target column exists
if target_column in df.columns:
    print(f"✅ Target variable found: {target_column}")
    
    # Analyze target distribution
    print("\\n=== TARGET DISTRIBUTION ===")
    target_counts = df[target_column].value_counts()
    print(target_counts)
    
    # Calculate class balance
    class_percentages = df[target_column].value_counts(normalize=True) * 100
    print("\\n=== CLASS PERCENTAGES ===")
    for class_name, percentage in class_percentages.items():
        print(f"{class_name}: {percentage:.2f}%")
    
    # Visualize target distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    target_counts.plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {target_column}')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
    plt.title(f'Proportion of {target_column}')
    
    plt.tight_layout()
    plt.show()
    
    # Check for class imbalance
    min_class_pct = class_percentages.min()
    if min_class_pct < 10:
        print(f"⚠️  WARNING: Class imbalance detected! Smallest class: {min_class_pct:.1f}%")
        print("Consider using techniques like SMOTE, class weights, or stratified sampling.")
    else:
        print("✅ Classes are reasonably balanced.")
        
else:
    print(f"❌ Column '{target_column}' not found!")
    print(f"Available columns: {list(df.columns)}")
    print("\\nPlease update the 'target_column' variable above.")"""))
        
        # Feature analysis
        cells.append(nbf.v4.new_markdown_cell("## 3. 📊 Feature Analysis & Preprocessing"))
        cells.append(nbf.v4.new_code_cell("""# Separate features and target (only if target column exists)
if target_column in df.columns:
    # Identify feature columns (exclude target and ID columns)
    id_columns = ['id', 'ID', 'index', 'customer_id', 'user_id']  # Add more ID columns if needed
    feature_columns = [col for col in df.columns 
                      if col != target_column and col not in id_columns]
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"✅ Features selected: {len(feature_columns)}")
    print(f"Feature columns: {feature_columns}")
    print(f"Target variable: {target_column}")
    
    # Analyze feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\\n📊 Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"📋 Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Check for missing values in features
    feature_missing = X.isnull().sum()
    if feature_missing.sum() > 0:
        print("\\n⚠️  Missing values in features:")
        print(feature_missing[feature_missing > 0])
    else:
        print("\\n✅ No missing values in features!")
        
else:
    print("❌ Please define the target column first!")"""))
        
        # Data preprocessing
        cells.append(nbf.v4.new_markdown_cell("## 4. 🔧 Data Preprocessing"))
        cells.append(nbf.v4.new_code_cell("""# Data preprocessing pipeline
if target_column in df.columns and 'X' in locals():
    
    # Handle missing values
    print("=== HANDLING MISSING VALUES ===")
    
    # For numeric features: fill with median
    if len(numeric_features) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
        print(f"✅ Filled missing values in numeric features with median")
    
    # For categorical features: fill with mode
    if len(categorical_features) > 0:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        print(f"✅ Filled missing values in categorical features with mode")
    
    # Encode categorical variables
    print("\\n=== ENCODING CATEGORICAL VARIABLES ===")
    label_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"✅ Encoded {col}: {len(le.classes_)} unique values")
    
    # Encode target variable if it's categorical
    target_encoder = None
    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
        print(f"\\n✅ Encoded target variable: {target_encoder.classes_}")
    
    # Feature scaling (for algorithms that need it)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print(f"\\n✅ Preprocessing completed!")
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
else:
    print("❌ Please complete previous steps first!")"""))
        
        # Train-test split
        cells.append(nbf.v4.new_markdown_cell("## 5. 🚂 Train-Test Split"))
        cells.append(nbf.v4.new_code_cell("""# Split data into training and testing sets
if 'X' in locals() and 'y' in locals():
    
    # Split with stratification to maintain class balance
    test_size = 0.2  # 80% train, 20% test
    random_state = 42
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    # Also create scaled versions
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled_df, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print("=== TRAIN-TEST SPLIT COMPLETED ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Check class distribution in splits
    print("\\n=== CLASS DISTRIBUTION ===")
    print("Training set:")
    print(pd.Series(y_train).value_counts(normalize=True).sort_index())
    
    print("\\nTest set:")
    print(pd.Series(y_test).value_counts(normalize=True).sort_index())
    
else:
    print("❌ Please complete preprocessing first!")"""))
        
        # Model training
        cells.append(nbf.v4.new_markdown_cell("## 6. 🤖 Model Training & Selection"))
        cells.append(nbf.v4.new_code_cell("""# Train multiple classification models
if 'X_train' in locals():
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(random_state=42, probability=True)
    }
    
    # Train and evaluate each model
    model_results = {}
    
    print("=== TRAINING MULTIPLE MODELS ===")
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        # Use scaled data for algorithms that need it
        if name in ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Machine']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Display results summary
    print("\\n=== MODEL COMPARISON SUMMARY ===")
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'Accuracy': [results['accuracy'] for results in model_results.values()],
        'Precision': [results['precision'] for results in model_results.values()],
        'Recall': [results['recall'] for results in model_results.values()],
        'F1-Score': [results['f1_score'] for results in model_results.values()]
    })
    
    results_df = results_df.sort_values('F1-Score', ascending=False)
    display(results_df)
    
    # Select best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = model_results[best_model_name]['model']
    best_predictions = model_results[best_model_name]['predictions']
    
    print(f"\\n🏆 BEST MODEL: {best_model_name}")
    
else:
    print("❌ Please complete train-test split first!")"""))
        
        # Model evaluation
        cells.append(nbf.v4.new_markdown_cell("## 7. 📈 Model Evaluation & Metrics"))
        cells.append(nbf.v4.new_code_cell("""# Detailed evaluation of the best model
if 'best_model' in locals():
    
    print(f"=== DETAILED EVALUATION: {best_model_name} ===")
    
    # Classification report
    print("\\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_test, best_predictions))
    
    # Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    # Confusion matrix heatmap
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix\\n{best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        plt.subplot(1, 3, 2)
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        top_features = importance_df.head(10)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
    
    # ROC Curve (for binary classification)
    if len(np.unique(y)) == 2 and model_results[best_model_name]['probabilities'] is not None:
        plt.subplot(1, 3, 3)
        fpr, tpr, _ = roc_curve(y_test, model_results[best_model_name]['probabilities'])
        auc_score = roc_auc_score(y_test, model_results[best_model_name]['probabilities'])
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    
    # Model performance metrics
    accuracy = accuracy_score(y_test, best_predictions)
    precision = precision_score(y_test, best_predictions, average='weighted')
    recall = recall_score(y_test, best_predictions, average='weighted')
    f1 = f1_score(y_test, best_predictions, average='weighted')
    
    print("\\n🎯 FINAL PERFORMANCE METRICS:")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    if len(np.unique(y)) == 2 and model_results[best_model_name]['probabilities'] is not None:
        auc = roc_auc_score(y_test, model_results[best_model_name]['probabilities'])
        print(f"AUC-ROC:   {auc:.4f}")
    
else:
    print("❌ Please complete model training first!")"""))
        
        # Feature importance analysis
        cells.append(nbf.v4.new_markdown_cell("## 8. 🔍 Feature Importance Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze feature importance and model insights
if 'best_model' in locals():
    
    print(f"=== FEATURE IMPORTANCE ANALYSIS ===")
    
    if hasattr(best_model, 'feature_importances_'):
        # Create detailed feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_,
            'Importance_Percentage': best_model.feature_importances_ * 100
        }).sort_values('Importance', ascending=False)
        
        print("\\n📊 TOP 15 MOST IMPORTANT FEATURES:")
        display(importance_df.head(15))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_20_features = importance_df.head(20)
        
        plt.barh(range(len(top_20_features)), top_20_features['Importance'])
        plt.yticks(range(len(top_20_features)), top_20_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        
        # Add percentage labels
        for i, v in enumerate(top_20_features['Importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance insights
        print("\\n💡 KEY INSIGHTS:")
        top_3_features = importance_df.head(3)
        total_importance = top_3_features['Importance'].sum()
        print(f"• Top 3 features account for {total_importance:.1%} of model decisions")
        
        for i, (_, row) in enumerate(top_3_features.iterrows(), 1):
            print(f"• #{i} Most important: '{row['Feature']}' ({row['Importance_Percentage']:.1f}%)")
            
    else:
        print(f"Feature importance not available for {best_model_name}")
        
        # For linear models, show coefficients
        if hasattr(best_model, 'coef_'):
            coef_df = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': best_model.coef_[0] if best_model.coef_.ndim > 1 else best_model.coef_
            })
            coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            print("\\n📊 TOP 10 FEATURES BY COEFFICIENT MAGNITUDE:")
            display(coef_df.head(10))
            
else:
    print("❌ Please complete model training first!")"""))
        
        # Predictions and results
        cells.append(nbf.v4.new_markdown_cell("## 9. 🎯 Predictions & Business Insights"))
        cells.append(nbf.v4.new_code_cell("""# Generate predictions and business insights
if 'best_model' in locals():
    
    print(f"=== PREDICTION ANALYSIS ===")
    
    # Create a results dataframe
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': best_predictions,
        'Correct': y_test == best_predictions
    })
    
    # Add original feature values for analysis
    test_indices = X_test.index
    results_df = results_df.merge(
        X.loc[test_indices], 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    # Prediction accuracy by class
    print("\\n📊 PREDICTION ACCURACY BY CLASS:")
    if target_encoder:
        class_accuracy = results_df.groupby('Actual')['Correct'].mean()
        for class_idx, accuracy in class_accuracy.items():
            class_name = target_encoder.classes_[class_idx]
            print(f"Class '{class_name}': {accuracy:.2%} correct predictions")
    else:
        class_accuracy = results_df.groupby('Actual')['Correct'].mean()
        for class_val, accuracy in class_accuracy.items():
            print(f"Class {class_val}: {accuracy:.2%} correct predictions")
    
    # Show some example predictions
    print("\\n🔍 SAMPLE PREDICTIONS:")
    sample_results = results_df.head(10)[['Actual', 'Predicted', 'Correct']]
    display(sample_results)
    
    # Misclassification analysis
    misclassified = results_df[results_df['Correct'] == False]
    if len(misclassified) > 0:
        print(f"\\n❌ MISCLASSIFIED EXAMPLES: {len(misclassified)} out of {len(results_df)}")
        
        # Show patterns in misclassifications
        print("\\nMost common misclassification patterns:")
        error_patterns = misclassified.groupby(['Actual', 'Predicted']).size().sort_values(ascending=False)
        print(error_patterns.head())
    
    print("\\n🎯 MODEL DEPLOYMENT READINESS:")
    print(f"✅ Model Type: {best_model_name}")
    print(f"✅ Overall Accuracy: {accuracy:.2%}")
    print(f"✅ Test Set Size: {len(y_test)} samples")
    print(f"✅ Features Used: {len(X.columns)}")
    
    if accuracy >= 0.8:
        print("🟢 HIGH ACCURACY: Model ready for deployment!")
    elif accuracy >= 0.7:
        print("🟡 MODERATE ACCURACY: Consider feature engineering or more data")
    else:
        print("🔴 LOW ACCURACY: Model needs significant improvement")
    
else:
    print("❌ Please complete model training first!")"""))
        
        # Summary and next steps
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Next Steps"))
        cells.append(nbf.v4.new_code_cell("""# Final summary and recommendations
print("=== 🎯 CLASSIFICATION ANALYSIS COMPLETE ===")
print()

if 'best_model' in locals():
    print("📊 ANALYSIS SUMMARY:")
    print(f"• Dataset: {dataset_name}")
    print(f"• Target Variable: {target_column}")
    print(f"• Best Model: {best_model_name}")
    print(f"• Final Accuracy: {accuracy:.2%}")
    print(f"• Features Used: {len(X.columns)}")
    print(f"• Training Samples: {len(X_train)}")
    print(f"• Test Samples: {len(X_test)}")
    
    print("\\n🚀 RECOMMENDED NEXT STEPS:")
    
    if accuracy >= 0.9:
        print("1. ✅ Excellent performance! Ready for production deployment")
        print("2. 🔄 Set up model monitoring and retraining pipeline")
        print("3. 📈 Consider A/B testing in production environment")
    elif accuracy >= 0.8:
        print("1. 🎯 Good performance! Consider hyperparameter tuning")
        print("2. 🔧 Try ensemble methods or advanced algorithms")
        print("3. 📊 Collect more data if possible")
    elif accuracy >= 0.7:
        print("1. 🔧 Feature engineering needed - create new features")
        print("2. 📊 Collect more training data")
        print("3. 🎯 Try different algorithms or ensemble methods")
        print("4. 🔍 Analyze and fix data quality issues")
    else:
        print("1. 🔍 Review data quality and target variable definition")
        print("2. 🎯 Significant feature engineering required")
        print("3. 📊 Consider if this is the right ML approach")
        print("4. 🤝 Consult domain experts for insights")
    
    print("\\n🛠️  TECHNICAL IMPROVEMENTS:")
    print("• Hyperparameter tuning with GridSearchCV or RandomSearchCV")
    print("• Cross-validation for more robust evaluation")
    print("• Feature selection techniques (RFE, SelectKBest)")
    print("• Handle class imbalance (SMOTE, class weights)")
    print("• Ensemble methods (Voting, Stacking)")
    print("• Deep learning approaches if dataset is large")
    
    print("\\n💾 SAVE YOUR MODEL:")
    print("# Uncomment to save the trained model")
    print("# import joblib")
    print("# joblib.dump(best_model, 'classification_model.pkl')")
    print("# print('Model saved successfully!')")
    
else:
    print("⚠️  Analysis incomplete. Please run all previous cells.")

print("\\n🎉 Classification analysis workflow completed!")"""))
        
        nb.cells = cells
        return nb
    
    def _create_regression_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Regression template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title and overview
        cells.append(nbf.v4.new_markdown_cell(f"""# 📈 Regression Analysis: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Regression Modeling  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides a complete regression modeling workflow to predict continuous numerical values.

## 📋 Workflow Steps
1. **Data Loading & Exploration**
2. **Target Variable Analysis**
3. **Feature Engineering & Preprocessing**
4. **Model Training & Comparison**
5. **Model Evaluation & Metrics**
6. **Residual Analysis**
7. **Predictions & Business Insights**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
%matplotlib inline

print("✅ All libraries imported successfully!")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Initial Exploration"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset - REPLACE '{data_load_path}' with your actual file path
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {{df.shape}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")

print("\\n=== DATA TYPES ===")
print(df.dtypes)

print("\\n=== FIRST 5 ROWS ===")
display(df.head())

print("\\n=== STATISTICAL SUMMARY ===")
display(df.describe())

print("\\n=== MISSING VALUES ===")
missing_data = df.isnull().sum()
if missing_data.sum() > 0:
    print(missing_data[missing_data > 0])
else:
    print("No missing values found!")"""))
        
        # Target variable analysis
        cells.append(nbf.v4.new_markdown_cell("## 2. 🎯 Target Variable Analysis"))
        cells.append(nbf.v4.new_code_cell("""# IMPORTANT: Define your target variable here
# REPLACE 'target_column' with your actual target column name
target_column = 'target_column'  # ⚠️ UPDATE THIS WITH YOUR TARGET COLUMN

# Check if target column exists
if target_column in df.columns:
    print(f"✅ Target variable found: {target_column}")
    
    # Target statistics
    print("\\n=== TARGET VARIABLE STATISTICS ===")
    print(f"Mean: {df[target_column].mean():.4f}")
    print(f"Median: {df[target_column].median():.4f}")
    print(f"Std Dev: {df[target_column].std():.4f}")
    print(f"Min: {df[target_column].min():.4f}")
    print(f"Max: {df[target_column].max():.4f}")
    print(f"Skewness: {df[target_column].skew():.4f}")
    print(f"Kurtosis: {df[target_column].kurtosis():.4f}")
    
    # Visualize target distribution
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(df[target_column], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(df[target_column].mean(), color='red', linestyle='--', label=f'Mean: {df[target_column].mean():.2f}')
    plt.axvline(df[target_column].median(), color='green', linestyle='--', label=f'Median: {df[target_column].median():.2f}')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {target_column}')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.boxplot(df[target_column].dropna())
    plt.ylabel(target_column)
    plt.title(f'Box Plot of {target_column}')
    
    plt.subplot(1, 3, 3)
    from scipy import stats
    stats.probplot(df[target_column].dropna(), dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Check)')
    
    plt.tight_layout()
    plt.show()
    
    # Check for outliers
    Q1 = df[target_column].quantile(0.25)
    Q3 = df[target_column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[target_column] < Q1 - 1.5*IQR) | (df[target_column] > Q3 + 1.5*IQR)).sum()
    print(f"\\n📊 Outliers detected (IQR method): {outliers} ({outliers/len(df)*100:.2f}%)")
    
    # Skewness check
    skewness = df[target_column].skew()
    if abs(skewness) > 1:
        print(f"⚠️  Target is highly skewed ({skewness:.2f}). Consider log transformation.")
    elif abs(skewness) > 0.5:
        print(f"📊 Target is moderately skewed ({skewness:.2f}).")
    else:
        print(f"✅ Target distribution is approximately normal ({skewness:.2f}).")
        
else:
    print(f"❌ Column '{target_column}' not found!")
    print(f"Available columns: {list(df.columns)}")
    print("\\nPlease update the 'target_column' variable above.")"""))
        
        # Feature correlation
        cells.append(nbf.v4.new_markdown_cell("## 3. 📊 Feature Analysis & Correlation"))
        cells.append(nbf.v4.new_code_cell("""# Analyze features and their correlation with target
if target_column in df.columns:
    
    # Identify feature columns (exclude target and ID columns)
    id_columns = ['id', 'ID', 'index', 'customer_id', 'user_id']
    feature_columns = [col for col in df.columns 
                      if col != target_column and col not in id_columns]
    
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    print(f"✅ Features selected: {len(feature_columns)}")
    
    # Analyze feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\\n📊 Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"📋 Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Correlation with target
    if len(numeric_features) > 0:
        print("\\n=== CORRELATION WITH TARGET ===")
        correlations = df[numeric_features + [target_column]].corr()[target_column].drop(target_column)
        correlations_sorted = correlations.abs().sort_values(ascending=False)
        
        print("\\nTop correlated features:")
        for feature in correlations_sorted.head(10).index:
            corr_value = correlations[feature]
            print(f"  {feature}: {corr_value:.4f}")
        
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        corr_matrix = df[numeric_features + [target_column]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Scatter plots with target
        top_features = correlations_sorted.head(4).index.tolist()
        if len(top_features) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(top_features[:4]):
                axes[i].scatter(df[feature], df[target_column], alpha=0.5)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel(target_column)
                axes[i].set_title(f'{feature} vs {target_column} (r={correlations[feature]:.3f})')
                
                # Add trend line
                z = np.polyfit(df[feature].dropna(), df[target_column].dropna(), 1)
                p = np.poly1d(z)
                axes[i].plot(df[feature].sort_values(), p(df[feature].sort_values()), 
                           "r--", alpha=0.8, label='Trend')
            
            plt.tight_layout()
            plt.show()
            
else:
    print("❌ Please define target column first!")"""))
        
        # Data preprocessing
        cells.append(nbf.v4.new_markdown_cell("## 4. 🔧 Data Preprocessing"))
        cells.append(nbf.v4.new_code_cell("""# Data preprocessing pipeline
if target_column in df.columns and 'X' in locals():
    
    # Handle missing values
    print("=== HANDLING MISSING VALUES ===")
    
    # For numeric features: fill with median
    if len(numeric_features) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        X[numeric_features] = numeric_imputer.fit_transform(X[numeric_features])
        print(f"✅ Filled missing values in numeric features with median")
    
    # For categorical features: fill with mode and encode
    if len(categorical_features) > 0:
        from sklearn.preprocessing import LabelEncoder
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            print(f"✅ Encoded {col}: {len(le.classes_)} unique values")
    
    # Handle missing values in target
    if y.isnull().sum() > 0:
        print(f"\\n⚠️  Dropping {y.isnull().sum()} rows with missing target values")
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print(f"\\n✅ Preprocessing completed!")
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
else:
    print("❌ Please complete previous steps first!")"""))
        
        # Train-test split
        cells.append(nbf.v4.new_markdown_cell("## 5. 🚂 Train-Test Split"))
        cells.append(nbf.v4.new_code_cell("""# Split data into training and testing sets
if 'X' in locals() and 'y' in locals():
    
    test_size = 0.2  # 80% train, 20% test
    random_state = 42
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Also create scaled versions
    X_train_scaled, X_test_scaled, _, _ = train_test_split(
        X_scaled_df, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print("=== TRAIN-TEST SPLIT COMPLETED ===")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    print("\\n=== TARGET DISTRIBUTION ===")
    print(f"Training - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    print(f"Test - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    
else:
    print("❌ Please complete preprocessing first!")"""))
        
        # Model training
        cells.append(nbf.v4.new_markdown_cell("## 6. 🤖 Model Training & Comparison"))
        cells.append(nbf.v4.new_code_cell("""# Train multiple regression models
if 'X_train' in locals():
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
        'Support Vector Regression': SVR(kernel='rbf')
    }
    
    # Train and evaluate each model
    model_results = {}
    
    print("=== TRAINING MULTIPLE MODELS ===")
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        # Use scaled data for algorithms that need it
        if name in ['Support Vector Regression', 'K-Nearest Neighbors']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_train_pred = model.predict(X_train_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_train_pred)
        
        # MAPE (handle division by zero)
        try:
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        except:
            mape = np.nan
        
        model_results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'r2_train': r2_train,
            'mape': mape,
            'predictions': y_pred
        }
        
        print(f"✅ {name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    # Display results summary
    print("\\n=== MODEL COMPARISON SUMMARY ===")
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'R² (Test)': [results['r2'] for results in model_results.values()],
        'R² (Train)': [results['r2_train'] for results in model_results.values()],
        'RMSE': [results['rmse'] for results in model_results.values()],
        'MAE': [results['mae'] for results in model_results.values()],
        'MAPE (%)': [results['mape'] for results in model_results.values()]
    })
    
    results_df = results_df.sort_values('R² (Test)', ascending=False)
    display(results_df)
    
    # Visualize model comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.barh(results_df['Model'], results_df['R² (Test)'])
    plt.xlabel('R² Score')
    plt.title('Model Comparison - R² Score')
    plt.xlim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.barh(results_df['Model'], results_df['RMSE'])
    plt.xlabel('RMSE')
    plt.title('Model Comparison - RMSE (lower is better)')
    
    plt.tight_layout()
    plt.show()
    
    # Select best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = model_results[best_model_name]['model']
    best_predictions = model_results[best_model_name]['predictions']
    
    print(f"\\n🏆 BEST MODEL: {best_model_name}")
    print(f"   R² Score: {model_results[best_model_name]['r2']:.4f}")
    print(f"   RMSE: {model_results[best_model_name]['rmse']:.4f}")
    
else:
    print("❌ Please complete train-test split first!")"""))
        
        # Model evaluation
        cells.append(nbf.v4.new_markdown_cell("## 7. 📈 Detailed Model Evaluation"))
        cells.append(nbf.v4.new_code_cell("""# Detailed evaluation of the best model
if 'best_model' in locals():
    
    print(f"=== DETAILED EVALUATION: {best_model_name} ===")
    
    # Get predictions
    y_pred = best_predictions
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\\n📊 REGRESSION METRICS:")
    print(f"R² Score:        {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"RMSE:            {rmse:.4f}")
    print(f"MAE:             {mae:.4f}")
    print(f"MSE:             {mse:.4f}")
    
    # Residual analysis
    residuals = y_test - y_pred
    
    plt.figure(figsize=(15, 10))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted\\nR² = {r2:.4f}')
    
    # Residuals vs Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    # Residual distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution\\nMean: {residuals.mean():.4f}, Std: {residuals.std():.4f}')
    
    # Q-Q plot for residuals
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Residual statistics
    print("\\n📊 RESIDUAL ANALYSIS:")
    print(f"Mean Residual:   {residuals.mean():.4f} (should be ~0)")
    print(f"Std Residual:    {residuals.std():.4f}")
    print(f"Min Residual:    {residuals.min():.4f}")
    print(f"Max Residual:    {residuals.max():.4f}")
    
    # Check for heteroscedasticity
    correlation = np.corrcoef(y_pred, np.abs(residuals))[0, 1]
    if abs(correlation) > 0.3:
        print(f"\\n⚠️  Potential heteroscedasticity detected (correlation: {correlation:.3f})")
    else:
        print(f"\\n✅ No significant heteroscedasticity (correlation: {correlation:.3f})")
        
else:
    print("❌ Please complete model training first!")"""))
        
        # Feature importance
        cells.append(nbf.v4.new_markdown_cell("## 8. 🔍 Feature Importance Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze feature importance
if 'best_model' in locals():
    
    print(f"=== FEATURE IMPORTANCE ANALYSIS ===")
    
    if hasattr(best_model, 'feature_importances_'):
        # Tree-based models
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_,
            'Importance_Percentage': best_model.feature_importances_ * 100
        }).sort_values('Importance', ascending=False)
        
        print("\\n📊 TOP 15 MOST IMPORTANT FEATURES:")
        display(importance_df.head(15))
        
        # Visualize
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 15 Feature Importance - {best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
    elif hasattr(best_model, 'coef_'):
        # Linear models
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': best_model.coef_,
            'Abs_Coefficient': np.abs(best_model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\\n📊 TOP 15 FEATURES BY COEFFICIENT MAGNITUDE:")
        display(coef_df.head(15))
        
        # Visualize
        plt.figure(figsize=(12, 8))
        top_features = coef_df.head(15)
        colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
        plt.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        print("\\n💡 Interpretation:")
        print("• GREEN bars: Positive effect on target")
        print("• RED bars: Negative effect on target")
        
    else:
        print(f"Feature importance not available for {best_model_name}")
        
else:
    print("❌ Please complete model training first!")"""))
        
        # Predictions analysis
        cells.append(nbf.v4.new_markdown_cell("## 9. 🎯 Predictions & Error Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze predictions and errors
if 'best_model' in locals():
    
    print(f"=== PREDICTION ANALYSIS ===")
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': best_predictions,
        'Residual': y_test.values - best_predictions,
        'Abs_Error': np.abs(y_test.values - best_predictions),
        'Pct_Error': np.abs(y_test.values - best_predictions) / np.abs(y_test.values) * 100
    })
    
    # Error statistics
    print("\\n📊 ERROR DISTRIBUTION:")
    print(f"Mean Absolute Error: {results_df['Abs_Error'].mean():.4f}")
    print(f"Median Absolute Error: {results_df['Abs_Error'].median():.4f}")
    print(f"90th Percentile Error: {results_df['Abs_Error'].quantile(0.9):.4f}")
    print(f"95th Percentile Error: {results_df['Abs_Error'].quantile(0.95):.4f}")
    
    # Sample predictions
    print("\\n🔍 SAMPLE PREDICTIONS:")
    sample = results_df.head(15).round(4)
    display(sample)
    
    # Best and worst predictions
    print("\\n✅ BEST PREDICTIONS (lowest error):")
    display(results_df.nsmallest(5, 'Abs_Error').round(4))
    
    print("\\n❌ WORST PREDICTIONS (highest error):")
    display(results_df.nlargest(5, 'Abs_Error').round(4))
    
    # Error distribution visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(results_df['Abs_Error'], bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(results_df['Abs_Error'].mean(), color='red', linestyle='--', 
                label=f'Mean: {results_df["Abs_Error"].mean():.2f}')
    plt.xlabel('Absolute Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Absolute Errors')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(results_df['Pct_Error'].clip(upper=100), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Percentage Errors')
    
    plt.tight_layout()
    plt.show()
    
else:
    print("❌ Please complete model training first!")"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Next Steps"))
        cells.append(nbf.v4.new_code_cell("""# Final summary and recommendations
print("=== 📈 REGRESSION ANALYSIS COMPLETE ===")
print()

if 'best_model' in locals():
    r2 = model_results[best_model_name]['r2']
    rmse = model_results[best_model_name]['rmse']
    mae = model_results[best_model_name]['mae']
    
    print("📊 ANALYSIS SUMMARY:")
    print(f"• Dataset: {dataset_name}")
    print(f"• Target Variable: {target_column}")
    print(f"• Best Model: {best_model_name}")
    print(f"• R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"• RMSE: {rmse:.4f}")
    print(f"• MAE: {mae:.4f}")
    print(f"• Features Used: {len(X.columns)}")
    print(f"• Training Samples: {len(X_train)}")
    print(f"• Test Samples: {len(X_test)}")
    
    print("\\n🚀 RECOMMENDED NEXT STEPS:")
    
    if r2 >= 0.9:
        print("1. ✅ Excellent model! Ready for production deployment")
        print("2. 🔄 Set up model monitoring and retraining pipeline")
        print("3. 📈 Consider A/B testing in production")
    elif r2 >= 0.7:
        print("1. 🎯 Good performance! Try hyperparameter tuning")
        print("2. 🔧 Consider polynomial features for non-linear relationships")
        print("3. 📊 Try ensemble methods (stacking, blending)")
    elif r2 >= 0.5:
        print("1. 🔧 Feature engineering needed")
        print("2. 📊 Look for non-linear relationships")
        print("3. 🎯 Consider more advanced algorithms (XGBoost, LightGBM)")
        print("4. 📈 Collect more relevant features")
    else:
        print("1. 🔍 Review data quality and target definition")
        print("2. 🎯 Significant feature engineering required")
        print("3. 📊 Consider if regression is the right approach")
        print("4. 🤝 Consult domain experts for insights")
    
    print("\\n🛠️  TECHNICAL IMPROVEMENTS:")
    print("• Hyperparameter tuning with GridSearchCV/RandomizedSearchCV")
    print("• Cross-validation for more robust evaluation")
    print("• Try XGBoost, LightGBM, or CatBoost")
    print("• Feature selection (RFE, SelectKBest)")
    print("• Polynomial features for non-linear relationships")
    print("• Log transformation if target is skewed")
    
    print("\\n💾 SAVE YOUR MODEL:")
    print("# Uncomment to save the trained model")
    print("# import joblib")
    print("# joblib.dump(best_model, 'regression_model.pkl')")
    print("# joblib.dump(scaler, 'scaler.pkl')")
    print("# print('Model saved successfully!')")
    
else:
    print("⚠️  Analysis incomplete. Please run all previous cells.")

print("\\n🎉 Regression analysis workflow completed!")"""))
        
        nb.cells = cells
        return nb
    
    def _create_timeseries_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Time Series template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title and overview
        cells.append(nbf.v4.new_markdown_cell(f"""# 📅 Time Series Analysis: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Time Series Modeling  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides a comprehensive time series analysis workflow including decomposition, stationarity testing, forecasting, and model evaluation.

## 📋 Workflow Steps
1. **Data Loading & Date Parsing**
2. **Time Series Visualization**
3. **Trend & Seasonality Decomposition**
4. **Stationarity Testing**
5. **Feature Engineering**
6. **Forecasting Models**
7. **Model Evaluation**
8. **Future Predictions**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series specific imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 6)
%matplotlib inline

print("✅ All libraries imported successfully!")
print("\\n📚 Available Time Series Methods:")
print("• Seasonal Decomposition")
print("• Stationarity Tests (ADF, KPSS)")
print("• Exponential Smoothing")
print("• Moving Averages")
print("• Prophet (if installed)")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Date Parsing"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Dataset shape: {{df.shape}}")
print(f"\\nColumn names: {{list(df.columns)}}")
print(f"\\nData types:\\n{{df.dtypes}}")

display(df.head(10))

# IMPORTANT: Define your date and value columns
# REPLACE these with your actual column names
date_column = 'date'      # ⚠️ UPDATE: Column containing dates/timestamps
value_column = 'value'    # ⚠️ UPDATE: Column containing values to forecast

print(f"\\n⚠️  Please update the following variables:")
print(f"   date_column = '{{date_column}}'")
print(f"   value_column = '{{value_column}}'")"""))
        
        # Parse dates
        cells.append(nbf.v4.new_code_cell("""# Parse dates and set as index
if date_column in df.columns and value_column in df.columns:
    
    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(date_column)
    
    # Set date as index
    df_ts = df.set_index(date_column)[[value_column]].copy()
    
    print("✅ Date parsing successful!")
    print(f"\\n=== TIME SERIES INFO ===")
    print(f"Start Date: {df_ts.index.min()}")
    print(f"End Date: {df_ts.index.max()}")
    print(f"Duration: {df_ts.index.max() - df_ts.index.min()}")
    print(f"Total observations: {len(df_ts)}")
    
    # Detect frequency
    if len(df_ts) > 1:
        time_diffs = pd.Series(df_ts.index).diff().dropna()
        most_common_diff = time_diffs.mode()[0]
        print(f"Detected frequency: {most_common_diff}")
    
    # Check for missing dates
    date_range = pd.date_range(start=df_ts.index.min(), end=df_ts.index.max(), freq='D')
    missing_dates = date_range.difference(df_ts.index)
    if len(missing_dates) > 0:
        print(f"\\n⚠️  Missing dates detected: {len(missing_dates)}")
    else:
        print(f"\\n✅ No missing dates in the series")
    
    display(df_ts.head())
    display(df_ts.describe())
    
else:
    print(f"❌ Columns not found!")
    print(f"Available columns: {list(df.columns)}")
    print("\\nPlease update date_column and value_column variables.")"""))
        
        # Time series visualization
        cells.append(nbf.v4.new_markdown_cell("## 2. 📊 Time Series Visualization"))
        cells.append(nbf.v4.new_code_cell("""# Visualize the time series
if 'df_ts' in locals():
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Original time series
    axes[0, 0].plot(df_ts.index, df_ts[value_column], linewidth=0.8)
    axes[0, 0].set_title(f'Time Series: {value_column}')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel(value_column)
    
    # Distribution
    axes[0, 1].hist(df_ts[value_column], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(df_ts[value_column].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_ts[value_column].mean():.2f}')
    axes[0, 1].set_title(f'Distribution of {value_column}')
    axes[0, 1].legend()
    
    # Rolling statistics
    window = min(30, len(df_ts) // 10)  # Adaptive window size
    rolling_mean = df_ts[value_column].rolling(window=window).mean()
    rolling_std = df_ts[value_column].rolling(window=window).std()
    
    axes[1, 0].plot(df_ts.index, df_ts[value_column], label='Original', alpha=0.5)
    axes[1, 0].plot(df_ts.index, rolling_mean, label=f'{window}-period Moving Avg', color='red')
    axes[1, 0].set_title('Time Series with Rolling Mean')
    axes[1, 0].legend()
    
    axes[1, 1].plot(df_ts.index, rolling_std, color='orange')
    axes[1, 1].set_title(f'{window}-period Rolling Standard Deviation')
    
    # Box plot by period (if enough data)
    if len(df_ts) > 365:
        df_ts['year'] = df_ts.index.year
        df_ts.boxplot(column=value_column, by='year', ax=axes[2, 0])
        axes[2, 0].set_title('Distribution by Year')
        df_ts.drop('year', axis=1, inplace=True)
    else:
        df_ts['month'] = df_ts.index.month
        df_ts.boxplot(column=value_column, by='month', ax=axes[2, 0])
        axes[2, 0].set_title('Distribution by Month')
        df_ts.drop('month', axis=1, inplace=True)
    
    # Lag plot
    pd.plotting.lag_plot(df_ts[value_column], lag=1, ax=axes[2, 1])
    axes[2, 1].set_title('Lag Plot (lag=1)')
    
    plt.tight_layout()
    plt.show()
    
    print("\\n📊 TIME SERIES STATISTICS:")
    print(f"Mean: {df_ts[value_column].mean():.4f}")
    print(f"Median: {df_ts[value_column].median():.4f}")
    print(f"Std Dev: {df_ts[value_column].std():.4f}")
    print(f"Min: {df_ts[value_column].min():.4f}")
    print(f"Max: {df_ts[value_column].max():.4f}")
    print(f"Skewness: {df_ts[value_column].skew():.4f}")
    
else:
    print("❌ Please complete data loading first!")"""))
        
        # Decomposition
        cells.append(nbf.v4.new_markdown_cell("## 3. 📈 Trend & Seasonality Decomposition"))
        cells.append(nbf.v4.new_code_cell("""# Decompose time series into components
if 'df_ts' in locals():
    
    # Determine period for decomposition
    # Adjust based on your data frequency
    period = 12  # Monthly seasonality (change to 7 for weekly, 365 for yearly, etc.)
    
    print(f"=== SEASONAL DECOMPOSITION (period={period}) ===")
    print("Adjust 'period' variable based on your data:")
    print("• Daily data with weekly pattern: period=7")
    print("• Monthly data with yearly pattern: period=12")
    print("• Hourly data with daily pattern: period=24")
    
    if len(df_ts) >= 2 * period:
        try:
            # Perform decomposition
            decomposition = seasonal_decompose(df_ts[value_column], model='additive', period=period)
            
            # Plot components
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            decomposition.observed.plot(ax=axes[0], title='Original Series')
            decomposition.trend.plot(ax=axes[1], title='Trend Component')
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component')
            decomposition.resid.plot(ax=axes[3], title='Residual Component')
            
            plt.tight_layout()
            plt.show()
            
            # Analyze components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            residual = decomposition.resid.dropna()
            
            print("\\n📊 COMPONENT ANALYSIS:")
            print(f"Trend range: {trend.min():.4f} to {trend.max():.4f}")
            print(f"Seasonal amplitude: {seasonal.max() - seasonal.min():.4f}")
            print(f"Residual std: {residual.std():.4f}")
            
            # Seasonal strength
            var_resid = residual.var()
            var_seasonal = seasonal.var()
            seasonal_strength = 1 - (var_resid / (var_resid + var_seasonal))
            print(f"\\nSeasonal strength: {seasonal_strength:.4f}")
            
            if seasonal_strength > 0.7:
                print("✅ Strong seasonality detected")
            elif seasonal_strength > 0.3:
                print("📊 Moderate seasonality detected")
            else:
                print("📉 Weak or no seasonality")
                
        except Exception as e:
            print(f"❌ Decomposition failed: {str(e)}")
            print("Try adjusting the 'period' parameter")
    else:
        print(f"⚠️  Need at least {2*period} observations for decomposition")
        print(f"Current observations: {len(df_ts)}")
        
else:
    print("❌ Please complete data loading first!")"""))
        
        # Stationarity testing
        cells.append(nbf.v4.new_markdown_cell("## 4. 🔍 Stationarity Testing"))
        cells.append(nbf.v4.new_code_cell("""# Test for stationarity
if 'df_ts' in locals():
    
    print("=== STATIONARITY TESTS ===")
    print("A stationary series has constant mean and variance over time.")
    print("Most forecasting models require stationary data.\\n")
    
    series = df_ts[value_column].dropna()
    
    # Augmented Dickey-Fuller Test
    print("📊 AUGMENTED DICKEY-FULLER (ADF) TEST:")
    adf_result = adfuller(series, autolag='AIC')
    
    print(f"   Test Statistic: {adf_result[0]:.4f}")
    print(f"   p-value: {adf_result[1]:.4f}")
    print(f"   Critical Values:")
    for key, value in adf_result[4].items():
        print(f"      {key}: {value:.4f}")
    
    if adf_result[1] < 0.05:
        print("   ✅ Result: Series is STATIONARY (reject null hypothesis)")
        adf_stationary = True
    else:
        print("   ⚠️  Result: Series is NON-STATIONARY (fail to reject null)")
        adf_stationary = False
    
    # KPSS Test
    print("\\n📊 KPSS TEST:")
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
        
        print(f"   Test Statistic: {kpss_result[0]:.4f}")
        print(f"   p-value: {kpss_result[1]:.4f}")
        print(f"   Critical Values:")
        for key, value in kpss_result[3].items():
            print(f"      {key}: {value:.4f}")
        
        if kpss_result[1] >= 0.05:
            print("   ✅ Result: Series is STATIONARY (fail to reject null)")
            kpss_stationary = True
        else:
            print("   ⚠️  Result: Series is NON-STATIONARY (reject null)")
            kpss_stationary = False
    except:
        kpss_stationary = None
        print("   ⚠️  KPSS test failed")
    
    # Summary
    print("\\n=== STATIONARITY SUMMARY ===")
    if adf_stationary:
        print("✅ Series appears to be stationary")
        print("Ready for modeling without differencing")
    else:
        print("⚠️  Series is non-stationary")
        print("Recommendation: Apply differencing or detrending")
        
        # Show differenced series
        print("\\n📊 FIRST DIFFERENCE:")
        diff_series = series.diff().dropna()
        adf_diff = adfuller(diff_series, autolag='AIC')
        print(f"   ADF p-value after differencing: {adf_diff[1]:.4f}")
        
        if adf_diff[1] < 0.05:
            print("   ✅ First difference is stationary")
        else:
            print("   ⚠️  May need second differencing")
            
else:
    print("❌ Please complete data loading first!")"""))
        
        # ACF and PACF
        cells.append(nbf.v4.new_markdown_cell("## 5. 📊 Autocorrelation Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze autocorrelation
if 'df_ts' in locals():
    
    series = df_ts[value_column].dropna()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ACF of original series
    plot_acf(series, lags=40, ax=axes[0, 0], title='ACF - Original Series')
    
    # PACF of original series
    plot_pacf(series, lags=40, ax=axes[0, 1], title='PACF - Original Series')
    
    # ACF of differenced series
    diff_series = series.diff().dropna()
    plot_acf(diff_series, lags=40, ax=axes[1, 0], title='ACF - First Difference')
    
    # PACF of differenced series
    plot_pacf(diff_series, lags=40, ax=axes[1, 1], title='PACF - First Difference')
    
    plt.tight_layout()
    plt.show()
    
    print("📊 ACF/PACF INTERPRETATION GUIDE:")
    print("• ACF shows correlation at different lags")
    print("• PACF shows direct correlation (removing intermediate effects)")
    print("• Significant spikes suggest important lags for modeling")
    print("• Gradual decay in ACF suggests AR process")
    print("• Sharp cutoff in ACF suggests MA process")
    
else:
    print("❌ Please complete data loading first!")"""))
        
        # Train-test split
        cells.append(nbf.v4.new_markdown_cell("## 6. 🚂 Train-Test Split"))
        cells.append(nbf.v4.new_code_cell("""# Split data for training and testing
if 'df_ts' in locals():
    
    # Use last 20% for testing
    test_size = 0.2
    split_idx = int(len(df_ts) * (1 - test_size))
    
    train = df_ts.iloc[:split_idx].copy()
    test = df_ts.iloc[split_idx:].copy()
    
    print("=== TRAIN-TEST SPLIT ===")
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(f"Testing period: {test.index.min()} to {test.index.max()}")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")
    
    # Visualize split
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train[value_column], label='Training', color='blue')
    plt.plot(test.index, test[value_column], label='Testing', color='orange')
    plt.axvline(x=train.index.max(), color='red', linestyle='--', label='Train/Test Split')
    plt.title('Train-Test Split')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    plt.show()
    
else:
    print("❌ Please complete data loading first!")"""))
        
        # Forecasting models
        cells.append(nbf.v4.new_markdown_cell("## 7. 🤖 Forecasting Models"))
        cells.append(nbf.v4.new_code_cell("""# Train multiple forecasting models
if 'train' in locals() and 'test' in locals():
    
    forecast_results = {}
    
    print("=== TRAINING FORECASTING MODELS ===\\n")
    
    # 1. Simple Moving Average
    print("📊 1. Simple Moving Average...")
    window = min(12, len(train) // 4)
    sma_forecast = train[value_column].rolling(window=window).mean().iloc[-1]
    sma_predictions = pd.Series([sma_forecast] * len(test), index=test.index)
    
    sma_rmse = np.sqrt(mean_squared_error(test[value_column], sma_predictions))
    sma_mae = mean_absolute_error(test[value_column], sma_predictions)
    
    forecast_results['Simple Moving Average'] = {
        'predictions': sma_predictions,
        'rmse': sma_rmse,
        'mae': sma_mae
    }
    print(f"   RMSE: {sma_rmse:.4f}, MAE: {sma_mae:.4f}")
    
    # 2. Exponential Smoothing (Simple)
    print("\\n📊 2. Simple Exponential Smoothing...")
    try:
        ses_model = ExponentialSmoothing(train[value_column], trend=None, seasonal=None)
        ses_fit = ses_model.fit()
        ses_predictions = ses_fit.forecast(len(test))
        ses_predictions.index = test.index
        
        ses_rmse = np.sqrt(mean_squared_error(test[value_column], ses_predictions))
        ses_mae = mean_absolute_error(test[value_column], ses_predictions)
        
        forecast_results['Simple Exp Smoothing'] = {
            'predictions': ses_predictions,
            'rmse': ses_rmse,
            'mae': ses_mae
        }
        print(f"   RMSE: {ses_rmse:.4f}, MAE: {ses_mae:.4f}")
    except Exception as e:
        print(f"   ⚠️  Failed: {str(e)}")
    
    # 3. Holt's Linear Trend
    print("\\n📊 3. Holt's Linear Trend...")
    try:
        holt_model = ExponentialSmoothing(train[value_column], trend='add', seasonal=None)
        holt_fit = holt_model.fit()
        holt_predictions = holt_fit.forecast(len(test))
        holt_predictions.index = test.index
        
        holt_rmse = np.sqrt(mean_squared_error(test[value_column], holt_predictions))
        holt_mae = mean_absolute_error(test[value_column], holt_predictions)
        
        forecast_results["Holt's Linear"] = {
            'predictions': holt_predictions,
            'rmse': holt_rmse,
            'mae': holt_mae
        }
        print(f"   RMSE: {holt_rmse:.4f}, MAE: {holt_mae:.4f}")
    except Exception as e:
        print(f"   ⚠️  Failed: {str(e)}")
    
    # 4. Holt-Winters (if enough data for seasonality)
    print("\\n📊 4. Holt-Winters Exponential Smoothing...")
    seasonal_period = 12  # Adjust based on your data
    
    if len(train) >= 2 * seasonal_period:
        try:
            hw_model = ExponentialSmoothing(
                train[value_column], 
                trend='add', 
                seasonal='add', 
                seasonal_periods=seasonal_period
            )
            hw_fit = hw_model.fit()
            hw_predictions = hw_fit.forecast(len(test))
            hw_predictions.index = test.index
            
            hw_rmse = np.sqrt(mean_squared_error(test[value_column], hw_predictions))
            hw_mae = mean_absolute_error(test[value_column], hw_predictions)
            
            forecast_results['Holt-Winters'] = {
                'predictions': hw_predictions,
                'rmse': hw_rmse,
                'mae': hw_mae
            }
            print(f"   RMSE: {hw_rmse:.4f}, MAE: {hw_mae:.4f}")
        except Exception as e:
            print(f"   ⚠️  Failed: {str(e)}")
    else:
        print(f"   ⚠️  Not enough data for seasonal model (need {2*seasonal_period} points)")
    
    # 5. Naive Forecast (baseline)
    print("\\n📊 5. Naive Forecast (baseline)...")
    naive_predictions = pd.Series([train[value_column].iloc[-1]] * len(test), index=test.index)
    
    naive_rmse = np.sqrt(mean_squared_error(test[value_column], naive_predictions))
    naive_mae = mean_absolute_error(test[value_column], naive_predictions)
    
    forecast_results['Naive (Baseline)'] = {
        'predictions': naive_predictions,
        'rmse': naive_rmse,
        'mae': naive_mae
    }
    print(f"   RMSE: {naive_rmse:.4f}, MAE: {naive_mae:.4f}")
    
    print("\\n✅ Model training completed!")
    
else:
    print("❌ Please complete train-test split first!")"""))
        
        # Model comparison
        cells.append(nbf.v4.new_markdown_cell("## 8. 📈 Model Evaluation & Comparison"))
        cells.append(nbf.v4.new_code_cell("""# Compare model performance
if 'forecast_results' in locals() and len(forecast_results) > 0:
    
    print("=== MODEL COMPARISON ===\\n")
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(forecast_results.keys()),
        'RMSE': [results['rmse'] for results in forecast_results.values()],
        'MAE': [results['mae'] for results in forecast_results.values()]
    })
    
    comparison_df = comparison_df.sort_values('RMSE')
    display(comparison_df)
    
    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_predictions = forecast_results[best_model_name]['predictions']
    
    print(f"\\n🏆 BEST MODEL: {best_model_name}")
    print(f"   RMSE: {forecast_results[best_model_name]['rmse']:.4f}")
    print(f"   MAE: {forecast_results[best_model_name]['mae']:.4f}")
    
    # Visualization
    plt.figure(figsize=(16, 10))
    
    # Plot 1: All forecasts comparison
    plt.subplot(2, 1, 1)
    plt.plot(train.index, train[value_column], label='Training', color='blue', alpha=0.7)
    plt.plot(test.index, test[value_column], label='Actual', color='black', linewidth=2)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(forecast_results)))
    for (name, results), color in zip(forecast_results.items(), colors):
        plt.plot(test.index, results['predictions'], label=f'{name}', linestyle='--', color=color)
    
    plt.axvline(x=train.index.max(), color='red', linestyle=':', alpha=0.5)
    plt.title('Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend(loc='best')
    
    # Plot 2: Best model detailed view
    plt.subplot(2, 1, 2)
    plt.plot(test.index, test[value_column], label='Actual', color='black', linewidth=2)
    plt.plot(test.index, best_predictions, label=f'{best_model_name} Forecast', 
             color='red', linestyle='--', linewidth=2)
    
    # Error bands
    residuals = test[value_column] - best_predictions
    std_resid = residuals.std()
    plt.fill_between(test.index, 
                     best_predictions - 1.96*std_resid, 
                     best_predictions + 1.96*std_resid,
                     alpha=0.2, color='red', label='95% Confidence Interval')
    
    plt.title(f'Best Model: {best_model_name}')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Residual analysis
    print("\\n📊 RESIDUAL ANALYSIS:")
    print(f"Mean Residual: {residuals.mean():.4f} (should be ~0)")
    print(f"Std Residual: {residuals.std():.4f}")
    
    # Calculate MAPE
    mape = np.mean(np.abs(residuals / test[value_column])) * 100
    print(f"MAPE: {mape:.2f}%")
    
    if mape < 10:
        print("\\n✅ Excellent forecast accuracy (MAPE < 10%)")
    elif mape < 20:
        print("\\n📊 Good forecast accuracy (MAPE < 20%)")
    elif mape < 30:
        print("\\n⚠️  Moderate forecast accuracy (MAPE < 30%)")
    else:
        print("\\n❌ Poor forecast accuracy (MAPE >= 30%)")
        
else:
    print("❌ Please complete model training first!")"""))
        
        # Future predictions
        cells.append(nbf.v4.new_markdown_cell("## 9. 🔮 Future Predictions"))
        cells.append(nbf.v4.new_code_cell("""# Generate future predictions
if 'best_model_name' in locals() and 'df_ts' in locals():
    
    # Number of periods to forecast
    forecast_periods = 30  # ⚠️ Adjust based on your needs
    
    print(f"=== GENERATING {forecast_periods}-PERIOD FORECAST ===\\n")
    
    # Retrain best model on full data
    full_series = df_ts[value_column]
    
    try:
        if best_model_name == 'Holt-Winters':
            final_model = ExponentialSmoothing(
                full_series, 
                trend='add', 
                seasonal='add', 
                seasonal_periods=12
            )
        elif best_model_name == "Holt's Linear":
            final_model = ExponentialSmoothing(full_series, trend='add', seasonal=None)
        else:
            final_model = ExponentialSmoothing(full_series, trend=None, seasonal=None)
        
        final_fit = final_model.fit()
        future_forecast = final_fit.forecast(forecast_periods)
        
        # Create future dates
        last_date = df_ts.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods)
        future_forecast.index = future_dates
        
        # Plot
        plt.figure(figsize=(14, 6))
        
        # Historical data (last 100 points)
        recent_data = df_ts.tail(min(100, len(df_ts)))
        plt.plot(recent_data.index, recent_data[value_column], label='Historical', color='blue')
        
        # Forecast
        plt.plot(future_forecast.index, future_forecast.values, label='Forecast', 
                color='red', linestyle='--', linewidth=2)
        
        # Confidence intervals
        std = full_series.std()
        plt.fill_between(future_forecast.index,
                        future_forecast.values - 1.96*std,
                        future_forecast.values + 1.96*std,
                        alpha=0.2, color='red', label='95% CI')
        
        plt.axvline(x=last_date, color='green', linestyle=':', label='Forecast Start')
        plt.title(f'{forecast_periods}-Period Forecast using {best_model_name}')
        plt.xlabel('Date')
        plt.ylabel(value_column)
        plt.legend()
        plt.show()
        
        # Display forecast values
        print("📊 FORECAST VALUES:")
        forecast_df = pd.DataFrame({
            'Date': future_forecast.index,
            'Forecast': future_forecast.values
        })
        display(forecast_df.head(15))
        
        print(f"\\n📈 Forecast Summary:")
        print(f"   Start: {future_forecast.iloc[0]:.4f}")
        print(f"   End: {future_forecast.iloc[-1]:.4f}")
        print(f"   Mean: {future_forecast.mean():.4f}")
        print(f"   Trend: {'📈 Increasing' if future_forecast.iloc[-1] > future_forecast.iloc[0] else '📉 Decreasing'}")
        
    except Exception as e:
        print(f"❌ Forecast generation failed: {str(e)}")
        
else:
    print("❌ Please complete model evaluation first!")"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Next Steps"))
        cells.append(nbf.v4.new_code_cell("""# Final summary
print("=== 📅 TIME SERIES ANALYSIS COMPLETE ===")
print()

if 'best_model_name' in locals():
    best_rmse = forecast_results[best_model_name]['rmse']
    best_mae = forecast_results[best_model_name]['mae']
    
    print("📊 ANALYSIS SUMMARY:")
    print(f"• Dataset: {dataset_name}")
    print(f"• Time Period: {df_ts.index.min()} to {df_ts.index.max()}")
    print(f"• Observations: {len(df_ts)}")
    print(f"• Best Model: {best_model_name}")
    print(f"• RMSE: {best_rmse:.4f}")
    print(f"• MAE: {best_mae:.4f}")
    
    print("\\n🚀 RECOMMENDED NEXT STEPS:")
    print("1. 🔧 Try ARIMA/SARIMA models for potentially better results")
    print("2. 📊 Experiment with different seasonal periods")
    print("3. 🎯 Add external regressors (holidays, events, etc.)")
    print("4. 🔄 Set up regular model retraining pipeline")
    print("5. 📈 Monitor forecast accuracy over time")
    
    print("\\n🛠️  ADVANCED TECHNIQUES TO EXPLORE:")
    print("• ARIMA/SARIMA models (statsmodels)")
    print("• Prophet (Facebook's forecasting library)")
    print("• LSTM neural networks (for complex patterns)")
    print("• XGBoost with time-based features")
    print("• Ensemble methods combining multiple models")
    
    print("\\n💾 SAVE YOUR MODEL:")
    print("# Uncomment to save")
    print("# import joblib")
    print("# joblib.dump(final_fit, 'timeseries_model.pkl')")
    
else:
    print("⚠️  Analysis incomplete. Please run all previous cells.")

print("\\n🎉 Time series analysis workflow completed!")"""))
        
        nb.cells = cells
        return nb

    def _create_clustering_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Clustering Analysis template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title
        cells.append(nbf.v4.new_markdown_cell(f"""# 🎯 Clustering Analysis: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Unsupervised Learning - Clustering  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides a comprehensive clustering workflow to discover natural groupings in your data using multiple algorithms and evaluation techniques.

## 📋 Workflow Steps
1. **Data Loading & Exploration**
2. **Data Preprocessing**
3. **Optimal Cluster Selection (Elbow Method)**
4. **K-Means Clustering**
5. **Hierarchical Clustering**
6. **DBSCAN Clustering**
7. **Cluster Evaluation & Comparison**
8. **Cluster Profiling**
9. **Visualization**
10. **Summary & Insights**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Visualization
from scipy.cluster.hierarchy import dendrogram, linkage

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
%matplotlib inline

print("✅ All libraries imported successfully!")
print("\\n📚 Available Clustering Algorithms:")
print("• K-Means")
print("• Hierarchical (Agglomerative)")
print("• DBSCAN")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Exploration"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset - REPLACE '{data_load_path}' with your actual file path
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print(f"\\nColumn names: {{list(df.columns)}}")
print(f"\\nData types:\\n{{df.dtypes}}")

display(df.head(10))
display(df.describe())

# Check for missing values
print("\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("✅ No missing values found!")"""))
        
        # Data preprocessing
        cells.append(nbf.v4.new_markdown_cell("## 2. 🔧 Data Preprocessing"))
        cells.append(nbf.v4.new_code_cell("""# Select numeric features for clustering
# IMPORTANT: Modify this list based on your data
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numeric columns available: {numeric_cols}")
print("\\n⚠️  Select features for clustering by updating 'features_for_clustering'")

# Select features - UPDATE THIS LIST
features_for_clustering = numeric_cols  # Or specify: ['col1', 'col2', 'col3']

# Create feature matrix
X = df[features_for_clustering].copy()

# Handle missing values
X = X.fillna(X.median())

print(f"\\nFeatures selected: {features_for_clustering}")
print(f"Feature matrix shape: {X.shape}")

# Scale features (IMPORTANT for clustering!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\\n✅ Features scaled using StandardScaler")
print(f"Scaled data mean: {X_scaled.mean():.4f} (should be ~0)")
print(f"Scaled data std: {X_scaled.std():.4f} (should be ~1)")"""))
        
        # Elbow method
        cells.append(nbf.v4.new_markdown_cell("## 3. 📊 Optimal Cluster Selection"))
        cells.append(nbf.v4.new_code_cell("""# Find optimal number of clusters using multiple methods
max_k = min(15, len(X) // 10)  # Max clusters to test
K_range = range(2, max_k + 1)

# Store metrics
inertias = []
silhouettes = []
calinski_scores = []
davies_bouldin_scores = []

print("=== EVALUATING CLUSTER NUMBERS ===")
print(f"Testing k = 2 to {max_k}...")

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

# Plot all metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Elbow Method
axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Inertia (Within-cluster SS)')
axes[0, 0].set_title('Elbow Method')
axes[0, 0].grid(True, alpha=0.3)

# Silhouette Score
axes[0, 1].plot(K_range, silhouettes, 'go-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('Silhouette Score (Higher is Better)')
axes[0, 1].grid(True, alpha=0.3)

# Calinski-Harabasz Index
axes[1, 0].plot(K_range, calinski_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('Calinski-Harabasz Index')
axes[1, 0].set_title('Calinski-Harabasz Index (Higher is Better)')
axes[1, 0].grid(True, alpha=0.3)

# Davies-Bouldin Index
axes[1, 1].plot(K_range, davies_bouldin_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Davies-Bouldin Index')
axes[1, 1].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find optimal k based on silhouette score
optimal_k = K_range[np.argmax(silhouettes)]
print(f"\\n🎯 RECOMMENDED: k = {optimal_k} (based on highest silhouette score)")
print(f"   Silhouette Score: {max(silhouettes):.4f}")"""))
        
        # K-Means
        cells.append(nbf.v4.new_markdown_cell("## 4. 🔵 K-Means Clustering"))
        cells.append(nbf.v4.new_code_cell("""# K-Means Clustering
n_clusters = optimal_k  # Use recommended k or change manually

print(f"=== K-MEANS CLUSTERING (k={n_clusters}) ===\\n")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Add labels to dataframe
df['KMeans_Cluster'] = kmeans_labels

# Evaluate
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
kmeans_davies = davies_bouldin_score(X_scaled, kmeans_labels)

print("📊 K-MEANS EVALUATION METRICS:")
print(f"   Silhouette Score: {kmeans_silhouette:.4f} (higher is better, max=1)")
print(f"   Calinski-Harabasz: {kmeans_calinski:.4f} (higher is better)")
print(f"   Davies-Bouldin: {kmeans_davies:.4f} (lower is better)")

# Cluster sizes
print(f"\\n📊 CLUSTER SIZES:")
cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()
for cluster, count in cluster_counts.items():
    pct = count / len(kmeans_labels) * 100
    print(f"   Cluster {cluster}: {count} samples ({pct:.1f}%)")

# Visualize with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', s=200, edgecolors='black', linewidths=2)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('K-Means Clusters (PCA Projection)')
plt.colorbar(scatter, label='Cluster')

plt.subplot(1, 2, 2)
cluster_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.title('Cluster Distribution')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()"""))
        
        # Hierarchical
        cells.append(nbf.v4.new_markdown_cell("## 5. 🌳 Hierarchical Clustering"))
        cells.append(nbf.v4.new_code_cell("""# Hierarchical (Agglomerative) Clustering
print(f"=== HIERARCHICAL CLUSTERING (k={n_clusters}) ===\\n")

# Create dendrogram (on sample if data is large)
sample_size = min(500, len(X_scaled))
X_sample = X_scaled[:sample_size]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
linkage_matrix = linkage(X_sample, method='ward')
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance')

# Fit Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X_scaled)

# Add to dataframe
df['Hierarchical_Cluster'] = hierarchical_labels

# Evaluate
hier_silhouette = silhouette_score(X_scaled, hierarchical_labels)
hier_calinski = calinski_harabasz_score(X_scaled, hierarchical_labels)
hier_davies = davies_bouldin_score(X_scaled, hierarchical_labels)

print("📊 HIERARCHICAL CLUSTERING METRICS:")
print(f"   Silhouette Score: {hier_silhouette:.4f}")
print(f"   Calinski-Harabasz: {hier_calinski:.4f}")
print(f"   Davies-Bouldin: {hier_davies:.4f}")

# Visualize
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Hierarchical Clusters (PCA Projection)')
plt.colorbar(scatter, label='Cluster')

plt.tight_layout()
plt.show()"""))
        
        # DBSCAN
        cells.append(nbf.v4.new_markdown_cell("## 6. 🔷 DBSCAN Clustering"))
        cells.append(nbf.v4.new_code_cell("""# DBSCAN - Density-Based Clustering
print("=== DBSCAN CLUSTERING ===\\n")

# Parameters - adjust based on your data
eps = 0.5  # Maximum distance between points
min_samples = 5  # Minimum samples in a neighborhood

# Find optimal eps using k-distance graph
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Sort distances for k-distance plot
distances = np.sort(distances[:, min_samples-1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(distances, linewidth=2)
plt.xlabel('Points (sorted by distance)')
plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
plt.title('K-Distance Graph (for choosing eps)')
plt.grid(True, alpha=0.3)

# Find "elbow" point
from scipy.ndimage import gaussian_filter1d
smooth = gaussian_filter1d(distances, sigma=5)
acceleration = np.diff(np.diff(smooth))
optimal_eps_idx = np.argmax(acceleration) + 2
suggested_eps = distances[optimal_eps_idx]
plt.axhline(y=suggested_eps, color='r', linestyle='--', label=f'Suggested eps: {suggested_eps:.2f}')
plt.legend()

print(f"💡 Suggested eps value: {suggested_eps:.2f}")
print("   (Adjust eps and min_samples if results aren't satisfactory)")

# Fit DBSCAN with suggested eps
dbscan = DBSCAN(eps=suggested_eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add to dataframe
df['DBSCAN_Cluster'] = dbscan_labels

# Statistics
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\\n📊 DBSCAN RESULTS:")
print(f"   Clusters found: {n_clusters_dbscan}")
print(f"   Noise points: {n_noise} ({n_noise/len(dbscan_labels)*100:.1f}%)")

# Evaluate (only if more than 1 cluster and not all noise)
if n_clusters_dbscan > 1 and n_noise < len(dbscan_labels):
    mask = dbscan_labels != -1
    if mask.sum() > 0:
        dbscan_silhouette = silhouette_score(X_scaled[mask], dbscan_labels[mask])
        print(f"   Silhouette Score (excl. noise): {dbscan_silhouette:.4f}")

# Visualize
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title(f'DBSCAN Clusters (eps={suggested_eps:.2f})')
plt.colorbar(scatter, label='Cluster (-1 = Noise)')

plt.tight_layout()
plt.show()"""))
        
        # Model comparison
        cells.append(nbf.v4.new_markdown_cell("## 7. 📊 Algorithm Comparison"))
        cells.append(nbf.v4.new_code_cell("""# Compare all clustering algorithms
print("=== CLUSTERING ALGORITHM COMPARISON ===\\n")

comparison_data = {
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
    'Silhouette': [kmeans_silhouette, hier_silhouette, 
                   silhouette_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if n_clusters_dbscan > 1 else 0],
    'Calinski-Harabasz': [kmeans_calinski, hier_calinski, 
                          calinski_harabasz_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if n_clusters_dbscan > 1 else 0],
    'Davies-Bouldin': [kmeans_davies, hier_davies, 
                       davies_bouldin_score(X_scaled[dbscan_labels != -1], dbscan_labels[dbscan_labels != -1]) if n_clusters_dbscan > 1 else float('inf')]
}

comparison_df = pd.DataFrame(comparison_data)
display(comparison_df)

# Best algorithm based on silhouette score
best_algo = comparison_df.loc[comparison_df['Silhouette'].idxmax(), 'Algorithm']
print(f"\\n🏆 BEST PERFORMING: {best_algo} (highest silhouette score)")

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# All three clusterings side by side
algorithms = ['KMeans_Cluster', 'Hierarchical_Cluster', 'DBSCAN_Cluster']
titles = ['K-Means', 'Hierarchical', 'DBSCAN']

for ax, algo, title in zip(axes, algorithms, titles):
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df[algo], cmap='viridis', alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()"""))
        
        # Cluster profiling
        cells.append(nbf.v4.new_markdown_cell("## 8. 📋 Cluster Profiling"))
        cells.append(nbf.v4.new_code_cell("""# Profile clusters (using K-Means as primary)
print("=== CLUSTER PROFILING (K-Means) ===\\n")

# Calculate cluster statistics for each feature
cluster_profiles = df.groupby('KMeans_Cluster')[features_for_clustering].agg(['mean', 'std'])
print("Cluster Centers (Mean ± Std):")
display(cluster_profiles)

# Simplified profile with means only
cluster_means = df.groupby('KMeans_Cluster')[features_for_clustering].mean()

# Visualize cluster profiles
plt.figure(figsize=(14, 6))

# Heatmap of cluster centers
plt.subplot(1, 2, 1)
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='RdYlBu_r', center=0)
plt.title('Cluster Profiles (Feature Means)')
plt.xlabel('Cluster')
plt.ylabel('Feature')

# Radar chart / parallel coordinates
plt.subplot(1, 2, 2)
# Normalize for comparison
cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
cluster_means_norm.T.plot(kind='bar', ax=plt.gca())
plt.title('Normalized Feature Comparison by Cluster')
plt.xlabel('Feature')
plt.ylabel('Normalized Value')
plt.legend(title='Cluster', bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Cluster descriptions
print("\\n📊 CLUSTER DESCRIPTIONS:")
overall_mean = df[features_for_clustering].mean()

for cluster in sorted(df['KMeans_Cluster'].unique()):
    cluster_data = df[df['KMeans_Cluster'] == cluster]
    print(f"\\n--- Cluster {cluster} ({len(cluster_data)} samples, {len(cluster_data)/len(df)*100:.1f}%) ---")
    
    for feature in features_for_clustering[:5]:  # Top 5 features
        cluster_mean = cluster_data[feature].mean()
        diff = (cluster_mean - overall_mean[feature]) / overall_mean[feature] * 100
        direction = "↑" if diff > 0 else "↓"
        print(f"   {feature}: {cluster_mean:.2f} ({direction} {abs(diff):.1f}% from avg)")"""))
        
        # Visualization
        cells.append(nbf.v4.new_markdown_cell("## 9. 📈 Advanced Visualization"))
        cells.append(nbf.v4.new_code_cell("""# Advanced cluster visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Cluster size distribution
axes[0, 0].pie(df['KMeans_Cluster'].value_counts().sort_index(), 
               labels=[f'Cluster {i}' for i in sorted(df['KMeans_Cluster'].unique())],
               autopct='%1.1f%%', colors=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
axes[0, 0].set_title('Cluster Size Distribution')

# 2. Box plots for top features by cluster
top_features = features_for_clustering[:4]  # Top 4 features
if len(top_features) >= 2:
    axes[0, 1].set_visible(False)
    gs = axes[0, 1].get_gridspec()
    
    for i, feature in enumerate(top_features[:2]):
        ax = fig.add_subplot(2, 4, i + 3)
        df.boxplot(column=feature, by='KMeans_Cluster', ax=ax)
        ax.set_title(feature)
        ax.set_xlabel('Cluster')
        plt.suptitle('')

# 3. Silhouette plot
from sklearn.metrics import silhouette_samples

silhouette_vals = silhouette_samples(X_scaled, kmeans_labels)
y_lower = 10

ax = axes[1, 0]
for i in range(n_clusters):
    cluster_silhouettes = silhouette_vals[kmeans_labels == i]
    cluster_silhouettes.sort()
    
    size_cluster = cluster_silhouettes.shape[0]
    y_upper = y_lower + size_cluster
    
    color = plt.cm.viridis(i / n_clusters)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouettes, 
                     facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
    y_lower = y_upper + 10

ax.axvline(x=kmeans_silhouette, color='red', linestyle='--', label=f'Avg: {kmeans_silhouette:.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster')
ax.set_title('Silhouette Plot')
ax.legend()

# 4. Feature importance for cluster separation
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, kmeans_labels)

feature_importance = pd.DataFrame({
    'Feature': features_for_clustering,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=True)

feature_importance.plot(kind='barh', x='Feature', y='Importance', ax=axes[1, 1], legend=False)
axes[1, 1].set_title('Feature Importance for Cluster Separation')
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.show()"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Next Steps"))
        cells.append(nbf.v4.new_code_cell("""# Final summary
print("=== 🎯 CLUSTERING ANALYSIS COMPLETE ===\\n")

print(f"📊 RESULTS SUMMARY:")
print(f"   Dataset: {dataset_name}")
print(f"   Samples: {len(df)}")
print(f"   Features used: {len(features_for_clustering)}")
print(f"   Optimal clusters (K-Means): {n_clusters}")
print(f"   Best algorithm: {best_algo}")
print(f"   Best silhouette score: {max(comparison_df['Silhouette']):.4f}")

print("\\n📋 CLUSTER ASSIGNMENTS SAVED:")
print("   • 'KMeans_Cluster' - K-Means labels")
print("   • 'Hierarchical_Cluster' - Hierarchical labels")
print("   • 'DBSCAN_Cluster' - DBSCAN labels")

print("\\n🚀 RECOMMENDED NEXT STEPS:")
print("1. 🔍 Analyze cluster profiles to understand segment characteristics")
print("2. 📊 Create business personas based on cluster insights")
print("3. 🎯 Develop targeted strategies for each cluster")
print("4. 🔄 Monitor cluster stability over time")
print("5. 📈 Test different feature combinations")

print("\\n💾 SAVE RESULTS:")
print("# Uncomment to save")
print("# df.to_csv('clustered_data.csv', index=False)")

print("\\n🎉 Clustering analysis workflow completed!")"""))
        
        nb.cells = cells
        return nb

    def _create_dimensionality_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Dimensionality Reduction template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title
        cells.append(nbf.v4.new_markdown_cell(f"""# 🔬 Dimensionality Reduction: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Dimensionality Reduction & Visualization  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides comprehensive dimensionality reduction techniques to visualize high-dimensional data, reduce noise, and discover underlying patterns.

## 📋 Workflow Steps
1. **Data Loading & Exploration**
2. **Data Preprocessing**
3. **PCA - Principal Component Analysis**
4. **Explained Variance Analysis**
5. **t-SNE Visualization**
6. **Feature Loadings Analysis**
7. **Dimensionality Reduction for ML**
8. **Comparison of Methods**
9. **Advanced Visualization**
10. **Summary & Recommendations**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
%matplotlib inline

print("✅ All libraries imported successfully!")
print("\\n📚 Available Dimensionality Reduction Methods:")
print("• PCA (Principal Component Analysis)")
print("• t-SNE (t-Distributed Stochastic Neighbor Embedding)")
print("• Feature Variance Analysis")
print("\\n💡 Tip: Install umap-learn for UMAP: pip install umap-learn")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Exploration"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset - REPLACE '{data_load_path}' with your actual file path
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print(f"\\nColumn names: {{list(df.columns)}}")
print(f"\\nData types:\\n{{df.dtypes}}")

display(df.head(10))

# Check dimensionality
n_features = len(df.select_dtypes(include=[np.number]).columns)
print(f"\\n📊 DIMENSIONALITY ASSESSMENT:")
print(f"   Total features: {{len(df.columns)}}")
print(f"   Numeric features: {{n_features}}")
print(f"   Categorical features: {{len(df.columns) - n_features}}")

if n_features > 10:
    print(f"\\n✅ High dimensionality detected - reduction recommended!")
elif n_features > 5:
    print(f"\\n📊 Moderate dimensionality - reduction may help visualization")
else:
    print(f"\\n💡 Low dimensionality - reduction mainly for visualization")"""))
        
        # Data preprocessing
        cells.append(nbf.v4.new_markdown_cell("## 2. 🔧 Data Preprocessing"))
        cells.append(nbf.v4.new_code_cell("""# Select numeric features for dimensionality reduction
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numeric columns: {numeric_cols}")

# Create feature matrix
X = df[numeric_cols].copy()

# Handle missing values
missing_before = X.isnull().sum().sum()
X = X.fillna(X.median())
print(f"\\nMissing values handled: {missing_before} → 0")

# Remove zero-variance features
print(f"\\nOriginal features: {X.shape[1]}")
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
selected_features = [col for col, selected in zip(numeric_cols, selector.get_support()) if selected]
print(f"After variance filter: {len(selected_features)} features")
print(f"Removed low-variance features: {set(numeric_cols) - set(selected_features)}")

# Update X with selected features
X = df[selected_features].fillna(df[selected_features].median())
feature_names = selected_features

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\\n✅ Preprocessing complete!")
print(f"   Final feature matrix: {X_scaled.shape}")"""))
        
        # PCA
        cells.append(nbf.v4.new_markdown_cell("## 3. 📊 PCA - Principal Component Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Perform PCA with all components first
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

print("=== PCA ANALYSIS ===\\n")
print(f"Original dimensions: {X_scaled.shape[1]}")
print(f"PCA components: {pca_full.n_components_}")

# Component summary
print(f"\\n📊 TOP COMPONENTS:")
for i in range(min(5, pca_full.n_components_)):
    print(f"   PC{i+1}: {pca_full.explained_variance_ratio_[i]*100:.2f}% variance explained")

# 2D Visualization
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.6, c='steelblue')
plt.xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('PCA: 2D Projection')
plt.grid(True, alpha=0.3)

# 3D Visualization (if enough components)
if pca_full.n_components_ >= 3:
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.scatter(X_pca_full[:, 0], X_pca_full[:, 1], X_pca_full[:, 2], alpha=0.6, c='steelblue')
    ax.set_xlabel(f'PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_zlabel(f'PC3 ({pca_full.explained_variance_ratio_[2]*100:.1f}%)')
    ax.set_title('PCA: 3D Projection')

plt.tight_layout()
plt.show()"""))
        
        # Explained variance
        cells.append(nbf.v4.new_markdown_cell("## 4. 📈 Explained Variance Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze explained variance
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find number of components for different variance thresholds
thresholds = [0.80, 0.90, 0.95, 0.99]
components_needed = {}
for thresh in thresholds:
    n_comp = np.argmax(cumulative_variance >= thresh) + 1
    components_needed[thresh] = n_comp

print("=== VARIANCE RETENTION ANALYSIS ===\\n")
for thresh, n_comp in components_needed.items():
    print(f"   {thresh*100:.0f}% variance → {n_comp} components (reduction: {X_scaled.shape[1]} → {n_comp})")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_) + 1), 
            pca_full.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Variance Explained by Each Component')

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-', linewidth=2)
axes[1].axhline(y=0.90, color='red', linestyle='--', label='90% threshold')
axes[1].axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
axes[1].fill_between(range(1, len(cumulative_variance) + 1), cumulative_variance, alpha=0.3)
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Recommendation
recommended_components = components_needed[0.95]
print(f"\\n🎯 RECOMMENDATION: Use {recommended_components} components to retain 95% variance")
print(f"   Dimensionality reduction: {X_scaled.shape[1]} → {recommended_components} ({(1-recommended_components/X_scaled.shape[1])*100:.1f}% reduction)")"""))
        
        # t-SNE
        cells.append(nbf.v4.new_markdown_cell("## 5. 🔮 t-SNE Visualization"))
        cells.append(nbf.v4.new_code_cell("""# t-SNE for visualization (works best with reduced dimensions first)
print("=== t-SNE VISUALIZATION ===\\n")

# Use PCA first if high dimensional (recommended for t-SNE)
if X_scaled.shape[1] > 50:
    print("High dimensionality detected - applying PCA first...")
    pca_pre = PCA(n_components=50)
    X_for_tsne = pca_pre.fit_transform(X_scaled)
else:
    X_for_tsne = X_scaled

# Sample if dataset is large (t-SNE is slow on large datasets)
max_samples = 5000
if len(X_for_tsne) > max_samples:
    print(f"Large dataset - sampling {max_samples} points for t-SNE...")
    sample_idx = np.random.choice(len(X_for_tsne), max_samples, replace=False)
    X_tsne_input = X_for_tsne[sample_idx]
else:
    X_tsne_input = X_for_tsne
    sample_idx = np.arange(len(X_for_tsne))

# Try different perplexity values
perplexities = [5, 30, 50]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, perp in zip(axes, perplexities):
    print(f"Running t-SNE with perplexity={perp}...")
    
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X_tsne_input)
    
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, c='steelblue', s=10)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(f't-SNE (perplexity={perp})')

plt.tight_layout()
plt.show()

print("\\n💡 t-SNE INTERPRETATION:")
print("• Perplexity affects local vs global structure preservation")
print("• Lower perplexity: more local structure, tighter clusters")
print("• Higher perplexity: more global structure, spread out")
print("• Clusters in t-SNE suggest groupings in high-dimensional space")"""))
        
        # Feature loadings
        cells.append(nbf.v4.new_markdown_cell("## 6. 🔍 Feature Loadings Analysis"))
        cells.append(nbf.v4.new_code_cell("""# Analyze which features contribute most to each component
print("=== FEATURE LOADINGS (PCA) ===\\n")

# Get loadings (components)
loadings = pd.DataFrame(
    pca_full.components_.T,
    columns=[f'PC{i+1}' for i in range(pca_full.n_components_)],
    index=feature_names
)

# Show top loadings for first 3 components
for i in range(min(3, pca_full.n_components_)):
    pc_name = f'PC{i+1}'
    print(f"\\n{pc_name} ({pca_full.explained_variance_ratio_[i]*100:.1f}% variance):")
    
    # Top positive loadings
    top_pos = loadings[pc_name].nlargest(3)
    print(f"   Top positive: {dict(top_pos.round(3))}")
    
    # Top negative loadings
    top_neg = loadings[pc_name].nsmallest(3)
    print(f"   Top negative: {dict(top_neg.round(3))}")

# Heatmap of loadings
plt.figure(figsize=(14, 8))

n_components_show = min(5, pca_full.n_components_)
loadings_subset = loadings.iloc[:, :n_components_show]

sns.heatmap(loadings_subset, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Feature Loadings for Top Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# Feature importance (sum of squared loadings)
feature_importance = (loadings.iloc[:, :n_components_show] ** 2).sum(axis=1).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='barh', color='steelblue')
plt.xlabel('Sum of Squared Loadings')
plt.ylabel('Feature')
plt.title('Feature Importance in PCA')
plt.tight_layout()
plt.show()

print("\\n💡 High loading = feature strongly influences that component")"""))
        
        # Dimensionality reduction for ML
        cells.append(nbf.v4.new_markdown_cell("## 7. 🤖 Dimensionality Reduction for ML"))
        cells.append(nbf.v4.new_code_cell("""# Create reduced datasets for machine learning
print("=== DIMENSIONALITY REDUCTION FOR ML ===\\n")

# Different reduction levels
reduction_configs = [
    ('90% variance', components_needed.get(0.90, 2)),
    ('95% variance', components_needed.get(0.95, 3)),
    ('99% variance', components_needed.get(0.99, 5)),
]

reduced_datasets = {}

for name, n_comp in reduction_configs:
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X_scaled)
    reduced_datasets[name] = X_reduced
    
    print(f"{name}: {X_scaled.shape[1]} → {n_comp} features")
    print(f"   Actual variance retained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    print()

# Create DataFrame with reduced features (using 95% variance)
optimal_n = components_needed.get(0.95, min(10, X_scaled.shape[1]))
pca_optimal = PCA(n_components=optimal_n)
X_reduced_optimal = pca_optimal.fit_transform(X_scaled)

# Add to dataframe
pca_columns = [f'PC{i+1}' for i in range(optimal_n)]
df_reduced = pd.DataFrame(X_reduced_optimal, columns=pca_columns, index=df.index)

print(f"✅ Reduced dataset created with {optimal_n} principal components")
display(df_reduced.head())

# Correlation between PCA components (should be zero!)
print("\\n📊 PCA Component Correlations (should be ~0):")
pca_corr = df_reduced.corr()
print(f"Max off-diagonal correlation: {pca_corr.where(~np.eye(len(pca_corr), dtype=bool)).abs().max().max():.6f}")"""))
        
        # Comparison
        cells.append(nbf.v4.new_markdown_cell("## 8. 📊 Comparison of Methods"))
        cells.append(nbf.v4.new_code_cell("""# Compare PCA vs t-SNE visualizations
print("=== PCA vs t-SNE COMPARISON ===\\n")

# Use sampled data for consistency
X_compare = X_scaled if len(X_scaled) <= 5000 else X_scaled[sample_idx]

# PCA 2D
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_compare)

# t-SNE 2D
tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_2d = tsne_2d.fit_transform(X_compare)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# PCA
axes[0].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.5, c='steelblue', s=10)
axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].set_title('PCA Projection')
axes[0].grid(True, alpha=0.3)

# t-SNE
axes[1].scatter(X_tsne_2d[:, 0], X_tsne_2d[:, 1], alpha=0.5, c='steelblue', s=10)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
axes[1].set_title('t-SNE Projection')

plt.tight_layout()
plt.show()

print("📊 METHOD COMPARISON:")
print()
print("PCA:")
print("   ✅ Linear transformation")
print("   ✅ Interpretable (feature loadings)")
print("   ✅ Fast and deterministic")
print("   ✅ Good for preprocessing & noise reduction")
print("   ❌ May miss non-linear patterns")
print()
print("t-SNE:")
print("   ✅ Non-linear transformation")
print("   ✅ Excellent for visualization")
print("   ✅ Preserves local structure well")
print("   ❌ Slow on large datasets")
print("   ❌ Not suitable for preprocessing")
print("   ❌ Results vary with parameters")"""))
        
        # Advanced visualization
        cells.append(nbf.v4.new_markdown_cell("## 9. 🎨 Advanced Visualization"))
        cells.append(nbf.v4.new_code_cell("""# Advanced visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Biplot (PCA with feature vectors)
ax = axes[0, 0]
ax.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.3, s=10)

# Add feature vectors
scale = 3  # Adjust for visibility
for i, feature in enumerate(feature_names):
    ax.arrow(0, 0, 
             loadings.iloc[i, 0] * scale,
             loadings.iloc[i, 1] * scale,
             head_width=0.1, head_length=0.05, fc='red', ec='red')
    ax.text(loadings.iloc[i, 0] * scale * 1.1,
            loadings.iloc[i, 1] * scale * 1.1,
            feature, fontsize=8)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Biplot')
ax.grid(True, alpha=0.3)

# 2. Scree plot
ax = axes[0, 1]
ax.bar(range(1, len(pca_full.explained_variance_) + 1), 
       pca_full.explained_variance_, alpha=0.7, color='steelblue', label='Eigenvalue')
ax.plot(range(1, len(pca_full.explained_variance_) + 1), 
        pca_full.explained_variance_, 'ro-', markersize=4)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Eigenvalue')
ax.set_title('Scree Plot')
ax.grid(True, alpha=0.3)

# 3. Pairwise PCA components
if pca_full.n_components_ >= 4:
    ax = axes[1, 0]
    scatter = ax.scatter(X_pca_full[:, 0], X_pca_full[:, 2], c=X_pca_full[:, 1], 
                         alpha=0.5, cmap='viridis', s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC3')
    ax.set_title('PC1 vs PC3 (colored by PC2)')
    plt.colorbar(scatter, ax=ax, label='PC2')

# 4. Component correlation with original features
ax = axes[1, 1]
# Show correlation of first 2 PCs with original features
pc_feature_corr = pd.DataFrame(X_pca_full[:, :2], columns=['PC1', 'PC2']).corrwith(
    pd.DataFrame(X_scaled, columns=feature_names)
).unstack()

if len(feature_names) <= 10:
    display_features = feature_names
else:
    # Show top features by correlation with PC1
    display_features = abs(loadings['PC1']).nlargest(10).index.tolist()

ax.barh(range(len(display_features)), 
        [loadings.loc[f, 'PC1'] for f in display_features], 
        alpha=0.7, label='PC1')
ax.barh([i + 0.3 for i in range(len(display_features))], 
        [loadings.loc[f, 'PC2'] for f in display_features], 
        alpha=0.7, label='PC2')
ax.set_yticks([i + 0.15 for i in range(len(display_features))])
ax.set_yticklabels(display_features)
ax.set_xlabel('Loading')
ax.set_title('Feature Loadings (PC1 & PC2)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Recommendations"))
        cells.append(nbf.v4.new_code_cell("""# Final summary
print("=== 🔬 DIMENSIONALITY REDUCTION COMPLETE ===\\n")

print(f"📊 ANALYSIS SUMMARY:")
print(f"   Dataset: {dataset_name}")
print(f"   Original dimensions: {X_scaled.shape[1]}")
print(f"   Samples: {len(X_scaled)}")

print(f"\\n📈 PCA RESULTS:")
print(f"   Components for 90% variance: {components_needed.get(0.90, 'N/A')}")
print(f"   Components for 95% variance: {components_needed.get(0.95, 'N/A')}")
print(f"   PC1 explains: {pca_full.explained_variance_ratio_[0]*100:.1f}%")
print(f"   PC1+PC2 explains: {sum(pca_full.explained_variance_ratio_[:2])*100:.1f}%")

print(f"\\n🔝 TOP CONTRIBUTING FEATURES:")
top_features = abs(loadings['PC1']).nlargest(3)
for feature, loading in top_features.items():
    print(f"   {feature}: {loading:.4f}")

print("\\n🚀 RECOMMENDATIONS:")
print(f"1. 📉 Use {components_needed.get(0.95, 'optimal')} PCA components for 95% variance")
print("2. 🔍 Use t-SNE/UMAP for visualization and clustering")
print("3. 🧹 Consider removing low-importance features")
print("4. 🤖 Use PCA-reduced data for faster ML training")

print("\\n💾 EXPORT REDUCED DATA:")
print("# Uncomment to save")
print("# df_reduced.to_csv('reduced_data.csv', index=False)")

print("\\n🎉 Dimensionality reduction workflow completed!")"""))
        
        nb.cells = cells
        return nb

    def _create_feature_engineering_template(self, profile_results: Dict, dataset_name: str, dataset_path: Optional[str] = None) -> nbf.NotebookNode:
        """Create Feature Engineering template."""
        nb = nbf.v4.new_notebook()
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
        
        cells = []
        
        # Determine the dataset path to use
        if dataset_path:
            data_load_path = dataset_path
        else:
            data_load_path = f"{dataset_name}.csv"  # Default fallback
        
        # Title
        cells.append(nbf.v4.new_markdown_cell(f"""# 🔧 Feature Engineering: {dataset_name}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Type:** Feature Engineering & Transformation  
**Dataset:** {dataset_name}

## 🎯 Objective
This notebook provides comprehensive feature engineering techniques to create, transform, and select the best features for machine learning models.

## 📋 Workflow Steps
1. **Data Loading & Exploration**
2. **Handling Missing Values**
3. **Encoding Categorical Variables**
4. **Numerical Transformations**
5. **Feature Creation**
6. **Feature Scaling**
7. **Feature Selection**
8. **Handling Imbalanced Features**
9. **Final Feature Set**
10. **Summary & Export**"""))
        
        # Import libraries
        cells.append(nbf.v4.new_code_cell("""# Import essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer

# Feature Selection
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
%matplotlib inline

print("✅ All libraries imported successfully!")
print("\\n📚 Feature Engineering Techniques Available:")
print("• Missing Value Imputation (Mean, Median, KNN)")
print("• Categorical Encoding (Label, One-Hot, Ordinal)")
print("• Numerical Transformations (Log, Power, Quantile)")
print("• Feature Scaling (Standard, MinMax, Robust)")
print("• Feature Selection (Filter, Wrapper, Embedded)")"""))
        
        # Data loading
        cells.append(nbf.v4.new_markdown_cell("## 1. 📁 Data Loading & Exploration"))
        cells.append(nbf.v4.new_code_cell(f"""# Load your dataset - REPLACE '{data_load_path}' with your actual file path
df = pd.read_csv('{data_load_path}')

print("=== DATASET OVERVIEW ===")
print(f"Shape: {{df.shape[0]}} rows × {{df.shape[1]}} columns")
print(f"\\nColumn names: {{list(df.columns)}}")

# Categorize columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

print(f"\\n📊 COLUMN TYPES:")
print(f"   Numeric: {{len(numeric_cols)}} - {{numeric_cols[:5]}}{{'...' if len(numeric_cols) > 5 else ''}}")
print(f"   Categorical: {{len(categorical_cols)}} - {{categorical_cols[:5]}}{{'...' if len(categorical_cols) > 5 else ''}}")
print(f"   Datetime: {{len(datetime_cols)}} - {{datetime_cols}}")

display(df.head())
display(df.describe())

# Store original for comparison
df_original = df.copy()
print("\\n✅ Original data stored for comparison")"""))
        
        # Missing values
        cells.append(nbf.v4.new_markdown_cell("## 2. 🔧 Handling Missing Values"))
        cells.append(nbf.v4.new_code_cell("""# Analyze missing values
print("=== MISSING VALUES ANALYSIS ===\\n")

missing_summary = pd.DataFrame({
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2),
    'Data Type': df.dtypes
}).sort_values('Missing %', ascending=False)

missing_summary = missing_summary[missing_summary['Missing Count'] > 0]

if len(missing_summary) > 0:
    display(missing_summary)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(missing_summary)), missing_summary['Missing %'], color='coral')
    plt.xticks(range(len(missing_summary)), missing_summary.index, rotation=45, ha='right')
    plt.ylabel('Missing %')
    plt.title('Missing Values by Column')
    plt.tight_layout()
    plt.show()
else:
    print("✅ No missing values found!")

# Imputation strategies
print("\\n📋 IMPUTATION STRATEGIES:")
print("• Numeric: Mean (normal dist), Median (skewed), KNN (complex patterns)")
print("• Categorical: Mode (most frequent), 'Unknown' category")
print("• High missing (>50%): Consider dropping column")"""))
        
        cells.append(nbf.v4.new_code_cell("""# Apply imputation
print("=== APPLYING IMPUTATION ===\\n")

# Numeric imputation
if len(numeric_cols) > 0:
    numeric_missing = [col for col in numeric_cols if df[col].isnull().sum() > 0]
    
    if numeric_missing:
        print(f"Imputing numeric columns: {numeric_missing}")
        
        # Choose strategy based on skewness
        for col in numeric_missing:
            skewness = df[col].skew()
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            if missing_pct > 50:
                print(f"   ⚠️  {col}: {missing_pct:.1f}% missing - consider dropping")
            elif abs(skewness) > 1:
                # Skewed - use median
                df[col].fillna(df[col].median(), inplace=True)
                print(f"   {col}: Median imputation (skewed: {skewness:.2f})")
            else:
                # Normal - use mean
                df[col].fillna(df[col].mean(), inplace=True)
                print(f"   {col}: Mean imputation (normal dist)")

# Categorical imputation
if len(categorical_cols) > 0:
    cat_missing = [col for col in categorical_cols if df[col].isnull().sum() > 0]
    
    if cat_missing:
        print(f"\\nImputing categorical columns: {cat_missing}")
        
        for col in cat_missing:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"   {col}: Mode imputation ('{mode_val}')")

# Verify
remaining_missing = df.isnull().sum().sum()
print(f"\\n✅ Imputation complete! Remaining missing: {remaining_missing}")"""))
        
        # Encoding
        cells.append(nbf.v4.new_markdown_cell("## 3. 🏷️ Encoding Categorical Variables"))
        cells.append(nbf.v4.new_code_cell("""# Analyze categorical columns
print("=== CATEGORICAL ENCODING ANALYSIS ===\\n")

if len(categorical_cols) > 0:
    encoding_plan = []
    
    for col in categorical_cols:
        n_unique = df[col].nunique()
        sample_values = df[col].unique()[:5]
        
        # Recommend encoding
        if n_unique == 2:
            recommendation = "Binary/Label Encoding"
        elif n_unique <= 10:
            recommendation = "One-Hot Encoding"
        elif n_unique <= 20:
            recommendation = "Target/Frequency Encoding"
        else:
            recommendation = "Target Encoding or Drop"
        
        encoding_plan.append({
            'Column': col,
            'Unique Values': n_unique,
            'Sample': str(sample_values[:3]),
            'Recommendation': recommendation
        })
    
    encoding_df = pd.DataFrame(encoding_plan)
    display(encoding_df)
else:
    print("No categorical columns to encode!")"""))
        
        cells.append(nbf.v4.new_code_cell("""# Apply encoding
print("=== APPLYING CATEGORICAL ENCODING ===\\n")

encoded_cols = []

for col in categorical_cols:
    n_unique = df[col].nunique()
    
    if n_unique == 2:
        # Binary encoding
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        encoded_cols.append(col + '_encoded')
        print(f"✅ {col}: Label Encoded → {col}_encoded")
        
    elif n_unique <= 10:
        # One-hot encoding
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        encoded_cols.extend(dummies.columns.tolist())
        print(f"✅ {col}: One-Hot Encoded → {len(dummies.columns)} new columns")
        
    else:
        # Frequency encoding
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col + '_freq'] = df[col].map(freq_map)
        encoded_cols.append(col + '_freq')
        print(f"✅ {col}: Frequency Encoded → {col}_freq")

print(f"\\n📊 Total encoded columns created: {len(encoded_cols)}")
print(f"New shape: {df.shape}")"""))
        
        # Numerical transformations
        cells.append(nbf.v4.new_markdown_cell("## 4. 📐 Numerical Transformations"))
        cells.append(nbf.v4.new_code_cell("""# Analyze distributions for transformation needs
print("=== NUMERICAL DISTRIBUTION ANALYSIS ===\\n")

# Get current numeric columns
current_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

# Analyze skewness
skewness_analysis = []
for col in current_numeric[:15]:  # Limit to first 15
    skew = df[col].skew()
    kurt = df[col].kurtosis()
    
    if abs(skew) > 2:
        recommendation = "Log/Power Transform"
    elif abs(skew) > 1:
        recommendation = "Sqrt Transform"
    else:
        recommendation = "No transform needed"
    
    skewness_analysis.append({
        'Column': col,
        'Skewness': round(skew, 2),
        'Kurtosis': round(kurt, 2),
        'Recommendation': recommendation
    })

skew_df = pd.DataFrame(skewness_analysis)
display(skew_df)

# Visualize top skewed columns
skewed_cols = [row['Column'] for row in skewness_analysis if abs(row['Skewness']) > 1][:4]

if skewed_cols:
    fig, axes = plt.subplots(2, len(skewed_cols), figsize=(4*len(skewed_cols), 8))
    
    for i, col in enumerate(skewed_cols):
        # Before
        axes[0, i].hist(df[col].dropna(), bins=50, alpha=0.7, color='coral')
        axes[0, i].set_title(f'{col}\\n(Original)')
        
        # After log transform
        positive_vals = df[col][df[col] > 0]
        if len(positive_vals) > 0:
            axes[1, i].hist(np.log1p(positive_vals), bins=50, alpha=0.7, color='steelblue')
            axes[1, i].set_title(f'{col}\\n(Log Transformed)')
    
    plt.tight_layout()
    plt.show()"""))
        
        cells.append(nbf.v4.new_code_cell("""# Apply transformations
print("=== APPLYING TRANSFORMATIONS ===\\n")

transformed_cols = []

for col in current_numeric:
    skew = df[col].skew()
    
    # Only transform highly skewed columns
    if abs(skew) > 1 and df[col].min() >= 0:
        # Log transform for positive values
        df[col + '_log'] = np.log1p(df[col])
        transformed_cols.append(col + '_log')
        
        new_skew = df[col + '_log'].skew()
        print(f"✅ {col}: Log transform (skew: {skew:.2f} → {new_skew:.2f})")

print(f"\\n📊 Transformed columns: {len(transformed_cols)}")

# Power Transform for severe skewness
print("\\n🔧 Applying Power Transform (Yeo-Johnson)...")
highly_skewed = [col for col in current_numeric if abs(df[col].skew()) > 2][:5]

if highly_skewed:
    pt = PowerTransformer(method='yeo-johnson')
    df_power = pd.DataFrame(
        pt.fit_transform(df[highly_skewed]),
        columns=[f'{col}_power' for col in highly_skewed],
        index=df.index
    )
    df = pd.concat([df, df_power], axis=1)
    print(f"   Power transformed: {highly_skewed}")

print(f"\\n✅ Transformation complete! New shape: {df.shape}")"""))
        
        # Feature creation
        cells.append(nbf.v4.new_markdown_cell("## 5. ✨ Feature Creation"))
        cells.append(nbf.v4.new_code_cell("""# Create new features from existing ones
print("=== FEATURE CREATION ===\\n")

created_features = []

# Get current numeric columns
current_numeric = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if not col.endswith(('_encoded', '_freq', '_log', '_power'))][:10]

print(f"Base numeric columns for feature creation: {current_numeric[:5]}...")

# 1. Interaction features (for top correlated pairs)
if len(current_numeric) >= 2:
    print("\\n📊 Creating Interaction Features...")
    
    # Calculate correlations
    corr_matrix = df[current_numeric].corr().abs()
    
    # Find top correlated pairs
    pairs = []
    for i in range(len(current_numeric)):
        for j in range(i+1, len(current_numeric)):
            pairs.append({
                'col1': current_numeric[i],
                'col2': current_numeric[j],
                'corr': corr_matrix.iloc[i, j]
            })
    
    top_pairs = sorted(pairs, key=lambda x: x['corr'], reverse=True)[:3]
    
    for pair in top_pairs:
        col1, col2 = pair['col1'], pair['col2']
        
        # Multiplication
        new_col = f"{col1}_x_{col2}"
        df[new_col] = df[col1] * df[col2]
        created_features.append(new_col)
        
        # Ratio (avoid division by zero)
        if (df[col2] != 0).all():
            ratio_col = f"{col1}_div_{col2}"
            df[ratio_col] = df[col1] / (df[col2] + 0.001)
            created_features.append(ratio_col)
        
        print(f"   ✅ {col1} × {col2} (correlation: {pair['corr']:.2f})")

# 2. Polynomial features
print("\\n📊 Creating Polynomial Features...")
if len(current_numeric) >= 2:
    top_cols = current_numeric[:3]
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(df[top_cols])
    
    # Get feature names
    poly_names = [f"poly_{i}" for i in range(poly_features.shape[1] - len(top_cols))]
    
    for i, name in enumerate(poly_names[:5]):
        df[name] = poly_features[:, len(top_cols) + i]
        created_features.append(name)
    
    print(f"   ✅ Created {len(poly_names[:5])} polynomial features")

# 3. Statistical aggregations
print("\\n📊 Creating Statistical Features...")
if len(current_numeric) >= 3:
    df['numeric_mean'] = df[current_numeric[:5]].mean(axis=1)
    df['numeric_std'] = df[current_numeric[:5]].std(axis=1)
    df['numeric_max'] = df[current_numeric[:5]].max(axis=1)
    df['numeric_min'] = df[current_numeric[:5]].min(axis=1)
    df['numeric_range'] = df['numeric_max'] - df['numeric_min']
    
    created_features.extend(['numeric_mean', 'numeric_std', 'numeric_max', 'numeric_min', 'numeric_range'])
    print(f"   ✅ Created 5 statistical aggregation features")

print(f"\\n📊 Total features created: {len(created_features)}")
print(f"New dataset shape: {df.shape}")"""))
        
        # Feature scaling
        cells.append(nbf.v4.new_markdown_cell("## 6. ⚖️ Feature Scaling"))
        cells.append(nbf.v4.new_code_cell("""# Compare different scaling methods
print("=== FEATURE SCALING ===\\n")

# Get all numeric columns for scaling
all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Numeric columns to scale: {len(all_numeric)}")

# Create scaled versions
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

# Show comparison on sample column
sample_col = all_numeric[0]
sample_data = df[[sample_col]].dropna()

print(f"\\n📊 Scaling Comparison on '{sample_col}':")
print(f"   Original - Mean: {sample_data[sample_col].mean():.2f}, Std: {sample_data[sample_col].std():.2f}")

for name, scaler in scalers.items():
    scaled = scaler.fit_transform(sample_data)
    print(f"   {name.capitalize()} - Mean: {scaled.mean():.2f}, Std: {scaled.std():.2f}")

# Apply StandardScaler (most common)
print("\\n🔧 Applying StandardScaler to all numeric features...")
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[all_numeric] = scaler.fit_transform(df[all_numeric].fillna(0))

print("✅ Scaling complete!")
print(f"\\n📋 SCALER SELECTION GUIDE:")
print("• StandardScaler: Best for normally distributed data, most ML algorithms")
print("• MinMaxScaler: Best for neural networks, bounded range [0, 1]")
print("• RobustScaler: Best for data with outliers (uses median/IQR)")"""))
        
        # Feature selection
        cells.append(nbf.v4.new_markdown_cell("## 7. 🎯 Feature Selection"))
        cells.append(nbf.v4.new_code_cell("""# Feature selection techniques
print("=== FEATURE SELECTION ===\\n")

# Get all numeric features
all_features = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[all_features].fillna(0)

print(f"Total features before selection: {len(all_features)}")

# 1. Variance Threshold
print("\\n📊 1. VARIANCE THRESHOLD:")
var_selector = VarianceThreshold(threshold=0.01)
var_selector.fit(X)
low_variance = [col for col, keep in zip(all_features, var_selector.get_support()) if not keep]
print(f"   Low variance features: {len(low_variance)}")
if low_variance:
    print(f"   Examples: {low_variance[:5]}")

# 2. Correlation Filter
print("\\n📊 2. CORRELATION FILTER:")
corr_matrix = X.corr().abs()

# Find highly correlated pairs
high_corr_pairs = []
for i in range(len(all_features)):
    for j in range(i+1, len(all_features)):
        if corr_matrix.iloc[i, j] > 0.95:
            high_corr_pairs.append((all_features[i], all_features[j], corr_matrix.iloc[i, j]))

print(f"   Highly correlated pairs (>0.95): {len(high_corr_pairs)}")
if high_corr_pairs:
    for pair in high_corr_pairs[:3]:
        print(f"   • {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")

# 3. Feature Importance (using Random Forest)
print("\\n📊 3. RANDOM FOREST IMPORTANCE:")

# Create a simple target for demonstration (or use your actual target)
# REPLACE with your actual target column!
if 'target' in df.columns:
    y = df['target']
else:
    # Create synthetic target for demonstration
    y = (X.mean(axis=1) > X.mean(axis=1).median()).astype(int)
    print("   ⚠️  Using synthetic target - replace with actual target!")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\\n   Top 10 Most Important Features:")
display(importance_df.head(10))

# Visualize
plt.figure(figsize=(12, 6))
top_20 = importance_df.head(20)
plt.barh(range(len(top_20)), top_20['Importance'].values, color='steelblue')
plt.yticks(range(len(top_20)), top_20['Feature'].values)
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""))
        
        cells.append(nbf.v4.new_code_cell("""# Select best features
print("=== SELECTING BEST FEATURES ===\\n")

# Method 1: Top K by importance
k = min(20, len(all_features))
top_k_features = importance_df.head(k)['Feature'].tolist()
print(f"📊 Top {k} features by importance: {top_k_features[:5]}...")

# Method 2: SelectKBest (statistical)
print("\\n📊 SelectKBest (F-statistic):")
selector = SelectKBest(score_func=f_classif, k=min(k, len(all_features)))
selector.fit(X, y)

scores_df = pd.DataFrame({
    'Feature': all_features,
    'F-Score': selector.scores_
}).sort_values('F-Score', ascending=False)

display(scores_df.head(10))

# Method 3: RFE (Recursive Feature Elimination)
print("\\n📊 RFE (selecting top 10 features):")
rfe_selector = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=10)
rfe_selector.fit(X, y)

rfe_features = [col for col, selected in zip(all_features, rfe_selector.support_) if selected]
print(f"   RFE selected: {rfe_features}")

# Consensus features (appear in multiple methods)
print("\\n🏆 CONSENSUS FEATURES (appear in top of multiple methods):")
top_importance = set(importance_df.head(15)['Feature'])
top_fstat = set(scores_df.head(15)['Feature'])
top_rfe = set(rfe_features)

consensus = top_importance.intersection(top_fstat)
print(f"   Features in both RF Importance & F-Score top 15:")
print(f"   {list(consensus)[:10]}")"""))
        
        # Handling imbalanced
        cells.append(nbf.v4.new_markdown_cell("## 8. ⚖️ Handling Imbalanced Features"))
        cells.append(nbf.v4.new_code_cell("""# Check for imbalanced categorical features
print("=== IMBALANCE ANALYSIS ===\\n")

# Check class balance for binary columns
binary_cols = [col for col in df.columns if df[col].nunique() == 2]

if binary_cols:
    print("📊 BINARY FEATURE BALANCE:")
    
    for col in binary_cols[:5]:
        value_counts = df[col].value_counts(normalize=True)
        imbalance_ratio = value_counts.min() / value_counts.max()
        
        status = "✅ Balanced" if imbalance_ratio > 0.3 else "⚠️  Imbalanced"
        print(f"   {col}: {status} (ratio: {imbalance_ratio:.2f})")
        
        if imbalance_ratio <= 0.3:
            print(f"      Distribution: {dict(value_counts.round(3))}")

# Check for rare categories
print("\\n📊 RARE CATEGORIES CHECK:")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

for col in categorical_cols[:5]:
    value_counts = df[col].value_counts(normalize=True)
    rare = value_counts[value_counts < 0.01]
    
    if len(rare) > 0:
        print(f"   {col}: {len(rare)} rare categories (<1% each)")

print("\\n💡 TIPS FOR IMBALANCED DATA:")
print("• Rare categories: Group into 'Other' category")
print("• Imbalanced target: Use SMOTE, class weights, or stratified sampling")
print("• Consider if imbalance reflects real-world distribution")"""))
        
        # Final feature set
        cells.append(nbf.v4.new_markdown_cell("## 9. 📦 Final Feature Set"))
        cells.append(nbf.v4.new_code_cell("""# Create final feature set
print("=== FINAL FEATURE SET ===\\n")

# Select features (customize based on your analysis)
# Option 1: Use top K by importance
final_features = importance_df.head(20)['Feature'].tolist()

# Option 2: Use consensus features
# final_features = list(consensus)

# Option 3: Manual selection
# final_features = ['feature1', 'feature2', ...]

print(f"Selected features: {len(final_features)}")
print(f"Features: {final_features[:10]}...")

# Create final dataset
df_final = df[final_features].copy()

# Apply final scaling
scaler_final = StandardScaler()
df_final_scaled = pd.DataFrame(
    scaler_final.fit_transform(df_final),
    columns=final_features,
    index=df_final.index
)

print(f"\\n📊 FINAL DATASET:")
print(f"   Shape: {df_final_scaled.shape}")
print(f"   Features: {len(final_features)}")
display(df_final_scaled.head())

# Correlation heatmap of final features
plt.figure(figsize=(12, 10))
sns.heatmap(df_final_scaled.corr(), annot=False, cmap='RdBu_r', center=0)
plt.title('Final Feature Correlation Matrix')
plt.tight_layout()
plt.show()"""))
        
        # Summary
        cells.append(nbf.v4.new_markdown_cell("## 10. ✅ Summary & Export"))
        cells.append(nbf.v4.new_code_cell("""# Final summary
print("=== 🔧 FEATURE ENGINEERING COMPLETE ===\\n")

print(f"📊 TRANSFORMATION SUMMARY:")
print(f"   Original dataset: {df_original.shape}")
print(f"   After engineering: {df.shape}")
print(f"   Final selected: {df_final_scaled.shape}")

print(f"\\n🔧 OPERATIONS PERFORMED:")
print(f"   • Missing values handled")
print(f"   • Categorical encoding applied")
print(f"   • Numerical transformations (log, power)")
print(f"   • Feature creation (interactions, polynomials, statistics)")
print(f"   • Feature selection (variance, correlation, importance)")
print(f"   • Scaling applied (StandardScaler)")

print(f"\\n🏆 TOP FEATURES BY IMPORTANCE:")
for i, row in importance_df.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.4f}")

print("\\n🚀 NEXT STEPS:")
print("1. 🧪 Test features with cross-validation")
print("2. 🔄 Iterate on feature creation based on model feedback")
print("3. 📊 Monitor feature drift in production")
print("4. 🔍 Consider domain-specific features")

print("\\n💾 EXPORT ENGINEERED DATA:")
print("# Uncomment to save")
print("# df_final_scaled.to_csv('engineered_features.csv', index=False)")
print("# import joblib")
print("# joblib.dump(scaler_final, 'feature_scaler.pkl')")

print("\\n🎉 Feature engineering workflow completed!")"""))
        
        nb.cells = cells
        return nb


def get_available_templates() -> Dict[str, str]:
    """Return available template types and descriptions."""
    return {
        'exploratory_data_analysis': '📊 Exploratory Data Analysis - Comprehensive data exploration',
        'data_cleaning': '🧹 Data Cleaning - Systematic data preparation workflow',
        'classification_analysis': '🎯 Classification Analysis - Binary/multi-class prediction', 
        'regression_analysis': '📈 Regression Analysis - Continuous value prediction',
        'time_series_analysis': '📅 Time Series Analysis - Temporal data modeling',
        'clustering_analysis': '🔮 Clustering Analysis - Unsupervised grouping & segmentation',
        'dimensionality_reduction': '🔬 Dimensionality Reduction - PCA, t-SNE visualization',
        'feature_engineering': '🔧 Feature Engineering - Create & select best features'
    }