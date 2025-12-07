"""
Project Notebook Generator for DataDojo
Generates educational Jupyter notebooks with guided learning, checkpoints, and progress tracking
"""

import nbformat as nbf
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import os


def generate_project_notebook(project, output_dir: str = None) -> str:
    """
    Generate an educational notebook for a learning project.
    
    Args:
        project: LearningProject instance
        output_dir: Directory to save the notebook (defaults to generated_notebooks/projects/)
    
    Returns:
        Path to the generated notebook
    """
    if output_dir is None:
        output_dir = Path("generated_notebooks") / "projects"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create notebook
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "datadojo": {
            "project_id": project.id,
            "project_name": project.name,
            "domain": project.domain.value,
            "difficulty": project.difficulty.value,
            "generated_at": datetime.now().isoformat()
        }
    }
    
    cells = []
    
    # === HEADER SECTION ===
    cells.append(nbf.v4.new_markdown_cell(f"""# 🎓 DataDojo Learning Project: {project.name}

<div style="background: linear-gradient(135deg, #1E293B, #0F172A); padding: 20px; border-radius: 10px; margin: 10px 0;">
<table style="width: 100%; border: none;">
<tr>
<td style="border: none;"><strong>📁 Domain:</strong> {project.domain.value.replace('_', ' ').title()}</td>
<td style="border: none;"><strong>📊 Difficulty:</strong> {project.difficulty.value.title()}</td>
<td style="border: none;"><strong>📅 Generated:</strong> {datetime.now().strftime('%Y-%m-%d')}</td>
</tr>
</table>
</div>

## 📋 Project Overview

{project.description}

---

## 🎯 Learning Objectives

By completing this project, you will learn to:

{chr(10).join(f"- ✅ **Objective {i+1}:** {obj}" for i, obj in enumerate(project.expected_outcomes))}

---

## 📖 How to Use This Notebook

1. **Read each section carefully** - Educational explanations are provided before each task
2. **Run the code cells** - Execute cells in order to see results
3. **Complete the exercises** - Fill in the `# YOUR CODE HERE` sections
4. **Check your progress** - Use the checkpoint cells to validate your work
5. **Track your learning** - Each completed section earns XP!

> 💡 **Tip:** Don't just run the code - understand WHY each step is important!
"""))
    
    # === SETUP SECTION ===
    cells.append(nbf.v4.new_markdown_cell("""---
## 🔧 Step 0: Environment Setup

First, let's import the necessary libraries and load our dataset.
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
plt.style.use('seaborn-v0_8-whitegrid')

print("✅ Libraries loaded successfully!")
"""))
    
    # === DATA LOADING SECTION ===
    dataset_path = project.dataset_path
    
    cells.append(nbf.v4.new_markdown_cell(f"""---
## 📁 Step 1: Load and Explore the Dataset

### 📚 Learning Concept: Data Loading

Before any analysis, we need to:
1. **Load the data** from a file (CSV, Excel, JSON, etc.)
2. **Inspect the structure** - rows, columns, data types
3. **Get a feel for the data** - look at sample rows

> 🎯 **Your Task:** Load the dataset and examine its basic properties.
"""))
    
    cells.append(nbf.v4.new_code_cell(f"""# Load the dataset
# Dataset path: {dataset_path}

df = pd.read_csv('{dataset_path}')

# Display basic information
print("=" * 60)
print("📊 DATASET OVERVIEW")
print("=" * 60)
print(f"📐 Shape: {{df.shape[0]:,}} rows × {{df.shape[1]}} columns")
print(f"💾 Memory: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")
print()
print("📋 Column Names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {{i}}. {{col}}")
"""))
    
    cells.append(nbf.v4.new_code_cell("""# View first few rows
print("\\n🔍 First 5 rows of the dataset:")
df.head()
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Check data types and missing values
print("\\n📊 Data Types and Missing Values:")
print("-" * 60)
info_df = pd.DataFrame({
    'Data Type': df.dtypes,
    'Non-Null Count': df.count(),
    'Missing Count': df.isnull().sum(),
    'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
})
print(info_df)
"""))
    
    # === CHECKPOINT 1 ===
    cells.append(nbf.v4.new_markdown_cell("""---
### ✅ Checkpoint 1: Data Loading Complete

**Questions to verify your understanding:**

1. How many rows and columns does the dataset have?
2. What are the data types present?
3. Are there any missing values?

> 🏆 **Progress:** +15 XP for completing Step 1!
"""))
    
    # === DATA QUALITY SECTION ===
    cells.append(nbf.v4.new_markdown_cell("""---
## 🔍 Step 2: Assess Data Quality

### 📚 Learning Concept: Data Quality Assessment

Data quality issues are common and must be identified before analysis:

| Issue | Description | Impact |
|-------|-------------|--------|
| **Missing Values** | Empty or null entries | Skewed analysis, errors |
| **Duplicates** | Repeated records | Inflated counts, bias |
| **Outliers** | Extreme values | Distorted statistics |
| **Inconsistencies** | Format/spelling variations | Grouping errors |

> 🎯 **Your Task:** Identify data quality issues in the dataset.
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Comprehensive data quality check
print("=" * 60)
print("🔍 DATA QUALITY REPORT")
print("=" * 60)

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print("\\n📉 Missing Values:")
for col in df.columns:
    if missing[col] > 0:
        print(f"   • {col}: {missing[col]:,} ({missing_pct[col]}%)")
if missing.sum() == 0:
    print("   ✅ No missing values found!")

# Duplicates
duplicates = df.duplicated().sum()
print(f"\\n🔄 Duplicate Rows: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")

# Numeric column statistics
print("\\n📊 Numeric Column Statistics:")
print(df.describe().T[['mean', 'std', 'min', 'max']])
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Visualize missing values
plt.figure(figsize=(12, 5))

# Missing values heatmap
plt.subplot(1, 2, 1)
missing_matrix = df.isnull().astype(int)
if missing_matrix.sum().sum() > 0:
    sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='Reds')
    plt.title('Missing Values Heatmap', fontsize=12, fontweight='bold')
else:
    plt.text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', fontsize=14)
    plt.title('Missing Values Check', fontsize=12, fontweight='bold')

# Data types distribution
plt.subplot(1, 2, 2)
type_counts = df.dtypes.astype(str).value_counts()
plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Data Types Distribution', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
"""))
    
    # === CHECKPOINT 2 ===
    cells.append(nbf.v4.new_markdown_cell("""---
### ✅ Checkpoint 2: Data Quality Assessment Complete

**What did you discover?**

Run the cell below to record your findings:
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Record your findings
# Fill in the values based on your analysis

findings = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "missing_values_total": df.isnull().sum().sum(),
    "duplicate_rows": df.duplicated().sum(),
    "columns_with_missing": df.columns[df.isnull().any()].tolist()
}

print("📝 YOUR FINDINGS:")
for key, value in findings.items():
    print(f"   • {key.replace('_', ' ').title()}: {value}")

print("\\n🏆 Progress: +15 XP for completing Step 2!")
"""))
    
    # === DATA CLEANING SECTION ===
    cells.append(nbf.v4.new_markdown_cell("""---
## 🧹 Step 3: Clean the Data

### 📚 Learning Concept: Data Cleaning Strategies

Different issues require different solutions:

| Issue | Strategy | Code Example |
|-------|----------|--------------|
| Missing (numeric) | Fill with mean/median | `df['col'].fillna(df['col'].median())` |
| Missing (categorical) | Fill with mode or 'Unknown' | `df['col'].fillna('Unknown')` |
| Duplicates | Remove duplicates | `df.drop_duplicates()` |
| Outliers | Cap/remove/transform | `df[df['col'] < upper_limit]` |

> 🎯 **Your Task:** Apply appropriate cleaning strategies to the dataset.
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Step 3.1: Handle Missing Values

# Create a copy to preserve original
df_clean = df.copy()

# Strategy: Fill numeric columns with median, categorical with mode
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype in ['int64', 'float64']:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"✅ Filled '{col}' missing values with median: {median_val:.2f}")
        else:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"✅ Filled '{col}' missing values with mode: {mode_val}")

print(f"\\n📊 Missing values after cleaning: {df_clean.isnull().sum().sum()}")
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Step 3.2: Remove Duplicates

initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
removed = initial_rows - len(df_clean)

print(f"🔄 Removed {removed:,} duplicate rows")
print(f"📊 Dataset now has {len(df_clean):,} rows")
"""))
    
    # === CHECKPOINT 3 ===
    cells.append(nbf.v4.new_markdown_cell("""---
### ✅ Checkpoint 3: Data Cleaning Complete

Let's verify the cleaning was successful:
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Verify cleaning results
print("=" * 60)
print("✅ DATA CLEANING VERIFICATION")
print("=" * 60)

print(f"\\n📐 Original shape: {df.shape}")
print(f"📐 Cleaned shape: {df_clean.shape}")
print(f"\\n📉 Missing values: {df_clean.isnull().sum().sum()}")
print(f"🔄 Duplicates: {df_clean.duplicated().sum()}")

if df_clean.isnull().sum().sum() == 0 and df_clean.duplicated().sum() == 0:
    print("\\n🎉 Data is clean and ready for analysis!")
    print("\\n🏆 Progress: +20 XP for completing Step 3!")
else:
    print("\\n⚠️ Some issues remain - review your cleaning steps.")
"""))
    
    # === EXPLORATORY ANALYSIS SECTION ===
    cells.append(nbf.v4.new_markdown_cell("""---
## 📊 Step 4: Exploratory Data Analysis (EDA)

### 📚 Learning Concept: EDA Fundamentals

EDA helps us understand patterns, relationships, and insights in data:

1. **Univariate Analysis** - Study one variable at a time
2. **Bivariate Analysis** - Study relationships between two variables
3. **Multivariate Analysis** - Study complex relationships

> 🎯 **Your Task:** Explore the data and discover interesting patterns.
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Univariate Analysis: Numeric columns
numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()

if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, min(3, len(numeric_cols)), figsize=(15, 8))
    axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
    
    for i, col in enumerate(numeric_cols[:6]):
        if i < len(axes):
            # Histogram
            df_clean[col].hist(ax=axes[i], bins=30, edgecolor='white', alpha=0.7)
            axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('')
    
    plt.suptitle('Distribution of Numeric Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("No numeric columns found for visualization")
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Univariate Analysis: Categorical columns
categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

if len(categorical_cols) > 0:
    fig, axes = plt.subplots(1, min(3, len(categorical_cols)), figsize=(15, 5))
    if len(categorical_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(categorical_cols[:3]):
        value_counts = df_clean[col].value_counts().head(10)
        value_counts.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='white')
        axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Distribution of Categorical Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("No categorical columns found for visualization")
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Correlation Analysis (for numeric columns)
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df_clean[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Find strong correlations
    print("\\n🔗 Strong Correlations (|r| > 0.5):")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"   • {correlation_matrix.columns[i]} ↔ {correlation_matrix.columns[j]}: {corr:.3f}")
else:
    print("Need at least 2 numeric columns for correlation analysis")
"""))
    
    # === CHECKPOINT 4 ===
    cells.append(nbf.v4.new_markdown_cell("""---
### ✅ Checkpoint 4: EDA Complete

**Key Insights Exercise:**

In the cell below, document at least 3 insights you discovered:
"""))
    
    cells.append(nbf.v4.new_code_cell("""# Document your insights
# YOUR CODE HERE: Replace the examples with your actual findings

insights = [
    "Insight 1: [Describe a pattern you found]",
    "Insight 2: [Describe a relationship you discovered]", 
    "Insight 3: [Describe an interesting distribution]"
]

print("📝 MY KEY INSIGHTS:")
print("=" * 50)
for i, insight in enumerate(insights, 1):
    print(f"\\n{i}. {insight}")

print("\\n🏆 Progress: +25 XP for completing Step 4!")
"""))
    
    # === PROJECT COMPLETION SECTION ===
    cells.append(nbf.v4.new_markdown_cell(f"""---
## 🎉 Project Completion

Congratulations on completing the **{project.name}** project!

### 📊 Summary of What You Learned:

{chr(10).join(f"✅ {obj}" for obj in project.expected_outcomes)}

### 🏆 XP Earned This Project:
- Step 1 (Data Loading): +15 XP
- Step 2 (Quality Assessment): +15 XP
- Step 3 (Data Cleaning): +20 XP
- Step 4 (EDA): +25 XP
- **Project Completion Bonus: +25 XP**

**Total: 100 XP** 🎯

---

### 🚀 Next Steps

1. **Save your cleaned dataset** for future use
2. **Try advanced analysis** techniques
3. **Move to the next project** in your learning path

Run the cell below to save your work and mark the project complete!
"""))
    
    cells.append(nbf.v4.new_code_cell(f"""# Save cleaned dataset
output_path = 'cleaned_{project.id}.csv'
df_clean.to_csv(output_path, index=False)
print(f"💾 Cleaned dataset saved to: {{output_path}}")

# Project completion summary
print("\\n" + "=" * 60)
print("🎉 PROJECT COMPLETE: {project.name}")
print("=" * 60)
print(f"\\n📊 Original dataset: {{len(df):,}} rows")
print(f"📊 Cleaned dataset: {{len(df_clean):,}} rows")
print(f"📊 Data quality: {{(1 - df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100:.1f}}%")
print("\\n🏆 Total XP Earned: 100 XP")
print("\\n👉 Return to DataDojo to mark this project complete!")
"""))
    
    # Add all cells to notebook
    nb.cells = cells
    
    # Generate filename
    filename = f"{project.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    filepath = output_dir / filename
    
    # Write notebook
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    return str(filepath)


def get_project_notebook_path(project_id: str) -> str:
    """Get the path where project notebook would be saved."""
    return str(Path("generated_notebooks") / "projects" / f"{project_id}_*.ipynb")
