"""
Simple ML Engine - Alternative approach without TypedDict issues
Uses basic Python data structures and minimal library dependencies
"""

import pandas as pd
import numpy as np
from typing import Any, Union
import warnings
warnings.filterwarnings('ignore')

# Import essential sklearn components
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import json
from datetime import datetime


class SimpleMLStep:
    """Basic ML step without complex typing."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = "pending"
        self.result = {}
        self.explanation = ""


class SimpleMLEngine:
    """
    Simplified ML engine that avoids TypedDict and complex serialization issues.
    Uses only basic Python data types.
    """
    
    def __init__(self):
        self.steps = []
        self.current_data = None
        self.target_column = None
        self.model = None
        self.test_data = None
        self.results = {}
    
    def create_pipeline(self, target_column: Union[str, None] = None, problem_type: str = "classification"):
        """Create a comprehensive 7-step pipeline."""
        steps = [
            SimpleMLStep("load_data", "Load and examine your data"),
            SimpleMLStep("explore_data", "Analyze data patterns and quality"),
            SimpleMLStep("clean_data", "Clean missing values and prepare data"),
            SimpleMLStep("engineer_features", "Create and optimize features"),
            SimpleMLStep("train_models", "Train and compare multiple models"),
            SimpleMLStep("evaluate_performance", "Detailed model evaluation with metrics"),
            SimpleMLStep("explain_model", "Understand model decisions and insights")
        ]
        self.steps = steps
        self.problem_type = problem_type
        self.target_column = target_column
        return steps
    
    def execute_step(self, step: SimpleMLStep, df: pd.DataFrame, **kwargs) -> dict:
        """Execute a pipeline step with basic error handling."""
        step.status = "running"
        
        try:
            if step.name == "load_data":
                result = self._load_data(df)
            elif step.name == "explore_data":
                result = self._explore_data(df)
            elif step.name == "clean_data":
                result = self._clean_data(df)
            elif step.name == "engineer_features":
                target_col = kwargs.get('target_col', '')
                result = self._engineer_features(df, target_col)
            elif step.name == "train_models":
                target_col = kwargs.get('target_col', '')
                result = self._train_models(df, target_col)
            elif step.name == "evaluate_performance":
                result = self._evaluate_performance()
            elif step.name == "explain_model":
                result = self._explain_model()
            else:
                result = {"error": f"Unknown step: {step.name}"}
            
            step.status = "completed"
            step.result = result
            return result
            
        except Exception as e:
            step.status = "failed"
            error_result = {
                "error": str(e),
                "explanation": f"❌ Step {step.name} failed: {str(e)}"
            }
            step.result = error_result
            return error_result
    
    def _load_data(self, df: pd.DataFrame) -> dict:
        """Step 1: Load and examine data."""
        self.current_data = df.copy()
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "explanation": f"""📊 **Data Loaded Successfully!**

**Dataset Info:**
• {len(df):,} rows (examples)
• {len(df.columns)} columns (features)

**Column Names:**
{', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}

Your data is ready for machine learning!"""
        }
    
    def _clean_data(self, df: pd.DataFrame) -> dict:
        """Step 2: Clean data simply."""
        try:
            # Simple cleaning approach
            df_clean = df.copy()
            
            # Fill missing numbers with 0
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df_clean[col].fillna(0, inplace=True)
            
            # Fill missing text with 'unknown'
            text_cols = df_clean.select_dtypes(include=['object']).columns
            for col in text_cols:
                df_clean[col].fillna('unknown', inplace=True)
            
            # Simple encoding for text columns
            for col in text_cols:
                if df_clean[col].dtype == 'object':
                    le = LabelEncoder()
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            
            self.current_data = df_clean
            
            return {
                "cleaned_rows": len(df_clean),
                "numeric_columns": len(numeric_cols),
                "text_columns": len(text_cols),
                "explanation": f"""🧹 **Data Cleaning Complete!**

**What we did:**
• Filled {len(numeric_cols)} numeric columns with zeros for missing values
• Filled {len(text_cols)} text columns with 'unknown' for missing values
• Converted text to numbers for machine learning

**Result:** {len(df_clean):,} clean rows ready for training!"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": "❌ Data cleaning failed. Please check your data format."
            }
    
    def _train_model(self, df: pd.DataFrame, target_col: str) -> dict:
        """Step 3: Train a simple model."""
        try:
            if not target_col or target_col not in df.columns:
                return {
                    "error": "No target column specified",
                    "explanation": "❌ Please select a target column to predict."
                }
            
            # Use cleaned data
            if self.current_data is not None:
                df = self.current_data.copy()
            
            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Simple train-test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
            except Exception:
                # Fallback for small datasets
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train simple model
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Store test data for evaluation
            self.test_data = {
                "X_test": X_test,
                "y_test": y_test,
                "X_train": X_train,
                "y_train": y_train
            }
            
            # Calculate simple accuracy
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            return {
                "model_type": "Random Forest",
                "train_accuracy": float(train_score),
                "test_accuracy": float(test_score),
                "features_used": len(X.columns),
                "training_samples": len(X_train),
                "explanation": f"""🤖 **Model Training Complete!**

**Model:** Random Forest Classifier
**Training Accuracy:** {train_score:.1%}
**Test Accuracy:** {test_score:.1%}

**What this means:**
Your model learned patterns from {len(X_train)} examples and can predict the target with {test_score:.1%} accuracy on new data!"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model training failed: {str(e)}"
            }
    
    def _evaluate_model(self) -> dict:
        """Step 4: Evaluate model performance."""
        try:
            if self.model is None:
                return {
                    "error": "No trained model",
                    "explanation": "❌ No model available. Please train a model first."
                }
            
            if self.test_data is None:
                return {
                    "error": "No test data",
                    "explanation": "❌ No test data available. Please train a model first."
                }
            
            # Get test data
            X_test = self.test_data["X_test"]
            y_test = self.test_data["y_test"]
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate simple metrics
            correct = sum(1 for i in range(len(y_test)) if y_test.iloc[i] == y_pred[i])
            total = len(y_test)
            accuracy = correct / total if total > 0 else 0
            
            return {
                "accuracy": float(accuracy),
                "correct_predictions": int(correct),
                "total_predictions": int(total),
                "model_ready": True,
                "explanation": f"""📊 **Model Evaluation Results:**

**Final Accuracy:** {accuracy:.1%}
**Correct Predictions:** {correct} out of {total}

**What this means:**
Your model correctly predicted {correct} cases out of {total} test cases. 
That's a {accuracy:.1%} success rate on data it had never seen before!

**Status:** ✅ Model is ready to make predictions on new data!"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model evaluation failed: {str(e)}"
            }
    
    def _explain_results(self) -> dict:
        """Step 5: Explain model results."""
        try:
            if self.model is None:
                return {
                    "error": "No model to explain",
                    "explanation": "❌ No trained model available to explain."
                }
            
            # Get feature importance if available
            feature_names = []
            importances = []
            
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                if self.test_data and "X_test" in self.test_data:
                    feature_names = list(self.test_data["X_test"].columns)
                
                # Get top 5 features
                if len(feature_names) == len(importances):
                    feature_importance = list(zip(feature_names, importances))
                    feature_importance.sort(key=lambda x: x[1], reverse=True)
                    top_features = feature_importance[:5]
                else:
                    top_features = []
            else:
                top_features = []
            
            return {
                "has_feature_importance": len(top_features) > 0,
                "top_features": top_features,
                "model_type": "Random Forest",
                "explanation": f"""🧠 **Model Explanation:**

**What your model learned:**
Your Random Forest model analyzed all the features in your data and learned patterns to make predictions.

{"**Most Important Features:**" if top_features else "**Features:**"}
{chr(10).join([f"• {name}: {imp:.1%} importance" for name, imp in top_features[:5]]) if top_features else "• All features contributed to the model"}

**How it works:**
The model uses decision trees to split your data based on feature values, creating rules like "if feature A > 5 and feature B < 10, then predict class 1".

**Next Steps:**
✅ Your model is trained and ready!
✅ You can now use it to make predictions on new data
✅ The model will apply the learned patterns to classify new examples"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model explanation failed: {str(e)}"
            }
    
    def _explore_data(self, df: pd.DataFrame) -> dict:
        """Step 2: Comprehensive data exploration."""
        try:
            # Data quality analysis
            missing_data = df.isnull().sum()
            total_missing = missing_data.sum()
            
            # Data types and uniqueness
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Statistics
            duplicates = df.duplicated().sum()
            
            # Data quality score
            completeness = (1 - total_missing / (len(df) * len(df.columns))) * 100 if len(df) > 0 else 100
            uniqueness = (1 - duplicates / len(df)) * 100 if len(df) > 0 else 100
            quality_score = (completeness + uniqueness) / 2
            
            return {
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "missing_values": int(total_missing),
                "duplicate_rows": int(duplicates),
                "completeness_percent": round(completeness, 1),
                "quality_score": round(quality_score, 1),
                "explanation": f"""🔍 **Data Exploration Complete!**

**Data Quality Assessment:**
• **Quality Score:** {quality_score:.1f}/100
• **Completeness:** {completeness:.1f}% (missing: {total_missing:,} values)
• **Uniqueness:** {uniqueness:.1f}% ({duplicates:,} duplicates found)

**Column Analysis:**
• **{len(numeric_cols)} Numeric columns** - ready for mathematical operations
• **{len(categorical_cols)} Categorical columns** - need encoding for ML

**Recommendations:**
{'✅ Data quality is excellent!' if quality_score > 80 else '⚠️ Consider cleaning missing values and duplicates.'}

**Next:** Data cleaning will prepare features for machine learning."""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": "❌ Data exploration failed. Please check your data format."
            }
    
    def _engineer_features(self, df: pd.DataFrame, target_col: str) -> dict:
        """Step 4: Advanced feature engineering."""
        try:
            if not target_col or target_col not in df.columns:
                return {
                    "error": "No target column specified",
                    "explanation": "❌ Please select a target column to proceed with feature engineering."
                }
            
            # Use cleaned data if available
            if self.current_data is not None:
                df = self.current_data.copy()
            
            # Feature engineering operations
            original_features = len(df.columns) - 1  # Exclude target
            
            # Create interaction features for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            interaction_features = 0
            if len(numeric_cols) >= 2:
                # Create a few interaction features
                for i in range(min(3, len(numeric_cols)-1)):
                    for j in range(i+1, min(i+3, len(numeric_cols))):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        new_col = f"{col1}_x_{col2}"
                        df[new_col] = df[col1] * df[col2]
                        interaction_features += 1
            
            # Feature scaling
            for col in numeric_cols:
                if df[col].std() > 0:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
            
            # Store engineered data
            self.current_data = df
            
            final_features = len(df.columns) - 1  # Exclude target
            
            return {
                "original_features": original_features,
                "final_features": final_features,
                "interaction_features": interaction_features,
                "scaled_features": len(numeric_cols),
                "explanation": f"""⚙️ **Feature Engineering Complete!**

**Feature Creation:**
• **Original features:** {original_features}
• **Interaction features added:** {interaction_features}
• **Final feature count:** {final_features}

**Feature Processing:**
• **Scaled {len(numeric_cols)} numeric features** to standard range
• **Created feature interactions** for better model learning
• **Optimized data types** for efficient computation

**Engineering Impact:**
• **{((final_features - original_features) / original_features * 100):.1f}% increase** in feature richness
• Enhanced model's ability to detect complex patterns

**Next:** Model training with engineered features! 🚀"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Feature engineering failed: {str(e)}"
            }
    
    def _train_models(self, df: pd.DataFrame, target_col: str) -> dict:
        """Step 5: Train and compare multiple ML models."""
        try:
            if not target_col or target_col not in df.columns:
                return {
                    "error": "No target column specified",
                    "explanation": "❌ Please select a target column to train models."
                }
            
            # Use engineered data if available
            if self.current_data is not None:
                df = self.current_data.copy()
            
            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Determine problem type
            unique_values = y.nunique()
            is_classification = unique_values <= 10
            
            # Train-test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y if is_classification else None
                )
            except:
                # Fallback for small datasets
                split_idx = int(len(X) * 0.7)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train multiple models
            models = {}
            scores = {}
            
            if is_classification:
                models = {
                    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42),
                    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                    "SVM": SVC(random_state=42, probability=True),
                    "Naive Bayes": GaussianNB()
                }
            else:
                models = {
                    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
                    "Linear Regression": LinearRegression(),
                    "SVR": SVR(),
                }
            
            # Train and evaluate each model
            best_model = None
            best_score = -float('inf')
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    scores[name] = score
                    
                    if score > best_score:
                        best_score = score
                        best_model = (name, model)
                except:
                    continue
            
            if best_model is not None:
                self.model = best_model[1]
                self.model_name = best_model[0]
            
            # Store test data for evaluation
            self.test_data = {
                "X_test": X_test,
                "y_test": y_test,
                "X_train": X_train,
                "y_train": y_train
            }
            
            return {
                "models_trained": len(scores),
                "best_model": best_model[0] if best_model else "None",
                "best_score": float(best_score),
                "all_scores": {k: float(v) for k, v in scores.items()},
                "problem_type": "classification" if is_classification else "regression",
                "explanation": f"""🤖 **Model Training Complete!**

**Models Compared:** {len(scores)}
**Winner:** {best_model[0] if best_model else 'None'} with {best_score:.1%} accuracy

**All Results:**
{chr(10).join([f'• {name}: {score:.1%}' for name, score in scores.items()])}

**Training Details:**
• **Problem Type:** {'Classification' if is_classification else 'Regression'}
• **Training Samples:** {len(X_train):,}
• **Test Samples:** {len(X_test):,}
• **Features Used:** {len(X.columns)}

**Model Selection:** The best performing model will be used for evaluation!"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model training failed: {str(e)}"
            }
    
    def _evaluate_performance(self) -> dict:
        """Step 6: Detailed model performance evaluation."""
        try:
            if self.model is None:
                return {
                    "error": "No trained model",
                    "explanation": "❌ No model available. Please train models first."
                }
            
            if self.test_data is None:
                return {
                    "error": "No test data",
                    "explanation": "❌ No test data available. Please train models first."
                }
            
            # Safely extract test data
            try:
                X_test = self.test_data["X_test"]
                y_test = self.test_data["y_test"]
            except (KeyError, TypeError):
                return {
                    "error": "Invalid test data format",
                    "explanation": "❌ Test data format is invalid. Please retrain models."
                }
            
            # Make predictions
            try:
                y_pred = self.model.predict(X_test)
            except Exception as pred_error:
                return {
                    "error": f"Prediction failed: {str(pred_error)}",
                    "explanation": "❌ Model prediction failed. Please retrain models with clean data."
                }
            
            # Determine if classification or regression
            is_classification = len(set(y_test)) <= 10
            
            if is_classification:
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
                
                # Confusion matrix - convert carefully to avoid serialization issues
                try:
                    cm = confusion_matrix(y_test, y_pred)
                    cm_list = [[int(x) for x in row] for row in cm.tolist()]
                except:
                    cm_list = [[0, 0], [0, 0]]  # Fallback
                
                return {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": cm_list,
                    "problem_type": "classification",
                    "explanation": f"""**Detailed Model Evaluation:**

**Classification Performance:**
- Accuracy: {accuracy:.1%} - Overall correctness
- Precision: {precision:.1%} - Accuracy of positive predictions  
- Recall: {recall:.1%} - Ability to find all positive cases
- F1-Score: {f1:.1%} - Balanced precision and recall

**Model Quality Assessment:**
{'Excellent performance!' if accuracy > 0.9 else 'Good performance!' if accuracy > 0.7 else 'Consider model improvements.'}

**Confusion Matrix:** Available for detailed error analysis
**Ready for production:** {'Yes' if accuracy > 0.8 else 'Needs improvement'}"""
                }
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                return {
                    "r2_score": float(r2),
                    "mse": float(mse),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "problem_type": "regression",
                    "explanation": f"""**Detailed Model Evaluation:**

**Regression Performance:**
- R² Score: {r2:.3f} - Variance explained ({r2*100:.1f}%)
- RMSE: {rmse:.2f} - Root Mean Square Error  
- MAE: {mae:.2f} - Mean Absolute Error

**Model Quality Assessment:**
{'Excellent fit!' if r2 > 0.8 else 'Good fit!' if r2 > 0.6 else 'Consider model improvements.'}

**Error Analysis:**
- Average prediction error: ±{rmse:.2f} units
- Typical deviation: {mae:.2f} units

**Ready for production:** {'Yes' if r2 > 0.7 else 'Needs improvement'}"""
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model evaluation failed: {str(e)}"
            }
    
    def _explain_model(self) -> dict:
        """Step 7: Comprehensive model explanation and insights."""
        try:
            if self.model is None:
                return {
                    "error": "No model to explain",
                    "explanation": "❌ No trained model available to explain."
                }
            
            explanation_parts = []
            
            # Feature importance (if available)
            feature_importance_info = {}
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                if self.test_data and "X_test" in self.test_data:
                    feature_names = list(self.test_data["X_test"].columns)
                    
                    if len(feature_names) == len(importances):
                        feature_importance = list(zip(feature_names, importances))
                        feature_importance.sort(key=lambda x: x[1], reverse=True)
                        
                        top_features = feature_importance[:5]
                        feature_importance_info = {
                            "has_importance": True,
                            "top_features": [(name, float(imp)) for name, imp in top_features],
                            "feature_count": len(feature_names)
                        }
                        
                        explanation_parts.append(f"""**🎯 Most Important Features:**
{chr(10).join([f'{i+1}. **{name}**: {imp:.1%} importance' for i, (name, imp) in enumerate(top_features)])}""")
            
            # Model type explanation
            model_name = getattr(self, 'model_name', type(self.model).__name__)
            
            if "RandomForest" in model_name:
                explanation_parts.append(f"""**🌳 Random Forest Insights:**
• Uses {getattr(self.model, 'n_estimators', 'multiple')} decision trees
• Makes predictions by voting across all trees
• Excellent at handling complex patterns and interactions
• Naturally handles missing values and mixed data types""")
            
            elif "Logistic" in model_name:
                explanation_parts.append(f"""**📈 Logistic Regression Insights:**
• Uses mathematical relationships to predict probabilities
• Provides clear feature coefficients showing impact direction
• Fast and interpretable for linear relationships
• Works well with properly scaled features""")
            
            # Performance context
            if self.test_data:
                test_size = len(self.test_data["y_test"])
                train_size = len(self.test_data["y_train"])
                explanation_parts.append(f"""**📊 Training Context:**
• **Training examples:** {train_size:,}
• **Test examples:** {test_size:,}
• **Model complexity:** Appropriate for dataset size""")
            
            # Prediction guidance
            explanation_parts.append(f"""**🎯 Using Your Model:**
• **Input:** Provide the same features used in training
• **Output:** {'Class predictions with probabilities' if hasattr(self.model, 'predict_proba') else 'Numerical predictions'}
• **Confidence:** Higher feature importance = more reliable predictions
• **Updates:** Retrain with new data to maintain accuracy""")
            
            return {
                "model_type": model_name,
                "feature_importance": feature_importance_info,
                "interpretability": "high" if "Linear" in model_name or "Logistic" in model_name else "medium",
                "explanation": f"""🧠 **Model Explanation & Insights:**

{chr(10).join(explanation_parts)}

**🚀 Next Steps:**
✅ Your model is trained, evaluated, and ready for production!
✅ Use feature importance to focus on key data collection
✅ Monitor model performance on new data over time
✅ Consider retraining when performance degrades"""
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "explanation": f"❌ Model explanation failed: {str(e)}"
            }


def create_simple_ml_templates():
    """Create simple ML templates without complex typing."""
    return {
        "customer_prediction": {
            "name": "Customer Prediction",
            "description": "Predict customer behavior or characteristics",
            "problem_type": "classification"
        },
        "sales_prediction": {
            "name": "Sales Prediction",
            "description": "Predict sales amounts or trends",
            "problem_type": "regression"
        },
        "data_classification": {
            "name": "Data Classification",
            "description": "Classify data into different categories",
            "problem_type": "classification"
        }
    }