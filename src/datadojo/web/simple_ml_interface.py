"""
Simple ML Pipeline Interface
Streamlit interface for the comprehensive 7-step ML pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional
import io
from pathlib import Path

from ..ml.simple_ml_engine import SimpleMLEngine


def render_simple_ml_pipeline():
    """Render the complete Simple ML Pipeline interface."""
    st.title("🤖 Simple ML Pipeline Alternative")
    st.markdown("""
    **Complete 7-step machine learning pipeline** with detailed evaluation and model explanation.
    
    ✅ **No TypedDict Issues** - Uses basic Python data structures  
    ✅ **Comprehensive Analysis** - Data exploration, feature engineering, model comparison  
    ✅ **Detailed Evaluation** - Performance metrics, confusion matrix, feature importance  
    ✅ **Educational Explanations** - Learn ML concepts at every step
    """)
    
    # Initialize session state
    if 'simple_ml_engine' not in st.session_state:
        st.session_state.simple_ml_engine = SimpleMLEngine()
    
    if 'simple_ml_data' not in st.session_state:
        st.session_state.simple_ml_data = None
        
    if 'simple_ml_target_col' not in st.session_state:
        st.session_state.simple_ml_target_col = None
        
    if 'simple_ml_pipeline' not in st.session_state:
        st.session_state.simple_ml_pipeline = None
        
    if 'simple_ml_results' not in st.session_state:
        st.session_state.simple_ml_results = {}

    # Data Upload Section
    st.header("📁 Upload Your Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset to get started with ML"
        )
    
    with col2:
        use_demo = st.button("🎲 Use Demo Data", help="Load sample loan approval dataset")
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.simple_ml_data = df
            st.success(f"✅ Loaded {len(df)} rows with {len(df.columns)} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return
    
    elif use_demo:
        # Create demo loan approval dataset
        demo_data = {
            'age': [25, 35, 45, 28, 52, 34, 41, 29, 47, 33],
            'income': [50000, 75000, 90000, 55000, 120000, 65000, 85000, 58000, 95000, 70000],
            'experience': [2, 10, 20, 5, 25, 8, 15, 6, 22, 9],
            'credit_score': [650, 720, 800, 680, 850, 700, 750, 660, 780, 710],
            'approved': [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
        }
        df = pd.DataFrame(demo_data)
        st.session_state.simple_ml_data = df
        st.success("✅ Loaded demo loan approval dataset")
    
    # Continue only if data is loaded
    if st.session_state.simple_ml_data is None:
        st.info("👆 Please upload a CSV file or use demo data to get started")
        return
    
    df = st.session_state.simple_ml_data
    
    # Show data preview
    with st.expander("👀 Data Preview", expanded=True):
        st.dataframe(df.head())
        st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Target Column Selection
    st.header("🎯 Choose Target Column")
    target_col = st.selectbox(
        "Select the column you want to predict:",
        options=df.columns.tolist(),
        index=len(df.columns)-1,  # Default to last column
        help="This is what your model will learn to predict"
    )
    
    if target_col:
        st.session_state.simple_ml_target_col = target_col
        
        # Show target distribution
        if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
            st.write("**Target Distribution:**")
            target_counts = df[target_col].value_counts()
            # Use dataframe instead of bar_chart to avoid TypedDict issues
            target_df = pd.DataFrame({
                'Value': target_counts.index,
                'Count': target_counts.values,
                'Percentage': (target_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(target_df)
        else:
            st.write("**Target Statistics:**")
            st.write(df[target_col].describe())
    
    # Create Pipeline
    if st.session_state.simple_ml_target_col:
        if st.session_state.simple_ml_pipeline is None:
            st.session_state.simple_ml_pipeline = st.session_state.simple_ml_engine.create_pipeline(
                target_column=st.session_state.simple_ml_target_col
            )
        
        # ML Pipeline Execution
        st.header("🚀 ML Pipeline Steps")
        
        # Execute steps
        for i, step in enumerate(st.session_state.simple_ml_pipeline):
            step_key = f"step_{i+1}_{step.name}"
            
            # Step header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader(f"Step {i+1}: {step.description}")
            with col2:
                if step_key in st.session_state.simple_ml_results:
                    st.success("✅")
                else:
                    if st.button(f"▶️ Run Step {i+1}", key=f"run_{step_key}"):
                        # Execute the step
                        with st.spinner(f"Running {step.name}..."):
                            try:
                                result = st.session_state.simple_ml_engine.execute_step(
                                    step, df, target_col=st.session_state.simple_ml_target_col
                                )
                                st.session_state.simple_ml_results[step_key] = result
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error in {step.name}: {str(e)}")
            
            # Show results if step completed
            if step_key in st.session_state.simple_ml_results:
                result = st.session_state.simple_ml_results[step_key]
                
                with st.expander(f"📊 Step {i+1} Results", expanded=True):
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        # Show explanation
                        if "explanation" in result:
                            st.markdown(result["explanation"])
                        
                        # Show key metrics in columns
                        if step.name == "load_data":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", result.get("rows", 0))
                            with col2:
                                st.metric("Columns", result.get("columns", 0))
                            with col3:
                                st.metric("Features", len(result.get("column_names", [])) - 1)
                        
                        elif step.name == "explore_data":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Quality Score", f"{result.get('quality_score', 0):.1f}%")
                            with col2:
                                st.metric("Missing Values", result.get("missing_values", 0))
                            with col3:
                                st.metric("Duplicates", result.get("duplicate_rows", 0))
                        
                        elif step.name == "clean_data":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Clean Rows", result.get("cleaned_rows", 0))
                            with col2:
                                st.metric("Numeric Features", result.get("numeric_columns", 0))
                            with col3:
                                st.metric("Text Features", result.get("text_columns", 0))
                        
                        elif step.name == "engineer_features":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Original Features", result.get("original_features", 0))
                            with col2:
                                st.metric("Final Features", result.get("final_features", 0))
                            with col3:
                                improvement = ((result.get("final_features", 0) - result.get("original_features", 1)) / result.get("original_features", 1)) * 100
                                st.metric("Feature Improvement", f"+{improvement:.1f}%")
                        
                        elif step.name == "train_models":
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Models Trained", result.get("models_trained", 0))
                            with col2:
                                st.metric("Best Model", result.get("best_model", "None"))
                            with col3:
                                st.metric("Best Score", f"{result.get('best_score', 0):.1%}")
                            
                            # Show all model scores
                            if "all_scores" in result:
                                st.write("**Model Comparison:**")
                                scores_df = pd.DataFrame(list(result["all_scores"].items()), 
                                                       columns=["Model", "Score"])
                                scores_df["Score"] = scores_df["Score"].apply(lambda x: f"{x:.1%}")
                                st.dataframe(scores_df)
                        
                        elif step.name == "evaluate_performance":
                            if result.get("problem_type") == "classification":
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Accuracy", f"{result.get('accuracy', 0):.1%}")
                                with col2:
                                    st.metric("Precision", f"{result.get('precision', 0):.1%}")
                                with col3:
                                    st.metric("Recall", f"{result.get('recall', 0):.1%}")
                                with col4:
                                    st.metric("F1-Score", f"{result.get('f1_score', 0):.1%}")
                                
                                # Show confusion matrix
                                if "confusion_matrix" in result:
                                    st.write("**Confusion Matrix:**")
                                    cm_df = pd.DataFrame(result["confusion_matrix"])
                                    st.dataframe(cm_df)
                            
                            else:  # regression
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² Score", f"{result.get('r2_score', 0):.3f}")
                                with col2:
                                    st.metric("RMSE", f"{result.get('rmse', 0):.2f}")
                                with col3:
                                    st.metric("MAE", f"{result.get('mae', 0):.2f}")
                        
                        elif step.name == "explain_model":
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Model Type", result.get("model_type", "Unknown"))
                            with col2:
                                st.metric("Interpretability", result.get("interpretability", "medium").title())
                            
                            # Show feature importance
                            if result.get("feature_importance", {}).get("has_importance"):
                                st.write("**🎯 Top Important Features:**")
                                features = result["feature_importance"]["top_features"]
                                feat_df = pd.DataFrame(features, columns=["Feature", "Importance"])
                                feat_df["Importance"] = feat_df["Importance"].apply(lambda x: f"{x:.1%}")
                                st.dataframe(feat_df)
        
        # Final Summary
        completed_steps = len(st.session_state.simple_ml_results)
        total_steps = len(st.session_state.simple_ml_pipeline)
        
        if completed_steps == total_steps:
            st.success("🎉 **Comprehensive ML Pipeline Completed!**")
            st.balloons()
            
            # Offer to download results
            if st.button("📥 Download Results Summary"):
                summary = generate_results_summary(st.session_state.simple_ml_results)
                st.download_button(
                    label="💾 Download ML Results",
                    data=summary,
                    file_name="ml_pipeline_results.txt",
                    mime="text/plain"
                )
        else:
            progress = completed_steps / total_steps
            st.progress(progress)
            st.info(f"Progress: {completed_steps}/{total_steps} steps completed ({progress:.0%})")


def generate_results_summary(results: dict) -> str:
    """Generate a text summary of ML pipeline results."""
    summary = "ML PIPELINE RESULTS SUMMARY\n"
    summary += "=" * 50 + "\n\n"
    
    for step_key, result in results.items():
        step_name = step_key.split("_", 2)[2]  # Extract step name
        summary += f"{step_name.upper().replace('_', ' ')}\n"
        summary += "-" * 30 + "\n"
        
        if "error" in result:
            summary += f"ERROR: {result['error']}\n"
        else:
            # Add key metrics
            for key, value in result.items():
                if key not in ["explanation", "error"]:
                    summary += f"{key}: {value}\n"
        
        summary += "\n"
    
    return summary
