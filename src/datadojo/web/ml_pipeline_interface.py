"""
ML Pipeline Interface - Web dashboard for visual pipeline building
Simple, educational machine learning with drag-and-drop interface.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional
import json
from pathlib import Path
import time

from ..ml.automl_engine import AutoMLEngine, MLPipelineStep, create_ml_templates


def render_ml_pipeline_page():
    """Render the ML Pipeline page with visual pipeline builder."""
    
    st.title("🤖 ML Pipeline Builder")
    st.markdown("""
    Build machine learning models with a simple, visual interface. 
    No coding required - just follow the steps and learn as you go!
    """)
    
    # Initialize session state
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = AutoMLEngine()
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'ml_data' not in st.session_state:
        st.session_state.ml_data = None
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Quick Start", 
        "📊 Pipeline Builder", 
        "📈 Results Dashboard",
        "📚 ML Templates"
    ])
    
    with tab1:
        render_quick_start()
    
    with tab2:
        render_pipeline_builder()
    
    with tab3:
        render_results_dashboard()
    
    with tab4:
        render_ml_templates()


def render_quick_start():
    """Quick start section for beginners."""
    
    st.header("🚀 Quick Start - Build Your First Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### What is Machine Learning? 🤔
        
        Think of machine learning like teaching a computer to recognize patterns, 
        just like how you learned to recognize faces or predict weather from clouds.
        
        **Here's how it works:**
        
        1. **📊 Show examples** - Give the computer lots of data
        2. **🔍 Find patterns** - Let it discover what makes things similar  
        3. **🎯 Make predictions** - Use patterns to guess new answers
        4. **✅ Check accuracy** - See how often it gets things right
        
        ### Ready to try? Pick your data below! 👇
        """)
    
    with col2:
        st.info("""
        **💡 Quick Tip**
        
        Start with a small, clean dataset 
        for your first model. 
        
        The system will guide you 
        through every step!
        """)
    
    # Data selection
    st.subheader("Step 1: Choose Your Data 📂")
    
    data_source = st.radio(
        "Where is your data?",
        ["Use demo dataset", "Upload my own file", "Connect to existing dataset"],
        horizontal=True
    )
    
    df = None
    
    if data_source == "Use demo dataset":
        demo_options = {
            "Customer Data": "demo_datasets/sample_customers.csv",
            "Bank Transactions": "demo_datasets/sample_bank_transactions.csv", 
            "Patient Records": "demo_datasets/sample_patients.csv",
            "Lab Results": "demo_datasets/sample_lab_results.csv"
        }
        
        selected_demo = st.selectbox("Choose a demo dataset:", list(demo_options.keys()))
        
        if st.button("Load Demo Data", type="primary"):
            try:
                df = pd.read_csv(demo_options[selected_demo])
                st.session_state.ml_data = df
                st.success(f"✅ Loaded {selected_demo} with {len(df)} rows!")
                
                # Show preview
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Could not load dataset: {str(e)}")
    
    elif data_source == "Upload my own file":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv",
            help="Upload a CSV file with your data. Make sure it has column headers!"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.ml_data = df
                st.success(f"✅ Uploaded file with {len(df)} rows and {len(df.columns)} columns!")
                
                with st.expander("📊 Data Preview"):
                    st.dataframe(df.head())
                    
            except Exception as e:
                st.error(f"Could not read file: {str(e)}")
    
    else:  # Connect to existing
        st.info("Feature coming soon! For now, use demo datasets or upload a CSV file.")
    
    # If we have data, show next steps
    if st.session_state.ml_data is not None:
        df = st.session_state.ml_data
        
        st.subheader("Step 2: What do you want to predict? 🎯")
        
        # Target selection
        columns = list(df.columns)
        target_col = st.selectbox(
            "Choose the column you want to predict:",
            ["Select a column..."] + columns,
            help="This is what your model will learn to predict for new data"
        )
        
        if target_col != "Select a column...":
            # Auto-detect problem type
            problem_type = st.session_state.ml_engine.detect_problem_type(df, target_col)
            
            # Show problem type explanation
            problem_explanations = {
                "classification": f"""
                🎯 **Classification Problem Detected**
                
                You want to predict **categories** or **groups**.
                Your target column "{target_col}" has {df[target_col].nunique()} different values.
                
                **Example predictions:**
                • Is this email spam or not spam?
                • Will this customer buy or not buy?
                • What category does this belong to?
                """,
                "regression": f"""
                📈 **Regression Problem Detected**
                
                You want to predict **numbers** or **amounts**.
                Your target column "{target_col}" has continuous numerical values.
                
                **Example predictions:**
                • What will the price be?
                • How much will sales increase?
                • What score will this get?
                """,
                "clustering": f"""
                🔍 **Clustering Problem Detected**
                
                You want to find **hidden groups** in your data.
                No target column needed - we'll discover natural patterns!
                
                **Example discoveries:**
                • Customer segments with similar behavior
                • Product categories based on features
                • Patient groups with similar symptoms
                """
            }
            
            st.info(problem_explanations[problem_type])
            
            # Start pipeline button
            if st.button("🚀 Start Building My Model!", type="primary", key="start_pipeline"):
                # Initialize pipeline
                st.session_state.pipeline_steps = st.session_state.ml_engine.create_pipeline_template(problem_type)
                st.session_state.ml_engine.target_column = target_col
                st.session_state.ml_engine.problem_type = problem_type
                st.session_state.current_step = 0
                
                st.success("🎉 Pipeline created! Go to the **Pipeline Builder** tab to continue.")
                st.balloons()


def render_pipeline_builder():
    """Visual pipeline builder interface."""
    
    st.header("📊 Pipeline Builder")
    
    if not st.session_state.pipeline_steps:
        st.info("""
        👈 **Start in the Quick Start tab first!**
        
        Choose your data and target column to create a custom pipeline
        that matches your specific machine learning problem.
        """)
        return
    
    # Pipeline progress
    st.subheader("🛤️ Your ML Pipeline")
    
    steps = st.session_state.pipeline_steps
    current_step = st.session_state.current_step
    
    # Visual pipeline progress
    render_pipeline_progress(steps, current_step)
    
    # Current step details
    if current_step < len(steps):
        step = steps[current_step]
        
        st.markdown(f"""
        ### Step {current_step + 1}: {step.name.replace('_', ' ').title()} 
        *{step.description}*
        """)
        
        # Step-specific interface
        if step.status == "pending":
            render_step_interface(step, current_step)
        elif step.status == "completed":
            render_step_results(step, current_step)
        elif step.status == "failed":
            st.error(f"❌ Step failed: {step.result.get('error', 'Unknown error')}")
            if st.button("🔄 Retry Step"):
                step.status = "pending"
                step.result = None
                st.rerun()
    
    else:
        # All steps completed
        st.success("🎉 **Pipeline Complete!**")
        st.markdown("""
        Congratulations! You've successfully built and trained a machine learning model.
        Check out the **Results Dashboard** tab to explore your results!
        """)
        st.balloons()


def render_pipeline_progress(steps: List[MLPipelineStep], current_step: int):
    """Render visual pipeline progress."""
    
    # Create progress visualization
    fig = go.Figure()
    
    # Step positions
    x_pos = list(range(len(steps)))
    y_pos = [0] * len(steps)
    
    # Colors based on status
    colors = []
    symbols = []
    sizes = []
    
    for i, step in enumerate(steps):
        if step.status == "completed":
            colors.append("green")
            symbols.append("circle")
            sizes.append(20)
        elif step.status == "running":
            colors.append("orange") 
            symbols.append("circle")
            sizes.append(25)
        elif step.status == "failed":
            colors.append("red")
            symbols.append("x")
            sizes.append(20)
        else:  # pending
            colors.append("lightgray")
            symbols.append("circle-open")
            sizes.append(15)
    
    # Add connecting lines
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='lines',
        line=dict(color='lightgray', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add step points
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(
            color=colors,
            symbol=symbols,
            size=sizes,
            line=dict(width=2, color='white')
        ),
        text=[f"Step {i+1}" for i in range(len(steps))],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>',
        customdata=[step.name.replace('_', ' ').title() for step in steps],
        showlegend=False
    ))
    
    fig.update_layout(
        title="Pipeline Progress",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[-0.5, 0.5]),
        height=150,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig)
    
    # Progress bar
    completed_steps = sum(1 for step in steps if step.status == "completed")
    progress = completed_steps / len(steps)
    st.progress(progress, text=f"Progress: {completed_steps}/{len(steps)} steps completed")


def render_step_interface(step: MLPipelineStep, step_index: int):
    """Render interface for executing a pipeline step."""
    
    # Step explanation
    with st.expander("ℹ️ What does this step do?", expanded=True):
        step_explanations = {
            "data_loading": "We'll examine your dataset to understand its size, structure, and basic properties.",
            "data_exploration": "We'll analyze your data to find patterns, missing values, and data quality issues.",
            "data_cleaning": "We'll automatically fix missing values, remove duplicates, and prepare clean data.",
            "feature_engineering": "We'll convert text to numbers and prepare features for machine learning algorithms.",
            "model_selection": "We'll train multiple algorithms and pick the best one for your specific problem.",
            "model_evaluation": "We'll test your model on new data to see how accurate it is in the real world.",
            "model_explanation": "We'll show you what your model learned and which features are most important.",
            "clustering": "We'll find natural groups in your data using advanced clustering algorithms.",
            "cluster_analysis": "We'll analyze the groups we found to understand what makes them different.",
            "cluster_interpretation": "We'll help you understand what each group represents in business terms."
        }
        
        st.markdown(step_explanations.get(step.name, step.description))
    
    # Execute button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"▶️ Execute Step {step_index + 1}", type="primary", key=f"execute_{step.name}"):
            execute_pipeline_step(step, step_index)


def execute_pipeline_step(step: MLPipelineStep, step_index: int):
    """Execute a single pipeline step with progress feedback."""
    
    # Progress indicator
    with st.spinner(f"⚙️ Executing {step.name.replace('_', ' ')}..."):
        
        # Simulate some processing time for educational effect
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("🔍 Analyzing data...")
            elif i < 60:
                status_text.text("⚙️ Processing...")  
            elif i < 90:
                status_text.text("🤖 Running algorithms...")
            else:
                status_text.text("✅ Finalizing results...")
            time.sleep(0.02)  # Small delay for visual effect
        
        try:
            # Execute the actual step
            engine = st.session_state.ml_engine
            df = st.session_state.ml_data
            
            # Prepare arguments based on step
            kwargs = {}
            if step.name == "feature_engineering":
                kwargs["target_col"] = engine.target_column
            elif step.name == "model_selection":
                kwargs["target_col"] = engine.target_column
                kwargs["problem_type"] = engine.problem_type
            
            result = engine.execute_step(step, df, **kwargs)
            
            # CRITICAL FIX: Clean result to avoid TypedDict serialization issues
            # Convert all values to basic Python types for Streamlit compatibility
            cleaned_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    cleaned_result[key] = value
                elif hasattr(value, 'tolist'):  # numpy arrays
                    cleaned_result[key] = value.tolist()
                elif isinstance(value, dict):
                    # Recursively clean nested dictionaries
                    cleaned_dict = {}
                    for k, v in value.items():
                        if isinstance(v, (str, int, float, bool, type(None))):
                            cleaned_dict[str(k)] = v
                        else:
                            cleaned_dict[str(k)] = str(v)
                    cleaned_result[str(key)] = cleaned_dict
                else:
                    cleaned_result[str(key)] = str(value)
            
            # Update UI
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Move to next step
            st.session_state.current_step = step_index + 1
            
            # Show success and results
            st.success(f"✅ Step {step_index + 1} completed successfully!")
            
            # Show step results immediately  
            with st.expander("📊 Step Results", expanded=True):
                st.markdown(cleaned_result.get("explanation", "Step completed."))
                
                # Additional result displays based on step type
                if step.name == "data_exploration" and "missing_data" in result:
                    missing_data = {k: v for k, v in result["missing_data"].items() if v > 0}
                    if missing_data:
                        st.markdown("**Missing Data Summary:**")
                        st.json(missing_data)
                
                elif step.name == "model_selection" and "model_scores" in result:
                    st.markdown("**Model Performance Comparison:**")
                    scores_df = pd.DataFrame(list(result["model_scores"].items()), 
                                           columns=["Model", "Score"])
                    st.bar_chart(scores_df.set_index("Model"))
            
            time.sleep(1)  # Brief pause before rerun
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Step failed: {str(e)}")
            step.status = "failed"
            step.result = {"error": str(e)}


def render_step_results(step: MLPipelineStep, step_index: int):
    """Render results from a completed pipeline step."""
    
    st.success(f"✅ Step {step_index + 1} completed in {step.duration:.1f} seconds")
    
    # Show results
    with st.expander("📊 Results", expanded=True):
        if step.result and "explanation" in step.result:
            st.markdown(step.result["explanation"])
        
        # Step-specific result displays
        if step.name == "model_selection" and step.result:
            if "model_scores" in step.result:
                st.markdown("**Model Performance:**")
                scores_df = pd.DataFrame(
                    list(step.result["model_scores"].items()),
                    columns=["Model", "Score"]
                )
                
                fig = px.bar(scores_df, x="Model", y="Score", 
                           title="Model Comparison")
                st.plotly_chart(fig)
        
        elif step.name == "model_explanation" and step.result:
            if "feature_importance" in step.result:
                st.markdown("**Feature Importance:**")
                importance_data = step.result["feature_importance"]
                
                # Create DataFrame and sort
                importance_df = pd.DataFrame(
                    list(importance_data.items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=True)
                
                fig = px.bar(importance_df.tail(10), x="Importance", y="Feature",
                           orientation='h', title="Top 10 Most Important Features")
                st.plotly_chart(fig)
    
    # Next step button
    if step_index + 1 < len(st.session_state.pipeline_steps):
        next_step = st.session_state.pipeline_steps[step_index + 1]
        if next_step.status == "pending":
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(f"➡️ Continue to Step {step_index + 2}", type="primary"):
                    st.session_state.current_step = step_index + 1
                    st.rerun()


def render_results_dashboard():
    """Render the results dashboard after pipeline completion."""
    
    st.header("📈 Results Dashboard")
    
    if not st.session_state.pipeline_steps:
        st.info("Complete a pipeline first to see results here!")
        return
    
    # Check if pipeline is complete
    completed_steps = [step for step in st.session_state.pipeline_steps if step.status == "completed"]
    total_steps = len(st.session_state.pipeline_steps)
    
    if len(completed_steps) == 0:
        st.info("Start executing your pipeline to see results here!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Steps Completed", len(completed_steps), f"of {total_steps}")
    
    with col2:
        total_time = sum(step.duration for step in completed_steps)
        st.metric("Total Time", f"{total_time:.1f}s")
    
    with col3:
        if st.session_state.ml_data is not None:
            st.metric("Dataset Size", f"{len(st.session_state.ml_data):,} rows")
    
    with col4:
        # Find best model score if available
        best_score = None
        for step in completed_steps:
            if step.name == "model_selection" and step.result:
                best_score = step.result.get("best_score")
                break
        
        if best_score is not None:
            st.metric("Best Model Score", f"{best_score:.1%}")
    
    # Detailed results by section
    st.subheader("📊 Detailed Results")
    
    for i, step in enumerate(completed_steps):
        with st.expander(f"Step {i+1}: {step.name.replace('_', ' ').title()}", expanded=False):
            if step.result and "explanation" in step.result:
                st.markdown(step.result["explanation"])
            
            # Additional visualizations
            if step.name == "data_exploration":
                render_data_exploration_charts(step.result)
            elif step.name == "model_selection":
                render_model_comparison_charts(step.result)
            elif step.name == "model_explanation":
                render_model_explanation_charts(step.result)


def render_data_exploration_charts(result: Dict):
    """Render charts for data exploration results."""
    
    if "missing_percent" in result:
        missing_data = {k: v for k, v in result["missing_percent"].items() if v > 0}
        
        if missing_data:
            df_missing = pd.DataFrame(list(missing_data.items()), 
                                    columns=["Column", "Missing %"])
            
            fig = px.bar(df_missing, x="Missing %", y="Column", 
                        orientation='h', title="Missing Data by Column")
            st.plotly_chart(fig)


def render_model_comparison_charts(result: Dict):
    """Render charts for model comparison results."""
    
    if "model_scores" in result:
        scores_df = pd.DataFrame(list(result["model_scores"].items()), 
                               columns=["Model", "Score"])
        
        fig = px.bar(scores_df, x="Model", y="Score", 
                    title="Model Performance Comparison",
                    color="Score", color_continuous_scale="viridis")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)


def render_model_explanation_charts(result: Dict):
    """Render charts for model explanation results."""
    
    if "feature_importance" in result:
        importance_data = result["feature_importance"]
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame(
            list(importance_data.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        
        # Show top 10 features
        top_features = importance_df.tail(10)
        
        fig = px.bar(top_features, x="Importance", y="Feature",
                    orientation='h', title="Feature Importance",
                    color="Importance", color_continuous_scale="blues")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)


def render_ml_templates():
    """Render ML templates for quick start."""
    
    st.header("📚 ML Templates")
    st.markdown("""
    Choose from pre-built templates for common business problems. 
    Each template includes sample data and step-by-step guidance.
    """)
    
    templates = create_ml_templates()
    
    # Template cards
    for template_id, template in templates.items():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **{template['name']}** 
                
                {template['description']}
                
                🎯 **Business Value:** {template['business_value']}  
                📊 **Problem Type:** {template['problem_type'].title()}  
                🎓 **Difficulty:** {template['difficulty']}
                """)
            
            with col2:
                if st.button(f"Use Template", key=f"template_{template_id}"):
                    # Set up template pipeline
                    engine = AutoMLEngine()
                    problem_type = template['problem_type']
                    
                    st.session_state.ml_engine = engine
                    st.session_state.pipeline_steps = engine.create_pipeline_template(problem_type)
                    st.session_state.ml_engine.problem_type = problem_type
                    st.session_state.ml_engine.target_column = template.get('example_target')
                    st.session_state.current_step = 0
                    
                    st.success(f"✅ {template['name']} template loaded!")
                    st.info("👈 Go to **Quick Start** to load your data and begin!")
        
        st.divider()