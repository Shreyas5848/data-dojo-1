"""
Notebook Templates Interface
Streamlit interface for generating and managing Jupyter notebook templates
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
import json
import nbformat as nbf

from ..notebook.template_engine import NotebookTemplateEngine, get_available_templates


# Constants
NOTEBOOKS_DIR = Path("generated_notebooks")
PROGRESS_FILE = Path("user_progress.json")


def save_notebook_to_workspace(notebook, filename: str, template_type: str) -> Path:
    """Save generated notebook to workspace folder."""
    # Create directory if it doesn't exist
    NOTEBOOKS_DIR.mkdir(exist_ok=True)
    
    # Create subdirectory for template type
    template_dir = NOTEBOOKS_DIR / template_type
    template_dir.mkdir(exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.ipynb"
    filepath = template_dir / full_filename
    
    # Write notebook
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(notebook, f)
    
    return filepath


def update_progress_on_notebook_generation(template_type: str, dataset_name: str):
    """Update progress tracking when a notebook is generated."""
    try:
        # Load existing progress
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
        else:
            progress = get_default_progress()
        
        # Map template type to skill
        skill_map = {
            'exploratory_data_analysis': 'data_exploration',
            'data_cleaning': 'data_cleaning',
            'classification_analysis': 'classification',
            'regression_analysis': 'regression',
            'time_series_analysis': 'time_series',
            'clustering_analysis': 'clustering',
            'dimensionality_reduction': 'data_exploration',
            'feature_engineering': 'feature_engineering'
        }
        
        skill = skill_map.get(template_type, 'data_exploration')
        
        # Award XP
        xp_amount = 25
        progress["skills"][skill]["xp"] += xp_amount
        
        # Check for level up
        skill_data = progress["skills"][skill]
        while skill_data["xp"] >= skill_data["max_xp"]:
            skill_data["xp"] -= skill_data["max_xp"]
            skill_data["level"] += 1
            skill_data["max_xp"] = 100 * (skill_data["level"] + 1)
        
        # Update total XP
        progress["user_info"]["xp"] += xp_amount
        
        # Calculate level
        level = 1
        xp = progress["user_info"]["xp"]
        xp_needed = 100
        while xp >= xp_needed:
            xp -= xp_needed
            level += 1
            xp_needed = 100 * level
        progress["user_info"]["level"] = level
        
        # Increment notebooks counter
        progress["notebooks_generated"] += 1
        
        # Record activity
        activity = {
            "name": f"Generated {template_type.replace('_', ' ').title()} notebook for {dataset_name}",
            "skill": skill,
            "xp": xp_amount,
            "date": datetime.now().isoformat()
        }
        progress["activities"].insert(0, activity)
        progress["activities"] = progress["activities"][:50]
        
        # Check achievements
        progress = check_notebook_achievements(progress)
        
        # Save progress
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        return xp_amount, skill, progress["notebooks_generated"]
        
    except Exception as e:
        st.warning(f"Could not update progress: {e}")
        return 0, None, 0


def check_notebook_achievements(progress):
    """Check and award notebook-related achievements."""
    existing = [a["name"] for a in progress.get("achievements", [])]
    notebooks = progress.get("notebooks_generated", 0)
    
    new_achievements = []
    
    if notebooks >= 1 and "First Notebook" not in existing:
        new_achievements.append({
            "name": "First Notebook",
            "description": "Generated your first analysis notebook",
            "date": datetime.now().isoformat(),
            "icon": "📓"
        })
    
    if notebooks >= 5 and "Notebook Ninja" not in existing:
        new_achievements.append({
            "name": "Notebook Ninja",
            "description": "Generated 5 analysis notebooks",
            "date": datetime.now().isoformat(),
            "icon": "🥷"
        })
    
    if notebooks >= 10 and "Notebook Master" not in existing:
        new_achievements.append({
            "name": "Notebook Master",
            "description": "Generated 10 analysis notebooks",
            "date": datetime.now().isoformat(),
            "icon": "🎓"
        })
    
    if notebooks >= 25 and "Notebook Legend" not in existing:
        new_achievements.append({
            "name": "Notebook Legend",
            "description": "Generated 25 analysis notebooks",
            "date": datetime.now().isoformat(),
            "icon": "🏆"
        })
    
    progress["achievements"].extend(new_achievements)
    return progress


def get_default_progress():
    """Return default progress structure."""
    return {
        "user_info": {
            "name": "Data Scientist",
            "started": datetime.now().isoformat(),
            "level": 1,
            "xp": 0
        },
        "skills": {
            "data_exploration": {"level": 0, "xp": 0, "max_xp": 100},
            "data_cleaning": {"level": 0, "xp": 0, "max_xp": 100},
            "visualization": {"level": 0, "xp": 0, "max_xp": 100},
            "classification": {"level": 0, "xp": 0, "max_xp": 100},
            "regression": {"level": 0, "xp": 0, "max_xp": 100},
            "clustering": {"level": 0, "xp": 0, "max_xp": 100},
            "feature_engineering": {"level": 0, "xp": 0, "max_xp": 100},
            "time_series": {"level": 0, "xp": 0, "max_xp": 100}
        },
        "achievements": [],
        "activities": [],
        "notebooks_generated": 0,
        "datasets_profiled": 0,
        "datasets_explored": 0,
        "total_sessions": 0,
        "streak_days": 0,
        "last_active": None
    }


def list_saved_notebooks() -> list:
    """List all saved notebooks in the workspace."""
    notebooks = []
    
    if NOTEBOOKS_DIR.exists():
        for template_dir in NOTEBOOKS_DIR.iterdir():
            if template_dir.is_dir():
                for nb_file in template_dir.glob("*.ipynb"):
                    stat = nb_file.stat()
                    notebooks.append({
                        "name": nb_file.stem,
                        "template": template_dir.name,
                        "path": str(nb_file),
                        "created": datetime.fromtimestamp(stat.st_ctime),
                        "size_kb": stat.st_size / 1024
                    })
    
    # Sort by creation time, newest first
    notebooks.sort(key=lambda x: x["created"], reverse=True)
    return notebooks


def render_notebook_templates():
    """Render the Notebook Templates interface with modern styling."""
    
    # Professional header
    st.markdown("""
    <h1 style="color: #F8FAFC; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">
    Notebook Templates
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem;">
    Generate interactive Jupyter notebooks from your data
    </p>
    """, unsafe_allow_html=True)
    
    # Feature cards - Clean design
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155;">
            <span style="color: #3B82F6; font-size: 1.25rem; font-weight: 600;">{}</span>
            <p style="color: #CBD5E1; margin: 0.5rem 0 0 0; font-size: 0.8rem;">Smart Templates</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155;">
            <span style="color: #14B8A6; font-size: 1.25rem; font-weight: 600;">></span>
            <p style="color: #CBD5E1; margin: 0.5rem 0 0 0; font-size: 0.8rem;">Ready-to-Run</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155;">
            <span style="color: #8B5CF6; font-size: 1.25rem; font-weight: 600;">?</span>
            <p style="color: #CBD5E1; margin: 0.5rem 0 0 0; font-size: 0.8rem;">Educational</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background: #1E293B; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #334155;">
            <span style="color: #22C55E; font-size: 1.25rem; font-weight: 600;">+</span>
            <p style="color: #CBD5E1; margin: 0.5rem 0 0 0; font-size: 0.8rem;">Customizable</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'notebook_engine' not in st.session_state:
        st.session_state.notebook_engine = NotebookTemplateEngine()
    if 'generated_notebook' not in st.session_state:
        st.session_state.generated_notebook = None
    if 'template_data' not in st.session_state:
        st.session_state.template_data = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'dataset_name' not in st.session_state:
        st.session_state.dataset_name = None
    if 'profile_results' not in st.session_state:
        st.session_state.profile_results = None
    if 'uploaded_file_key' not in st.session_state:
        st.session_state.uploaded_file_key = None

    # Step 1: Data Input
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #3B82F6;">1.</span> Provide Your Data
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload the dataset you want to create a notebook template for",
            key="notebook_file_uploader"
        )
    
    with col2:
        use_demo = st.button("Use Demo Data", help="Load sample dataset for testing", type="primary")
    
    # Load data - only process new uploads, keep existing data in session
    if uploaded_file is not None:
        try:
            # Create unique key for this file
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Only reload if this is a different file
            if st.session_state.get('uploaded_file_key') != file_key:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_df = df
                st.session_state.dataset_name = uploaded_file.name.replace('.csv', '')
                st.session_state.uploaded_file_key = file_key
                st.session_state.profile_results = None  # Reset profile for new data
                st.success(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            # else: data already in session state, no need to reload
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    
    elif use_demo:
        # Reset profile cache for demo dataset
        if st.session_state.dataset_name != "customer_demo":
            st.session_state.profile_results = None
        
        # Create demo dataset
        demo_data = {
            'customer_id': range(1, 101),
            'age': np.random.randint(18, 80, 100),
            'income': np.random.randint(30000, 150000, 100),
            'spending_score': np.random.randint(1, 100, 100),
            'gender': np.random.choice(['Male', 'Female'], 100),
            'purchase_amount': np.random.uniform(10, 500, 100).round(2),
            'is_premium': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(demo_data)
        st.session_state.current_df = df
        st.session_state.dataset_name = "customer_demo"
        st.success("Loaded demo customer dataset")
    
    # Use data from session state if available
    df = st.session_state.current_df
    dataset_name = st.session_state.dataset_name
    
    if df is None:
        st.info("👆 Please upload a CSV file or use demo data to get started")
        return
    
    # Show data preview
    with st.expander("👀 Data Preview", expanded=True):
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Step 2: Generate Profile Results
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #14B8A6;">2.</span> Analyze Data Characteristics
    </h3>
    """, unsafe_allow_html=True)
    
    # Generate or use cached profile results
    if st.session_state.profile_results is None:
        with st.spinner("Analyzing dataset characteristics..."):
            profile_results = generate_profile_summary(df)
            st.session_state.profile_results = profile_results
    else:
        profile_results = st.session_state.profile_results
    
    # Display analysis results
    with st.expander("Data Analysis Results", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Quality Score", f"{profile_results['overall_quality_score']:.1f}%")
        with col2:
            st.metric("Missing Values", profile_results['total_missing'])
        with col3:
            st.metric("Numeric Columns", len(profile_results['numeric_columns']))
        with col4:
            st.metric("Categorical Columns", len(profile_results['categorical_columns']))
        
        # Show column details
        st.write("**Column Types:**")
        col_info = []
        for col in df.columns:
            col_type = "Numeric" if col in profile_results['numeric_columns'] else "Categorical"
            missing_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            col_info.append({
                'Column': col,
                'Type': col_type,
                'Missing': missing_count,
                'Unique Values': unique_count,
                'Sample': str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            })
        
        st.dataframe(pd.DataFrame(col_info))
    
    # Step 3: Template Selection
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #8B5CF6;">3.</span> Choose Template Type
    </h3>
    """, unsafe_allow_html=True)
    
    available_templates = get_available_templates()
    
    # Smart template recommendations
    recommended_template = recommend_template(profile_results, df)
    recommendation_reason = get_recommendation_reason(recommended_template, profile_results, df)
    
    st.info(f"**Recommended:** {available_templates[recommended_template]}\n\n*{recommendation_reason}*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_template = st.selectbox(
            "Select template type:",
            options=list(available_templates.keys()),
            format_func=lambda x: available_templates[x],
            index=list(available_templates.keys()).index(recommended_template)
        )
    
    with col2:
        st.write("**Template Features:**")
        template_features = get_template_features(selected_template)
        for feature in template_features:
            st.write(f"• {feature}")
    
    # Step 3.5: Template Customization
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #8B5CF6;">3.5</span> Customize Template (Optional)
    </h3>
    """, unsafe_allow_html=True)
    
    with st.expander("Advanced Options", expanded=False):
        st.write("**Select sections to include in your notebook:**")
        
        # Get sections for selected template
        template_sections = get_template_sections(selected_template)
        
        # Initialize session state for section selection
        if 'selected_sections' not in st.session_state:
            st.session_state.selected_sections = {section: True for section in template_sections}
        
        # Reset selections when template changes
        if 'last_template' not in st.session_state or st.session_state.last_template != selected_template:
            st.session_state.selected_sections = {section: True for section in template_sections}
            st.session_state.last_template = selected_template
        
        # Quick actions
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("✅ Select All"):
                st.session_state.selected_sections = {section: True for section in template_sections}
                st.rerun()
        with col_b:
            if st.button("❌ Deselect All"):
                st.session_state.selected_sections = {section: False for section in template_sections}
                st.rerun()
        with col_c:
            if st.button("🔄 Reset to Default"):
                st.session_state.selected_sections = {section: True for section in template_sections}
                st.rerun()
        
        st.write("---")
        
        # Section checkboxes in columns
        n_sections = len(template_sections)
        cols = st.columns(2)
        
        for i, section in enumerate(template_sections):
            col_idx = i % 2
            with cols[col_idx]:
                st.session_state.selected_sections[section] = st.checkbox(
                    section,
                    value=st.session_state.selected_sections.get(section, True),
                    key=f"section_{selected_template}_{i}"
                )
        
        # Show selection count
        selected_count = sum(st.session_state.selected_sections.values())
        st.write(f"**Selected:** {selected_count}/{len(template_sections)} sections")
        
        if selected_count == 0:
            st.warning("Please select at least one section!")

    # Step 4: Generate Notebook
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #22C55E;">4.</span> Generate Notebook
    </h3>
    """, unsafe_allow_html=True)
    
    # Auto-save option
    auto_save = st.checkbox("Auto-save to workspace", value=True, 
                            help="Automatically save generated notebooks to the generated_notebooks/ folder")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        generate_btn = st.button("Generate Notebook", type="primary")
    
    with col2:
        if st.session_state.generated_notebook:
            preview_btn = st.button("Preview")
        else:
            preview_btn = st.button("Preview", disabled=True)
    
    with col3:
        if st.session_state.generated_notebook:
            download_btn = st.button("Download")
        else:
            download_btn = st.button("Download", disabled=True)
    
    # Generate notebook
    if generate_btn and dataset_name and profile_results:
        with st.spinner(f"Generating {available_templates[selected_template]} notebook..."):
            try:
                notebook = st.session_state.notebook_engine.generate_notebook(
                    profile_results, selected_template, dataset_name
                )
                st.session_state.generated_notebook = notebook
                st.session_state.template_data = {
                    'name': dataset_name,
                    'template_type': selected_template,
                    'profile_results': profile_results
                }
                
                # Auto-save to workspace
                saved_path = None
                if auto_save:
                    saved_path = save_notebook_to_workspace(notebook, dataset_name, selected_template)
                    st.session_state.last_saved_path = str(saved_path)
                
                # Update progress tracking
                xp_earned, skill, total_notebooks = update_progress_on_notebook_generation(
                    selected_template, dataset_name
                )
                
                # Success message with details
                st.success("✅ Notebook generated successfully!")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    if xp_earned > 0:
                        st.info(f"🎯 +{xp_earned} XP earned!")
                with col_b:
                    if total_notebooks > 0:
                        st.info(f"📓 Total notebooks: {total_notebooks}")
                with col_c:
                    if saved_path:
                        st.info(f"💾 Saved to workspace")
                
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error generating notebook: {str(e)}")
                st.error(f"Debug info: dataset_name={dataset_name}, template={selected_template}")
    
    # Preview notebook
    if st.session_state.generated_notebook and preview_btn:
        st.markdown("""
        <h3 style="color: white; margin: 1.5rem 0 1rem 0;">
        📖 Notebook Preview
        </h3>
        """, unsafe_allow_html=True)
        
        notebook = st.session_state.generated_notebook
        
        with st.expander("📝 Notebook Contents", expanded=True):
            st.write(f"**Cells:** {len(notebook.cells)}")
            if st.session_state.template_data:
                st.write(f"**Template:** {available_templates[st.session_state.template_data['template_type']]}")
            else:
                st.write("**Template:** Unknown")
            
            # Show first few cells
            for i, cell in enumerate(notebook.cells[:5]):  # Show first 5 cells
                st.write(f"**Cell {i+1} ({cell.cell_type}):**")
                if cell.cell_type == 'markdown':
                    st.markdown(cell.source[:200] + "..." if len(cell.source) > 200 else cell.source)
                else:
                    st.code(cell.source[:300] + "..." if len(cell.source) > 300 else cell.source, language='python')
                st.write("---")
            
            if len(notebook.cells) > 5:
                st.write(f"... and {len(notebook.cells) - 5} more cells")
    
    # Download notebook
    if st.session_state.generated_notebook and download_btn:
        notebook = st.session_state.generated_notebook
        template_info = st.session_state.template_data
        
        # Convert notebook to string
        notebook_json = nbf.writes(notebook)
        
        # Create download
        if template_info:
            filename = f"{template_info['name']}_{template_info['template_type']}.ipynb"
        else:
            filename = "notebook_template.ipynb"
        
        st.download_button(
            label="📥 Download Jupyter Notebook",
            data=notebook_json,
            file_name=filename,
            mime="application/json",
            help="Download the generated notebook to run in Jupyter Lab/Notebook"
        )
        
        st.success(f"✅ Ready to download: {filename}")
        
        # Usage instructions
        with st.expander("📚 How to Use Your Notebook", expanded=False):
            st.markdown("""
            ### 🚀 Getting Started with Your Notebook
            
            1. **Download** the notebook file using the button above
            2. **Install Jupyter** (if not already installed):
               ```bash
               pip install jupyter notebook
               ```
            3. **Launch Jupyter**:
               ```bash
               jupyter notebook
               ```
            4. **Open** your downloaded notebook file
            5. **Update** the data loading path in the first code cell
            6. **Run** all cells to see the complete analysis
            
            ### 💡 Tips for Best Results
            - Replace `'your_file.csv'` with your actual data file path
            - Install any missing libraries with `pip install package_name`
            - Customize the analysis based on your specific needs
            - Add your own cells for additional analysis
            
            ### 📊 What's Included
            - Data loading and initial exploration
            - Comprehensive statistical analysis
            - Visualizations and charts
            - Data cleaning recommendations
            - Next steps guidance
            """)
    
    # Step 5: Saved Notebooks (optional section)
    st.markdown("---")
    st.markdown("""
    <h3 style="color: #F8FAFC; margin: 1.5rem 0 1rem 0; font-size: 1.125rem; font-weight: 600;">
    <span style="color: #14B8A6;">5.</span> Saved Notebooks
    </h3>
    """, unsafe_allow_html=True)
    
    saved_notebooks = list_saved_notebooks()
    
    if saved_notebooks:
        st.success(f"Found {len(saved_notebooks)} saved notebooks in workspace")
        
        # Group by template type
        templates_used = set(nb["template"] for nb in saved_notebooks)
        
        for template_type in templates_used:
            template_notebooks = [nb for nb in saved_notebooks if nb["template"] == template_type]
            
            with st.expander(f"{template_type.replace('_', ' ').title()} ({len(template_notebooks)} notebooks)", expanded=False):
                for nb in template_notebooks[:5]:  # Show max 5 per category
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.write(f"**{nb['name']}**")
                    with col2:
                        st.caption(nb["created"].strftime("%Y-%m-%d %H:%M"))
                    with col3:
                        st.caption(f"{nb['size_kb']:.1f} KB")
                
                if len(template_notebooks) > 5:
                    st.caption(f"... and {len(template_notebooks) - 5} more")
        
        # Show last saved path if available
        if hasattr(st.session_state, 'last_saved_path') and st.session_state.last_saved_path:
            st.info(f"📍 Last saved: `{st.session_state.last_saved_path}`")
    else:
        st.info("📝 No notebooks saved yet. Generate a notebook with auto-save enabled!")


def generate_profile_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a comprehensive profile summary of the dataset."""
    profile = {}
    
    # Basic info
    profile['total_rows'] = len(df)
    profile['total_columns'] = len(df.columns)
    profile['total_missing'] = df.isnull().sum().sum()
    profile['duplicate_rows'] = df.duplicated().sum()
    
    # Column categorization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    profile['numeric_columns'] = numeric_cols
    profile['categorical_columns'] = categorical_cols
    
    # Missing values details
    missing_summary = df.isnull().sum()
    profile['columns_with_missing'] = missing_summary[missing_summary > 0].index.tolist()
    profile['missing_values_summary'] = missing_summary.to_dict()
    
    # Data quality score
    completeness = (1 - profile['total_missing'] / (profile['total_rows'] * profile['total_columns'])) * 100
    uniqueness = (1 - profile['duplicate_rows'] / profile['total_rows']) * 100 if profile['total_rows'] > 0 else 100
    profile['overall_quality_score'] = (completeness + uniqueness) / 2
    
    # Statistical summary for numeric columns
    if numeric_cols:
        profile['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical summary
    if categorical_cols:
        cat_summary = {}
        for col in categorical_cols:
            cat_summary[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'value_counts': df[col].value_counts().head().to_dict()
            }
        profile['categorical_summary'] = cat_summary
    
    return profile


def recommend_template(profile_results: Dict, df: pd.DataFrame) -> str:
    """Recommend the most appropriate template based on data characteristics."""
    numeric_cols = len(profile_results['numeric_columns'])
    categorical_cols = len(profile_results['categorical_columns'])
    total_rows = profile_results['total_rows']
    total_cols = profile_results['total_columns']
    missing_ratio = profile_results['total_missing'] / (total_rows * total_cols) if total_rows * total_cols > 0 else 0
    
    # Check for time series data
    time_cols = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'time', 'timestamp', 'datetime'])]
    
    # Check for potential datetime columns by trying to parse
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(5)
                if len(sample) > 0:
                    pd.to_datetime(sample)
                    time_cols.append(col)
            except:
                pass
    
    # Recommendation logic with priority
    
    # 1. High missing values -> Data Cleaning
    if missing_ratio > 0.15:
        return 'data_cleaning'
    
    # 2. Time series data detected
    if time_cols and numeric_cols >= 1:
        return 'time_series_analysis'
    
    # 3. High dimensionality -> Dimensionality Reduction
    if numeric_cols > 15:
        return 'dimensionality_reduction'
    
    # 4. Look for classification target (categorical with few unique values)
    for col in profile_results['categorical_columns']:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 10:
            return 'classification_analysis'
    
    # 5. Look for regression target (numeric with many unique values)
    if numeric_cols >= 2:
        for col in profile_results['numeric_columns']:
            if df[col].nunique() > 20:
                return 'regression_analysis'
    
    # 6. Good for clustering (multiple numeric features, no obvious target)
    if numeric_cols >= 3 and categorical_cols <= 2:
        return 'clustering_analysis'
    
    # 7. Feature engineering if moderate complexity
    if numeric_cols >= 5 or categorical_cols >= 3:
        return 'feature_engineering'
    
    # 8. Default to EDA
    return 'exploratory_data_analysis'


def get_recommendation_reason(template: str, profile_results: Dict, df: pd.DataFrame) -> str:
    """Get the reason for template recommendation."""
    reasons = {
        'exploratory_data_analysis': "Good starting point for understanding your data",
        'data_cleaning': f"High missing value ratio ({profile_results['total_missing']} missing values detected)",
        'classification_analysis': "Categorical target variable detected with few unique values",
        'regression_analysis': "Numeric target variable suitable for prediction",
        'time_series_analysis': "Date/time columns detected for temporal analysis",
        'clustering_analysis': "Multiple numeric features suitable for segmentation",
        'dimensionality_reduction': f"High dimensionality ({len(profile_results['numeric_columns'])} numeric features)",
        'feature_engineering': "Complex dataset that would benefit from feature creation"
    }
    return reasons.get(template, "Based on data characteristics")


def get_template_features(template_type: str) -> list:
    """Get the key features of each template type."""
    features = {
        'exploratory_data_analysis': [
            "Data overview and statistics",
            "Distribution visualizations", 
            "Correlation analysis",
            "Missing values heatmap",
            "Data quality assessment"
        ],
        'data_cleaning': [
            "Missing values treatment",
            "Duplicate removal",
            "Outlier detection",
            "Data type optimization",
            "Memory usage reduction"
        ],
        'classification_analysis': [
            "Target variable analysis",
            "Feature selection",
            "Model comparison",
            "Performance metrics",
            "Confusion matrix"
        ],
        'regression_analysis': [
            "Continuous target analysis",
            "Feature correlation",
            "Linear regression models",
            "Residual analysis",
            "Performance metrics"
        ],
        'time_series_analysis': [
            "Temporal patterns",
            "Seasonality detection",
            "Trend analysis",
            "Forecasting models",
            "Time series visualization"
        ],
        'clustering_analysis': [
            "K-Means, Hierarchical, DBSCAN",
            "Elbow method & silhouette",
            "Cluster profiling",
            "Algorithm comparison",
            "Segment visualization"
        ],
        'dimensionality_reduction': [
            "PCA with variance analysis",
            "t-SNE visualization",
            "Feature loadings & importance",
            "Component interpretation",
            "ML-ready reduced datasets"
        ],
        'feature_engineering': [
            "Missing value imputation",
            "Categorical encoding",
            "Numerical transformations",
            "Feature creation & selection",
            "Scaling & final export"
        ]
    }
    
    return features.get(template_type, ["Advanced data analysis", "Custom workflows", "Professional insights"])


def get_template_sections(template_type: str) -> list:
    """Get the sections available for each template type."""
    sections = {
        'exploratory_data_analysis': [
            "📁 Data Loading",
            "🔍 Data Overview",
            "📊 Numeric Analysis",
            "🏷️ Categorical Analysis",
            "📈 Visualizations",
            "✅ Summary"
        ],
        'data_cleaning': [
            "📁 Data Loading",
            "🔍 Missing Values Analysis",
            "🔧 Missing Values Treatment",
            "🔄 Duplicate Handling",
            "📊 Outlier Detection",
            "🏷️ Data Type Optimization",
            "✅ Summary"
        ],
        'classification_analysis': [
            "📁 Data Loading & Exploration",
            "🎯 Target Variable Analysis",
            "📊 Feature Analysis",
            "🔧 Data Preprocessing",
            "🚂 Train-Test Split",
            "🤖 Model Training (8 algorithms)",
            "📈 Model Evaluation",
            "🔍 Feature Importance",
            "🎯 Predictions",
            "✅ Summary & Next Steps"
        ],
        'regression_analysis': [
            "📁 Data Loading & Exploration",
            "🎯 Target Variable Analysis",
            "📊 Feature Correlation",
            "🔧 Data Preprocessing",
            "🚂 Train-Test Split",
            "🤖 Model Training (7 algorithms)",
            "📈 Model Evaluation",
            "🔍 Feature Importance",
            "🎯 Predictions & Error Analysis",
            "✅ Summary & Next Steps"
        ],
        'time_series_analysis': [
            "📁 Data Loading & Date Parsing",
            "📊 Time Series Visualization",
            "📈 Trend & Seasonality Decomposition",
            "🔍 Stationarity Testing",
            "📊 Autocorrelation Analysis",
            "🚂 Train-Test Split",
            "🤖 Forecasting Models",
            "📈 Model Evaluation",
            "🔮 Future Predictions",
            "✅ Summary & Next Steps"
        ],
        'clustering_analysis': [
            "📁 Data Loading & Exploration",
            "🔧 Data Preprocessing",
            "📊 Optimal Cluster Selection",
            "🔵 K-Means Clustering",
            "🌳 Hierarchical Clustering",
            "🔷 DBSCAN Clustering",
            "📊 Algorithm Comparison",
            "📋 Cluster Profiling",
            "📈 Advanced Visualization",
            "✅ Summary & Next Steps"
        ],
        'dimensionality_reduction': [
            "📁 Data Loading & Exploration",
            "🔧 Data Preprocessing",
            "📊 PCA Analysis",
            "📈 Explained Variance Analysis",
            "🔮 t-SNE Visualization",
            "🔍 Feature Loadings Analysis",
            "🤖 Dimensionality Reduction for ML",
            "📊 Comparison of Methods",
            "🎨 Advanced Visualization",
            "✅ Summary & Recommendations"
        ],
        'feature_engineering': [
            "📁 Data Loading & Exploration",
            "🔧 Handling Missing Values",
            "🏷️ Encoding Categorical Variables",
            "📐 Numerical Transformations",
            "✨ Feature Creation",
            "⚖️ Feature Scaling",
            "🎯 Feature Selection",
            "⚖️ Handling Imbalanced Features",
            "📦 Final Feature Set",
            "✅ Summary & Export"
        ]
    }
    
    return sections.get(template_type, ["📁 Data Loading", "📊 Analysis", "✅ Summary"])