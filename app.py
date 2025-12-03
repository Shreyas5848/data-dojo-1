"""
DataDojo Web Dashboard - Main Streamlit Application
Professional web interface for interactive data exploration and profiling.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path
from datetime import datetime
import json
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Environment detection
IS_CLOUD_DEPLOYMENT = os.getenv('STREAMLIT_SHARING_MODE', False) or 'streamlit.io' in os.getenv('HOME', '')

# Import DataDojo components
from datadojo.cli.list_datasets import discover_datasets, DatasetInfo
from datadojo.cli.profile_data import profile_data
from datadojo.cli.generate_data import generate_data
from datadojo.utils.intelligent_profiler import IntelligentProfiler, quick_profile
from datadojo.utils.synthetic_data_generator import SyntheticDataGenerator
from datadojo.web.visualizations import DataVisualizationEngine, create_data_quality_summary_card
from datadojo.web.config import apply_theme
from datadojo.web.notebook_interface import render_notebook_templates
from datadojo.web.help_interface import render_help_page
from datadojo.web.progress_interface import render_progress_dashboard

# Page configuration
st.set_page_config(
    page_title="DataDojo Dashboard",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF9900;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FF9900, #FFB366);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .dataset-card {
        border: 2px solid #FF9900;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background: #FFFAF0;
    }
    
    .quality-high { color: #4CAF50; font-weight: bold; }
    .quality-medium { color: #FF9800; font-weight: bold; }
    .quality-low { color: #F44336; font-weight: bold; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #FFE0B2, #FFCC80);
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = []
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'profile_cache' not in st.session_state:
        st.session_state.profile_cache = {}

def load_datasets():
    """Load and cache available datasets."""
    if not st.session_state.datasets:
        with st.spinner("🔍 Discovering datasets..."):
            st.session_state.datasets = discover_datasets(['datasets', 'test_datasets'])
    return st.session_state.datasets

def get_quality_class(score):
    """Get CSS class based on quality score."""
    if score >= 0.8:
        return "quality-high"
    elif score >= 0.6:
        return "quality-medium"
    else:
        return "quality-low"

def format_size(size_mb):
    """Format file size nicely."""
    if size_mb < 1:
        return f"{size_mb*1024:.0f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb/1024:.1f} GB"

def create_overview_metrics(datasets):
    """Create overview metrics cards."""
    if not datasets:
        return
        
    total_datasets = len(datasets)
    total_rows = sum(d.rows for d in datasets)
    total_size = sum(d.size_mb for d in datasets)
    
    domains = {}
    for d in datasets:
        domains[d.domain] = domains.get(d.domain, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Datasets</h3>
            <h2>{total_datasets}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Total Records</h3>
            <h2>{total_rows:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💾 Total Size</h3>
            <h2>{format_size(total_size)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏷️ Domains</h3>
            <h2>{len(domains)}</h2>
        </div>
        """, unsafe_allow_html=True)

def show_home_page():
    """Display the home page with overview."""
    st.markdown('<h1 class="main-header">🥋 DataDojo Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Your AI-powered data exploration and profiling platform
    </div>
    """, unsafe_allow_html=True)
    
    datasets = load_datasets()
    
    if not datasets:
        st.warning("No datasets found. Generate some data using the Data Generator page!")
        return
    
    create_overview_metrics(datasets)
    
    # New Feature: Notebook Templates
    st.markdown("---")
    st.subheader("🎉 NEW: Notebook Templates")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **📓 Transform data insights into interactive Jupyter notebooks!**
        
        **Option 3 is now LIVE** - Generate professional analysis notebooks:
        • **Smart Templates** - EDA, Data Cleaning, Classification, Regression
        • **Auto-Generated** - Pre-populated with your data characteristics
        • **Ready-to-Run** - Complete analysis code included
        • **Educational** - Learn data science through hands-on examples
        • **Customizable** - Modify templates for your specific needs
        
        Perfect bridge between data profiling and advanced analysis!
        """)
    
    with col2:
        st.markdown("""
        <div style="margin-top: 2rem;">
        <p style="text-align: center; color: #FF6B35; font-weight: bold;">
        👈 Try "📓 Notebook Templates" now!
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Domain distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Datasets by Domain")
        domain_counts = {}
        for d in datasets:
            domain_counts[d.domain.title()] = domain_counts.get(d.domain.title(), 0) + 1
        
        if domain_counts:
            fig = px.pie(
                values=list(domain_counts.values()),
                names=list(domain_counts.keys()),
                color_discrete_sequence=px.colors.sequential.Oranges_r
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📈 Dataset Sizes")
        size_data = []
        for d in datasets:
            size_data.append({
                'Dataset': d.name[:20] + ('...' if len(d.name) > 20 else ''),
                'Size (MB)': d.size_mb,
                'Domain': d.domain.title()
            })
        
        if size_data:
            df_sizes = pd.DataFrame(size_data)
            fig = px.bar(
                df_sizes, 
                x='Dataset', 
                y='Size (MB)',
                color='Domain',
                color_discrete_sequence=px.colors.sequential.Oranges_r
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, width='stretch')
    
    # Recent datasets table
    st.subheader("📋 Recent Datasets")
    recent_datasets = sorted(datasets, key=lambda x: x.size_mb, reverse=True)[:10]
    
    table_data = []
    for d in recent_datasets:
        table_data.append({
            'Name': d.name,
            'Domain': d.domain.title(),
            'Rows': f"{d.rows:,}",
            'Columns': d.columns,
            'Size': format_size(d.size_mb),
            'Path': d.path
        })
    
    if table_data:
        st.dataframe(pd.DataFrame(table_data), width='stretch')

def show_dataset_explorer():
    """Display the dataset explorer page."""
    st.title("📁 Dataset Explorer")
    
    datasets = load_datasets()
    
    if not datasets:
        st.warning("No datasets found. Generate some data first!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        domains = ['All'] + sorted(list(set(d.domain.title() for d in datasets)))
        selected_domain = st.selectbox("🏷️ Filter by Domain", domains)
    
    with col2:
        min_rows = st.number_input("📊 Minimum Rows", min_value=0, value=0)
    
    with col3:
        min_size = st.number_input("💾 Minimum Size (MB)", min_value=0.0, value=0.0, step=0.1)
    
    # Apply filters
    filtered_datasets = datasets
    if selected_domain != 'All':
        filtered_datasets = [d for d in filtered_datasets if d.domain.title() == selected_domain]
    if min_rows > 0:
        filtered_datasets = [d for d in filtered_datasets if d.rows >= min_rows]
    if min_size > 0:
        filtered_datasets = [d for d in filtered_datasets if d.size_mb >= min_size]
    
    st.write(f"Found **{len(filtered_datasets)}** datasets matching your criteria")
    
    # Dataset cards
    for dataset in filtered_datasets:
        with st.expander(f"📊 {dataset.name} ({dataset.domain.title()})"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Rows", f"{dataset.rows:,}")
            with col2:
                st.metric("Columns", dataset.columns)
            with col3:
                st.metric("Size", format_size(dataset.size_mb))
            with col4:
                st.metric("Domain", dataset.domain.title())
            
            st.write(f"**Path:** `{dataset.path}`")
            
            col1, col2 = st.columns(2)
            with col1:
                # Use path hash for unique keys to avoid duplicates
                path_hash = hash(str(dataset.path))
                if st.button(f"🔍 Profile Dataset", key=f"profile_{abs(path_hash)}"):
                    st.session_state.selected_dataset = dataset
                    st.rerun()
            
            with col2:
                if st.button(f"📊 View Data", key=f"view_{abs(path_hash)}"):
                    try:
                        df = pd.read_csv(dataset.path)
                        st.subheader(f"Data Preview: {dataset.name}")
                        st.dataframe(df.head(100), width='stretch')
                        
                        # Quick stats
                        st.subheader("Quick Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Numeric Columns:**")
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.dataframe(df[numeric_cols].describe(), width='stretch')
                            else:
                                st.write("No numeric columns found")
                        
                        with col2:
                            st.write("**Missing Values:**")
                            missing_data = df.isnull().sum()
                            missing_data = missing_data[missing_data > 0]
                            if len(missing_data) > 0:
                                st.dataframe(missing_data.to_frame("Missing Count"), width='stretch')
                            else:
                                st.success("No missing values found!")
                                
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")

def show_data_profiler():
    """Display the data profiler page."""
    st.title("🔍 Data Profiler")
    
    datasets = load_datasets()
    
    if not datasets:
        st.warning("No datasets found. Generate some data first!")
        return
    
    # Dataset selection
    dataset_names = [f"{d.name} ({d.domain})" for d in datasets]
    selected_idx = st.selectbox(
        "Select a dataset to profile",
        range(len(dataset_names)),
        format_func=lambda x: dataset_names[x]
    )
    
    if st.button("🚀 Profile Dataset", type="primary"):
        dataset = datasets[selected_idx]
        
        with st.spinner(f"Profiling {dataset.name}... This may take a moment."):
            try:
                # Load and profile the dataset
                df = pd.read_csv(dataset.path)
                profiler = IntelligentProfiler()
                viz_engine = DataVisualizationEngine()
                profile = profiler.profile_dataset(df, dataset.name)
                
                # Cache the result
                st.session_state.profile_cache[dataset.path] = profile
                
                # Display results
                st.success(f"✅ Profile completed for {dataset.name}")
                
                # Enhanced quality dashboard
                st.markdown(create_data_quality_summary_card(profile), unsafe_allow_html=True)
                
                # Quality visualization dashboard
                quality_figs = viz_engine.create_quality_dashboard(profile)
                
                if quality_figs:
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(quality_figs) > 0:
                            st.plotly_chart(quality_figs[0], width='stretch')
                    with col2:
                        if len(quality_figs) > 1:
                            st.plotly_chart(quality_figs[1], width='stretch')
                
                # Quality metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Completeness", f"{profile.completeness_score:.1%}")
                with col2:
                    st.metric("Consistency", f"{profile.consistency_score:.1%}")
                with col3:
                    st.metric("Uniqueness", f"{profile.uniqueness_score:.1%}")
                
                # Dataset overview
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Dataset Overview")
                    st.write(f"**Shape:** {profile.shape[0]:,} rows × {profile.shape[1]} columns")
                    st.write(f"**Memory Usage:** {profile.memory_usage_mb:.1f} MB")
                    st.write(f"**Duplicate Rows:** {profile.duplicate_rows:,} ({profile.duplicate_percentage:.1f}%)")
                
                with col2:
                    st.subheader("💡 Business Insights")
                    for insight in profile.business_insights[:5]:
                        st.write(f"• {insight}")
                
                # Automatic visualization recommendations
                st.subheader("📊 Recommended Visualizations")
                viz_recommendations = viz_engine.recommend_visualizations(df)
                
                if viz_recommendations:
                    # Show first few recommendations
                    num_to_show = min(4, len(viz_recommendations))
                    
                    for i in range(0, num_to_show, 2):
                        cols = st.columns(2)
                        
                        for j, col in enumerate(cols):
                            if i + j < len(viz_recommendations):
                                rec = viz_recommendations[i + j]
                                with col:
                                    with st.expander(f"📈 {rec['title']}", expanded=(i + j < 2)):
                                        st.write(rec['description'])
                                        try:
                                            fig = viz_engine.create_visualization(df, rec)
                                            st.plotly_chart(fig, width='stretch')
                                        except Exception as e:
                                            st.error(f"Could not create visualization: {str(e)}")
                    
                    if len(viz_recommendations) > num_to_show:
                        with st.expander(f"View {len(viz_recommendations) - num_to_show} more recommendations"):
                            for rec in viz_recommendations[num_to_show:]:
                                st.write(f"**{rec['title']}**: {rec['description']}")
                
                # Column analysis
                st.subheader("📋 Column Analysis")
                
                col_data = []
                for col_name, col_profile in profile.column_profiles.items():
                    col_data.append({
                        'Column': col_name,
                        'Type': col_profile.dtype,
                        'Missing %': f"{col_profile.null_percentage:.1f}%",
                        'Unique %': f"{col_profile.unique_percentage:.1f}%",
                        'Quality': f"{col_profile.data_quality_score:.1%}",
                        'Issues': len(col_profile.quality_issues)
                    })
                
                col_df = pd.DataFrame(col_data)
                st.dataframe(col_df, width='stretch')
                
                # Recommendations
                if profile.recommendations:
                    st.subheader("🚀 Recommendations")
                    for i, rec in enumerate(profile.recommendations, 1):
                        st.write(f"{i}. {rec}")
                
                # Column-specific recommendations
                col_recs = []
                for col_name, col_profile in profile.column_profiles.items():
                    if col_profile.recommendations:
                        for rec in col_profile.recommendations:
                            col_recs.append(f"**{col_name}**: {rec}")
                
                if col_recs:
                    st.subheader("🔧 Column-Specific Recommendations")
                    for rec in col_recs[:10]:  # Show first 10
                        st.write(f"• {rec}")
                
                # Visualizations
                st.subheader("📈 Data Quality Visualizations")
                
                # Missing data heatmap
                if profile.shape[1] <= 20:  # Only show for reasonable number of columns
                    missing_data = []
                    for col_name, col_profile in profile.column_profiles.items():
                        missing_data.append(col_profile.null_percentage)
                    
                    if any(m > 0 for m in missing_data):
                        fig = go.Figure(data=go.Bar(
                            x=list(profile.column_profiles.keys()),
                            y=missing_data,
                            marker_color='#FF9900'
                        ))
                        fig.update_layout(
                            title="Missing Data by Column",
                            xaxis_title="Columns",
                            yaxis_title="Missing Percentage (%)"
                        )
                        st.plotly_chart(fig, width='stretch')
                
                # Export options
                st.subheader("📤 Export Profile")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📄 Download Text Report"):
                        report = profiler.generate_report(profile)
                        st.download_button(
                            label="📄 Download Report",
                            data=report,
                            file_name=f"{dataset.name}_profile_report.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    if st.button("📊 Download JSON Profile"):
                        # Convert profile to JSON-serializable format
                        profile_dict = {
                            'dataset_name': profile.name,
                            'overall_quality_score': profile.overall_quality_score,
                            'completeness_score': profile.completeness_score,
                            'consistency_score': profile.consistency_score,
                            'uniqueness_score': profile.uniqueness_score,
                            'shape': profile.shape,
                            'memory_usage_mb': profile.memory_usage_mb,
                            'recommendations': profile.recommendations,
                            'business_insights': profile.business_insights
                        }
                        
                        json_data = json.dumps(profile_dict, indent=2)
                        st.download_button(
                            label="📊 Download JSON",
                            data=json_data,
                            file_name=f"{dataset.name}_profile.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"Error profiling dataset: {str(e)}")

def show_data_generator():
    """Display the data generator page."""
    st.title("⚡ Data Generator")
    
    st.markdown("Generate realistic synthetic datasets for learning and testing purposes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎛️ Generation Settings")
        
        domain = st.selectbox(
            "Select Domain",
            ["All Domains", "Healthcare", "E-commerce", "Finance"],
            help="Choose which domain to generate data for"
        )
        
        size = st.selectbox(
            "Dataset Size",
            ["Small", "Medium", "Large"],
            index=1,
            help="Small: ~500-2K records, Medium: ~1K-5K records, Large: ~2K-10K records"
        )
        
        output_dir = st.text_input(
            "Output Directory",
            value="generated_datasets",
            help="Directory where the generated files will be saved"
        )
        
        seed = st.number_input(
            "Random Seed",
            value=42,
            help="Set seed for reproducible results"
        )
    
    with col2:
        st.subheader("📋 What You'll Get")
        
        if domain == "All Domains" or domain == "Healthcare":
            st.write("**🏥 Healthcare:**")
            st.write("• Patient Demographics (age, gender, medical history)")
            st.write("• Lab Results (blood tests, reference ranges)")
            
        if domain == "All Domains" or domain == "E-commerce":
            st.write("**🛒 E-commerce:**")
            st.write("• Customer Profiles (demographics, preferences)")
            st.write("• Transaction Records (purchases, payments)")
            
        if domain == "All Domains" or domain == "Finance":
            st.write("**💰 Finance:**")
            st.write("• Bank Transactions (deposits, withdrawals)")
            st.write("• Credit Applications (income, credit scores)")
    
    if st.button("🚀 Generate Data", type="primary"):
        with st.spinner("Generating synthetic data... This may take a moment."):
            try:
                # Convert domain selection
                domain_param = None if domain == "All Domains" else domain.lower()
                
                result = generate_data(
                    domain=domain_param,
                    size=size.lower(),
                    output_dir=output_dir,
                    seed=seed
                )
                
                if result.success:
                    st.success("✅ Data generation completed!")
                    st.text(result.output)
                    
                    # Refresh datasets cache
                    st.session_state.datasets = []
                    
                    st.balloons()
                else:
                    st.error(f"❌ Generation failed: {result.error_message}")
                    
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")

def main():
    """Main application function."""
    # Apply dark theme
    apply_theme()
    
    # Apply additional dark theme CSS
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117 !important;
        }
        .stApp > div {
            background-color: #0E1117 !important;
        }
        /* Make all text light colored */
        .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
            color: #FAFAFA !important;
        }
        /* Style dataframes for dark theme */
        .stDataFrame {
            background-color: #262730 !important;
        }
        /* Orange glow effects for buttons and interactive elements */
        .stButton > button {
            background: linear-gradient(135deg, #FF9900, #FFB366) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 8px rgba(255, 153, 0, 0.3) !important;
        }
        .stButton > button:hover {
            box-shadow: 0 6px 12px rgba(255, 153, 0, 0.5) !important;
            transform: translateY(-2px);
        }
    </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.markdown("## 🥋 DataDojo")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "🏠 Home",
            "📁 Dataset Explorer", 
            "🔍 Data Profiler",
            "⚡ Data Generator",
            "📓 Notebook Templates",
            "📊 Progress Dashboard",
            "📚 Tutorial & Help"
        ]
    )
    
    # Navigation routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Dataset Explorer":
        show_dataset_explorer()
    elif page == "🔍 Data Profiler":
        show_data_profiler()
    elif page == "⚡ Data Generator":
        show_data_generator()
    elif page == "📓 Notebook Templates":
        render_notebook_templates()
    elif page == "📊 Progress Dashboard":
        render_progress_dashboard()
    elif page == "📚 Tutorial & Help":
        render_help_page()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Quick Tips")
    st.sidebar.markdown("• Use **Data Generator** to create sample datasets")
    st.sidebar.markdown("• **Profile** datasets to understand data quality")
    st.sidebar.markdown("• Generate **Notebook Templates** for analysis")
    st.sidebar.markdown("• Track your learning in **Progress Dashboard**")
    st.sidebar.markdown("• Check **Tutorial & Help** for guidance")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built with ❤️ using Streamlit**")

if __name__ == "__main__":
    main()