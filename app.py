"""
DataDojo Web Dashboard - Main Streamlit Application
Professional UI with Clean Design & High Contrast
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
from datadojo.web.notebook_interface import render_notebook_templates
from datadojo.web.help_interface import render_help_page
from datadojo.web.progress_interface import render_progress_dashboard
from datadojo.web.projects_interface import render_projects_page
from datadojo.web.styles import (
    get_modern_css, 
    create_hero_header,
    create_metric_card,
    create_feature_card,
    create_glass_card,
    create_badge,
    create_divider,
    create_stat_card
)

# Page configuration
st.set_page_config(
    page_title="DataDojo - Master Your Data",
    page_icon="🥋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply modern styles
st.markdown(get_modern_css(), unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables."""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = []
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    if 'profile_cache' not in st.session_state:
        st.session_state.profile_cache = {}
    if 'theme' not in st.session_state:
        st.session_state.theme = 'dark'

def load_datasets():
    """Load and cache available datasets."""
    if not st.session_state.datasets:
        with st.spinner("🔍 Discovering datasets..."):
            st.session_state.datasets = discover_datasets(['datasets', 'test_datasets'])
    return st.session_state.datasets


def format_size(size_mb):
    """Format file size nicely."""
    if size_mb < 1:
        return f"{size_mb*1024:.0f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb/1024:.1f} GB"


def show_home_page():
    """Display the modern home page with overview."""
    
    # Hero Section - Clean, professional
    st.markdown(create_hero_header(
        "DataDojo",
        "Master data science with intelligent exploration, profiling, and analysis tools"
    ), unsafe_allow_html=True)
    
    # Load datasets
    datasets = load_datasets()
    
    # Quick Stats Row
    if datasets:
        total_datasets = len(datasets)
        total_rows = sum(d.rows for d in datasets)
        total_size = sum(d.size_mb for d in datasets)
        domains = len(set(d.domain for d in datasets))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("#", str(total_datasets), "Datasets"), unsafe_allow_html=True)
        with col2:
            st.markdown(create_metric_card("~", f"{total_rows:,}", "Records"), unsafe_allow_html=True)
        with col3:
            st.markdown(create_metric_card("&gt;", format_size(total_size), "Volume"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_metric_card("@", str(domains), "Domains"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature Cards Section
    st.markdown("""
    <h2 style="text-align: center; margin: 2rem 0 1.5rem; color: #F8FAFC; font-size: 1.5rem; font-weight: 600;">
    Features
    </h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_feature_card(
            "{}",
            "Smart Profiling",
            "AI-powered data analysis with quality assessment and pattern detection."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_feature_card(
            "[]",
            "Notebook Templates",
            "Generate ready-to-run Jupyter notebooks for EDA, ML, and data cleaning."
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_feature_card(
            "++",
            "Progress Tracking",
            "Track your learning with XP, achievements, and skill progression."
        ), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_feature_card(
            "&lt;&gt;",
            "Data Generation",
            "Create realistic synthetic datasets for healthcare, finance, and e-commerce."
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_feature_card(
            "||",
            "Visualizations",
            "Intelligent chart recommendations based on your data types."
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_feature_card(
            "?!",
            "Learning Hub",
            "Interactive tutorials and guided workflows for all skill levels."
        ), unsafe_allow_html=True)
    
    st.markdown(create_divider(), unsafe_allow_html=True)
    
    # Charts Section
    if datasets:
        st.markdown("""
        <h2 style="text-align: center; margin: 2rem 0 1rem; color: #F8FAFC; font-size: 1.25rem; font-weight: 600;">
        Data Overview
        </h2>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Domain distribution - Modern donut chart
            domain_counts = {}
            for d in datasets:
                domain_counts[d.domain.title()] = domain_counts.get(d.domain.title(), 0) + 1
            
            if domain_counts:
                fig = go.Figure(data=[go.Pie(
                    labels=list(domain_counts.keys()),
                    values=list(domain_counts.values()),
                    hole=0.6,
                    marker=dict(
                        colors=['#3B82F6', '#14B8A6', '#8B5CF6', '#22C55E', '#F59E0B'],
                        line=dict(color='#0F172A', width=2)
                    ),
                    textinfo='label+percent',
                    textfont=dict(size=12, color='#F8FAFC'),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>"
                )])
                
                fig.update_layout(
                    title=dict(text="Datasets by Domain", font=dict(size=18, color='white')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True,
                    legend=dict(
                        font=dict(color='white'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    annotations=[dict(
                        text=f'<b>{len(datasets)}</b><br>Total',
                        x=0.5, y=0.5,
                        font_size=20,
                        font_color='white',
                        showarrow=False
                    )],
                    margin=dict(t=60, b=20, l=20, r=20)
                )
                st.plotly_chart(fig)
        
        with col2:
            # Size distribution - Modern bar chart
            size_data = []
            for d in sorted(datasets, key=lambda x: x.size_mb, reverse=True)[:8]:
                size_data.append({
                    'Dataset': d.name[:15] + ('...' if len(d.name) > 15 else ''),
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
                    color_discrete_sequence=['#FF6B35', '#00D9FF', '#A855F7', '#10B981']
                )
                
                fig.update_layout(
                    title=dict(text="Dataset Sizes", font=dict(size=18, color='white')),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(
                        tickangle=45,
                        tickfont=dict(color='rgba(255,255,255,0.7)'),
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        tickfont=dict(color='rgba(255,255,255,0.7)'),
                        gridcolor='rgba(255,255,255,0.1)'
                    ),
                    legend=dict(
                        font=dict(color='white'),
                        bgcolor='rgba(0,0,0,0)'
                    ),
                    margin=dict(t=60, b=80, l=50, r=20)
                )
                
                fig.update_traces(
                    marker=dict(
                        line=dict(width=0),
                        opacity=0.9
                    )
                )
                st.plotly_chart(fig)
        
        # Recent Datasets Table
        st.markdown("""
        <h3 style="color: white; margin: 2rem 0 1rem 0;">
        📋 Recent Datasets
        </h3>
        """, unsafe_allow_html=True)
        
        recent_datasets = sorted(datasets, key=lambda x: x.size_mb, reverse=True)[:10]
        
        table_data = []
        for d in recent_datasets:
            table_data.append({
                '📊 Name': d.name,
                '🏷️ Domain': d.domain.title(),
                '📈 Rows': f"{d.rows:,}",
                '📋 Columns': d.columns,
                '💾 Size': format_size(d.size_mb)
            })
        
        if table_data:
            st.dataframe(
                pd.DataFrame(table_data),
                hide_index=True
            )
    else:
        # Empty state
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🚀</div>
            <h2 style="color: white; margin-bottom: 1rem;">Ready to Get Started?</h2>
            <p style="color: rgba(255,255,255,0.7); margin-bottom: 2rem;">
                Generate your first dataset and begin your data science journey!
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_dataset_explorer():
    """Display the dataset explorer page with modern UI."""
    
    st.markdown("""
    <h1 style="color: #F8FAFC; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">
    Dataset Explorer
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem;">
    Browse, filter, and preview your datasets
    </p>
    """, unsafe_allow_html=True)
    
    datasets = load_datasets()
    
    if not datasets:
        st.warning("No datasets found. Generate some data first!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        domains = ['All'] + sorted(list(set(d.domain.title() for d in datasets)))
        selected_domain = st.selectbox("Filter by Domain", domains)
    
    with col2:
        min_rows = st.number_input("Minimum Rows", min_value=0, value=0)
    
    with col3:
        min_size = st.number_input("Minimum Size (MB)", min_value=0.0, value=0.0, step=0.1)
    
    # Apply filters
    filtered_datasets = datasets
    if selected_domain != 'All':
        filtered_datasets = [d for d in filtered_datasets if d.domain.title() == selected_domain]
    if min_rows > 0:
        filtered_datasets = [d for d in filtered_datasets if d.rows >= min_rows]
    if min_size > 0:
        filtered_datasets = [d for d in filtered_datasets if d.size_mb >= min_size]
    
    st.markdown(f"""
    <p style="color: #CBD5E1; margin: 1rem 0;">
    Found <span style="color: #3B82F6; font-weight: 600;">{len(filtered_datasets)}</span> datasets matching your criteria
    </p>
    """, unsafe_allow_html=True)
    
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
                        st.dataframe(df.head(100), use_container_width=True)
                        
                        # Quick stats
                        st.subheader("Quick Statistics")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Numeric Columns:**")
                            numeric_cols = df.select_dtypes(include=['number']).columns
                            if len(numeric_cols) > 0:
                                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                            else:
                                st.write("No numeric columns found")
                        
                        with col2:
                            st.write("**Missing Values:**")
                            missing_data = df.isnull().sum()
                            missing_data = missing_data[missing_data > 0]
                            if len(missing_data) > 0:
                                st.dataframe(missing_data.to_frame("Missing Count"), use_container_width=True)
                            else:
                                st.success("No missing values found!")
                                
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")

def show_data_profiler():
    """Display the data profiler page with modern UI."""
    
    st.markdown("""
    <h1 style="color: #F8FAFC; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">
    Data Profiler
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem;">
    AI-powered data quality assessment and insights
    </p>
    """, unsafe_allow_html=True)
    
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
    
    if st.button("Profile Dataset", type="primary"):
        dataset = datasets[selected_idx]
        
        with st.spinner(f"Profiling {dataset.name}..."):
            try:
                # Load and profile the dataset
                df = pd.read_csv(dataset.path)
                profiler = IntelligentProfiler()
                viz_engine = DataVisualizationEngine()
                profile = profiler.profile_dataset(df, dataset.name)
                
                # Cache the result
                st.session_state.profile_cache[dataset.path] = profile
                
                # Display results
                st.success(f"Profile completed for {dataset.name}")
                
                # Enhanced quality dashboard
                st.markdown(create_data_quality_summary_card(profile), unsafe_allow_html=True)
                
                # Quality visualization dashboard
                quality_figs = viz_engine.create_quality_dashboard(profile)
                
                if quality_figs:
                    col1, col2 = st.columns(2)
                    with col1:
                        if len(quality_figs) > 0:
                            st.plotly_chart(quality_figs[0], use_container_width=True)
                    with col2:
                        if len(quality_figs) > 1:
                            st.plotly_chart(quality_figs[1], use_container_width=True)
                
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
                                            st.plotly_chart(fig, use_container_width=True)
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
                st.dataframe(col_df, use_container_width=True)
                
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
                        st.plotly_chart(fig, use_container_width=True)
                
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
    """Display the data generator page with modern UI."""
    
    st.markdown("""
    <h1 style="color: #F8FAFC; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">
    Data Generator
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-bottom: 1.5rem;">
    Create realistic synthetic datasets for learning and testing
    </p>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: #1E293B; border: 1px solid #334155; 
        border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
            <h4 style="color: #3B82F6; margin-bottom: 0.5rem; font-size: 1rem;">Generation Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
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
        st.markdown("""
        <div style="background: #1E293B; border: 1px solid #334155; 
        border-radius: 8px; padding: 1.25rem; margin-bottom: 1rem;">
            <h4 style="color: #14B8A6; margin-bottom: 0.5rem; font-size: 1rem;">What You'll Get</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if domain == "All Domains" or domain == "Healthcare":
            st.markdown("""
            <div style="background: #1E293B; padding: 0.875rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #22C55E;">
            <strong style="color: #F8FAFC;">Healthcare</strong><br>
            <span style="color: #94A3B8; font-size: 0.85rem;">Patient demographics, lab results, medical records</span>
            </div>
            """, unsafe_allow_html=True)
            
        if domain == "All Domains" or domain == "E-commerce":
            st.markdown("""
            <div style="background: #1E293B; padding: 0.875rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #3B82F6;">
            <strong style="color: #F8FAFC;">E-commerce</strong><br>
            <span style="color: #94A3B8; font-size: 0.85rem;">Customer profiles, transactions, interactions</span>
            </div>
            """, unsafe_allow_html=True)
            
        if domain == "All Domains" or domain == "Finance":
            st.markdown("""
            <div style="background: #1E293B; padding: 0.875rem; border-radius: 6px; margin-bottom: 0.75rem; border-left: 3px solid #8B5CF6;">
            <strong style="color: #F8FAFC;">Finance</strong><br>
            <span style="color: #94A3B8; font-size: 0.85rem;">Bank transactions, credit applications, stock data</span>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Generate Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
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
    init_session_state()
    
    # Sidebar - Clean Professional Design
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1.25rem 0; border-bottom: 1px solid #334155; margin-bottom: 1rem;">
            <h2 style="color: #F8FAFC; font-size: 1.5rem; margin: 0; font-weight: 700;">
            DataDojo
            </h2>
            <p style="color: #94A3B8; font-size: 0.75rem; margin-top: 0.25rem;">
            Data Science Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "Navigation",
            [
                "Home",
                "Learning Projects",
                "Dataset Explorer", 
                "Data Profiler",
                "Data Generator",
                "Notebook Templates",
                "Progress Dashboard",
                "Tutorial & Help"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #334155;'></div>", unsafe_allow_html=True)
        
        # Quick Stats in Sidebar
        datasets = load_datasets()
        if datasets:
            st.markdown("""
            <p style="color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; 
            letter-spacing: 1px; margin-bottom: 0.5rem; font-weight: 500;">Statistics</p>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1E293B; padding: 0.625rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid #334155;">
                <span style="color: #3B82F6; font-weight: 600;">{len(datasets)}</span>
                <span style="color: #CBD5E1;"> datasets</span>
            </div>
            """, unsafe_allow_html=True)
            
            total_rows = sum(d.rows for d in datasets)
            st.markdown(f"""
            <div style="background: #1E293B; padding: 0.625rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid #334155;">
                <span style="color: #14B8A6; font-weight: 600;">{total_rows:,}</span>
                <span style="color: #CBD5E1;"> records</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin: 1rem 0; border-top: 1px solid #334155;'></div>", unsafe_allow_html=True)
        
        # Tips
        st.markdown("""
        <p style="color: #94A3B8; font-size: 0.7rem; text-transform: uppercase; 
        letter-spacing: 1px; margin-bottom: 0.5rem; font-weight: 500;">Tips</p>
        <ul style="color: #94A3B8; font-size: 0.8rem; padding-left: 1rem; margin: 0;">
            <li style="margin-bottom: 0.25rem;">Use Notebook Templates for quick starts</li>
            <li style="margin-bottom: 0.25rem;">Track progress to unlock achievements</li>
            <li>Generate synthetic data for practice</li>
        </ul>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>" * 2, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.7rem;">
        v2.0
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if page == "Home":
        show_home_page()
    elif page == "Learning Projects":
        render_projects_page()
    elif page == "Dataset Explorer":
        show_dataset_explorer()
    elif page == "Data Profiler":
        show_data_profiler()
    elif page == "Data Generator":
        show_data_generator()
    elif page == "Notebook Templates":
        render_notebook_templates()
    elif page == "Progress Dashboard":
        render_progress_dashboard()
    elif page == "Tutorial & Help":
        render_help_page()

if __name__ == "__main__":
    main()