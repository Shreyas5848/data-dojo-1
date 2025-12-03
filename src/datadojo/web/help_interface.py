"""
Tutorial & Help Interface
Streamlit interface for guiding new users through the app features
"""

import streamlit as st


def render_help_page():
    """Render the Tutorial & Help page with modern styling."""
    
    # Professional header
    st.markdown("""
    <h1 style="color: #F8FAFC; font-size: 1.75rem; font-weight: 700; margin-bottom: 0.5rem;">
    Tutorial & Help
    </h1>
    <p style="color: #94A3B8; font-size: 1rem; margin-bottom: 2rem;">
    Your guide to mastering data science with DataDojo
    </p>
    """, unsafe_allow_html=True)
    
    # Welcome Card
    st.markdown("""
    <div style="background: #1E293B; border: 1px solid #334155; 
    border-radius: 8px; padding: 1.25rem; margin-bottom: 1.5rem;">
        <h3 style="color: #3B82F6; margin: 0 0 0.5rem 0; font-size: 1rem;">Welcome to DataDojo</h3>
        <p style="color: #CBD5E1; margin: 0; font-size: 0.875rem;">
        This guide will help you get started and make the most of all features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("""
    <h2 style="color: #F8FAFC; font-size: 1.25rem; font-weight: 600; margin: 1.5rem 0 1rem 0;">
    Quick Start Guide
    </h2>
    """, unsafe_allow_html=True)
    
    with st.expander("**Step 1: Generate or Upload Data**", expanded=True):
        st.markdown("""
        ### Getting Data into DataDojo
        
        **Option A: Generate Sample Data**
        1. Go to **Data Generator** in the sidebar
        2. Select a domain (Healthcare, E-commerce, Finance)
        3. Choose dataset size
        4. Click **Generate Data**
        
        **Option B: Upload Your Own Data**
        1. Go to **Notebook Templates**
        2. Use the file uploader to upload a CSV file
        3. Or click **Use Demo Data** for a quick test
        
        **Tip:** Start with demo data to explore features quickly.
        """)
    
    with st.expander("**Step 2: Explore Your Data**"):
        st.markdown("""
        ### Understanding Your Dataset
        
        1. Go to **Dataset Explorer**
        2. Browse available datasets by domain
        3. Use filters to find specific datasets
        4. Click **View Data** to see a preview
        
        **What you'll see:**
        - Dataset dimensions (rows x columns)
        - Data types for each column
        - Sample values
        - Quick statistics
        
        **Tip:** Look for datasets with issues (missing values, outliers) to practice cleaning.
        """)
    
    with st.expander("**Step 3: Profile Your Data**"):
        st.markdown("""
        ### Deep Data Analysis
        
        1. Go to **Data Profiler**
        2. Select a dataset from the dropdown
        3. Click **Profile Dataset**
        
        **What you'll get:**
        - Data quality scores (Completeness, Consistency, Uniqueness)
        - Automatic visualizations
        - Business insights
        - Improvement recommendations
        - Column-by-column analysis
        
        **Tip:** Pay attention to the recommendations - they guide your next steps.
        """)
    
    with st.expander("**Step 4: Generate Analysis Notebooks**"):
        st.markdown("""
        ### Create Jupyter Notebooks
        
        1. Go to **Notebook Templates**
        2. Upload data or use demo data
        3. Review the auto-generated data profile
        4. Select a template type (see recommendations)
        5. Optionally customize which sections to include
        6. Click **Generate Notebook**
        7. Preview and download your notebook
        
        **Available Templates:**
        - Exploratory Data Analysis
        - Data Cleaning
        - Classification
        - Regression
        - Time Series
        - Clustering
        - Dimensionality Reduction
        - Feature Engineering
        
        **Tip:** The app recommends the best template based on your data.
        """)
    
    st.markdown("---")
    
    # Feature Guide
    st.markdown("<h2 style='color: #F8FAFC; font-size: 1.25rem; font-weight: 600;'>Feature Guide</h2>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Home", "Explorer", "Profiler", "Notebooks"])
    
    with tab1:
        st.markdown("""
        ### Home Page
        
        The home page gives you an overview of your data ecosystem:
        
        | Feature | Description |
        |---------|-------------|
        | **Total Datasets** | Number of datasets available |
        | **Total Records** | Combined row count across all datasets |
        | **Total Size** | Storage used by all datasets |
        | **Domains** | Number of data domains (Healthcare, Finance, etc.) |
        
        **Charts:**
        - **Pie Chart**: Distribution of datasets by domain
        - **Bar Chart**: Dataset sizes comparison
        - **Recent Datasets Table**: Quick access to your data
        """)
    
    with tab2:
        st.markdown("""
        ### Dataset Explorer
        
        Browse and filter your datasets:
        
        **Filters:**
        - **Domain**: Healthcare, E-commerce, Finance, etc.
        - **Minimum Rows**: Filter by dataset size
        - **Minimum Size**: Filter by file size
        
        **Actions:**
        - **Profile Dataset**: Deep analysis
        - **View Data**: Quick preview with stats
        
        **Pro Tips:**
        - Use filters to find datasets with specific characteristics
        - Check "Missing Values" section for data quality issues
        """)
    
    with tab3:
        st.markdown("""
        ### Data Profiler
        
        Get comprehensive insights about your data:
        
        **Quality Metrics:**
        - **Completeness**: % of non-null values
        - **Consistency**: Data format consistency
        - **Uniqueness**: Duplicate detection
        
        **Analysis Includes:**
        - Column-by-column statistics
        - Missing data visualization
        - Distribution analysis
        - Correlation detection
        - Outlier identification
        
        **Outputs:**
        - Text report (downloadable)
        - JSON profile (for automation)
        """)
    
    with tab4:
        st.markdown("""
        ### Notebook Templates
        
        Generate professional Jupyter notebooks:
        
        **Workflow:**
        1. **Load Data** - Upload CSV or use demo
        2. **Analyze** - Automatic profiling
        3. **Choose Template** - 8 template types
        4. **Customize** - Select sections to include
        5. **Generate** - Create notebook
        6. **Download** - Get .ipynb file
        
        **Template Types:**
        
        | Template | Best For |
        |----------|----------|
        | EDA | Initial data exploration |
        | Data Cleaning | Messy data with missing values |
        | Classification | Categorical target prediction |
        | Regression | Numeric target prediction |
        | Time Series | Date/time based data |
        | Clustering | Finding natural groupings |
        | Dim. Reduction | High-dimensional data |
        | Feature Engineering | Building ML features |
        """)
    
    st.markdown("---")
    
    # FAQ Section
    st.markdown("<h2 style='color: #F8FAFC; font-size: 1.25rem; font-weight: 600;'>Frequently Asked Questions</h2>", unsafe_allow_html=True)
    
    with st.expander("How do I choose the right template?"):
        st.markdown("""
        The app automatically recommends templates based on your data. Here's the logic:
        
        - **High missing values (>15%)** - Data Cleaning
        - **Date/time columns detected** - Time Series
        - **Many numeric features (>15)** - Dimensionality Reduction
        - **Categorical target (2-10 values)** - Classification
        - **Numeric target (many values)** - Regression
        - **Multiple numeric features** - Clustering
        - **Complex dataset** - Feature Engineering
        - **Default** - EDA (Exploratory Data Analysis)
        
        You can always override the recommendation.
        """)
    
    with st.expander("What file formats are supported?"):
        st.markdown("""
        Currently, DataDojo supports:
        
        - **CSV** (Comma-Separated Values) - Primary format
        - **Generated datasets** - From the Data Generator
        
        **Coming Soon:**
        - Excel (.xlsx, .xls)
        - JSON
        - Parquet
        """)
    
    with st.expander("How do I run the generated notebooks?"):
        st.markdown("""
        1. **Download** the .ipynb file from DataDojo
        2. **Install Jupyter** (if not already):
           ```bash
           pip install jupyter notebook
           ```
        3. **Launch Jupyter**:
           ```bash
           jupyter notebook
           ```
        4. **Open** your downloaded notebook
        5. **Update** the data path in the first cell
        6. **Run** all cells (Cell → Run All)
        
        **Common issues:**
        - Missing libraries: Install with `pip install library_name`
        - Data path wrong: Update the file path in the loading cell
        """)
    
    with st.expander("Can I customize the notebook templates?"):
        st.markdown("""
        Yes! There are two ways:
        
        **1. Before Generation (in DataDojo):**
        - Expand "Advanced Options" in Step 3.5
        - Uncheck sections you don't need
        - Only selected sections will be included
        
        **2. After Generation (in Jupyter):**
        - Open the downloaded notebook
        - Add, modify, or delete cells as needed
        - The generated code is fully editable
        
        The templates are meant as starting points, not final products!
        """)
    
    with st.expander("What do the quality scores mean?"):
        st.markdown("""
        **Completeness Score (0-100%)**
        - Measures: Percentage of non-null values
        - 100% = No missing values
        - <80% = Consider data cleaning
        
        **Consistency Score (0-100%)**
        - Measures: Data format consistency
        - Checks: Date formats, numeric patterns, categorical values
        - Low score = Mixed formats in columns
        
        **Uniqueness Score (0-100%)**
        - Measures: Duplicate detection
        - 100% = No duplicate rows
        - Low score = Many duplicate entries
        
        **Overall Quality Score**
        - Weighted average of all three
        - >80% = Good quality
        - 60-80% = Needs attention
        - <60% = Significant issues
        """)
    
    st.markdown("---")
    
    # Keyboard Shortcuts
    st.header("⌨️ Tips & Tricks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 💡 Pro Tips
        
        1. **Use Demo Data First** - Explore features without uploading
        2. **Check Recommendations** - The app learns from your data
        3. **Download Both Formats** - Text for reading, JSON for automation
        4. **Customize Templates** - Uncheck sections you don't need
        5. **Run Cells Sequentially** - Notebooks build on previous cells
        """)
    
    with col2:
        st.markdown("""
        ### 🔧 Troubleshooting
        
        - **Slow loading?** - Use smaller datasets first
        - **Generation fails?** - Check for valid data types
        - **Notebook errors?** - Install missing libraries
        - **Missing features?** - Refresh the page (Ctrl+R)
        - **Need help?** - Check this guide or ask!
        """)
    
    st.markdown("---")
    
    # Glossary
    st.header("📝 Glossary")
    
    with st.expander("Data Science Terms"):
        st.markdown("""
        | Term | Definition |
        |------|------------|
        | **EDA** | Exploratory Data Analysis - Initial data investigation |
        | **Feature** | A column/variable in your dataset |
        | **Target** | The variable you want to predict |
        | **Imputation** | Filling in missing values |
        | **Encoding** | Converting categories to numbers |
        | **Scaling** | Normalizing feature ranges |
        | **Clustering** | Grouping similar data points |
        | **PCA** | Principal Component Analysis - dimension reduction |
        | **t-SNE** | Visualization technique for high-dimensional data |
        | **RMSE** | Root Mean Square Error - regression metric |
        | **F1-Score** | Classification metric balancing precision & recall |
        | **Silhouette** | Clustering quality metric |
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888;">
    📧 Need more help? Check the documentation or reach out!<br>
    <strong>DataDojo</strong> - Your Data Science Training Ground
    </div>
    """, unsafe_allow_html=True)
