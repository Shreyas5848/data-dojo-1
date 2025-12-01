"""
DataDojo Web Dashboard Configuration
Settings and theme configuration for the Streamlit application.
"""

import streamlit as st

# Theme Configuration
DATADOJO_THEME = {
    'primary_color': '#FF9900',        # Claude Orange
    'secondary_color': '#FFB366',      # Light Orange
    'accent_color': '#FF7043',         # Warm Orange
    'background_color': '#0E1117',     # Dark Background
    'surface_color': '#262730',        # Dark Surface
    'text_color': '#FAFAFA',           # Light Text
    'success_color': '#00D084',        # Bright Green
    'warning_color': '#FFB020',        # Bright Amber
    'error_color': '#FF6B6B',          # Bright Red
    'info_color': '#4FC3F7'            # Bright Blue
}

# Dashboard Layout Configuration
LAYOUT_CONFIG = {
    'sidebar_width': 300,
    'main_content_padding': '2rem',
    'card_border_radius': '12px',
    'chart_height': 400,
    'max_datasets_display': 50,
    'max_visualizations': 10,
    'cache_ttl': 3600  # 1 hour
}

# Data Processing Configuration
DATA_CONFIG = {
    'max_file_size_mb': 100,
    'max_rows_preview': 1000,
    'max_columns_display': 20,
    'chunk_size': 10000,
    'supported_formats': ['.csv', '.xlsx', '.json', '.parquet'],
    'encoding_options': ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
}

# Visualization Configuration
VIZ_CONFIG = {
    'color_palettes': {
        'primary': ['#FF9900', '#FFB366', '#FFCC80', '#FFE0B2'],
        'sequential': ['#FFF3E0', '#FFE0B2', '#FFCC80', '#FFB366', '#FF9900', '#FF8F00'],
        'diverging': ['#FF5722', '#FF8A65', '#FFCC80', '#C8E6C9', '#81C784', '#4CAF50'],
        'qualitative': ['#FF9900', '#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    },
    'chart_defaults': {
        'height': 400,
        'opacity': 0.8,
        'margin': {'l': 50, 'r': 50, 't': 80, 'b': 50},
        'font_family': 'Arial, sans-serif',
        'font_size': 12
    }
}

# Quality Score Thresholds
QUALITY_THRESHOLDS = {
    'excellent': 0.9,
    'good': 0.8,
    'fair': 0.6,
    'poor': 0.4
}

# Custom CSS Styles
def get_custom_css():
    return f"""
    <style>
    /* Main container styling */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background-color: {DATADOJO_THEME['background_color']};
    }}
    
    /* Header styling */
    .datadojo-header {{
        background: linear-gradient(135deg, {DATADOJO_THEME['primary_color']}, {DATADOJO_THEME['secondary_color']});
        padding: 2rem;
        border-radius: {LAYOUT_CONFIG['card_border_radius']};
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(255, 153, 0, 0.3);
    }}
    
    /* Metric cards */
    .metric-card {{
        background: {DATADOJO_THEME['surface_color']};
        padding: 1.5rem;
        border-radius: {LAYOUT_CONFIG['card_border_radius']};
        border-left: 4px solid {DATADOJO_THEME['primary_color']};
        box-shadow: 0 2px 8px rgba(255, 153, 0, 0.2);
        margin: 1rem 0;
        color: {DATADOJO_THEME['text_color']};
    }}
    
    /* Quality score indicators */
    .quality-excellent {{ color: {DATADOJO_THEME['success_color']}; }}
    .quality-good {{ color: {DATADOJO_THEME['info_color']}; }}
    .quality-fair {{ color: {DATADOJO_THEME['warning_color']}; }}
    .quality-poor {{ color: {DATADOJO_THEME['error_color']}; }}
    
    /* Button styling */
    .stButton > button {{
        background-color: {DATADOJO_THEME['primary_color']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        background-color: {DATADOJO_THEME['accent_color']};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: {DATADOJO_THEME['surface_color']};
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: {DATADOJO_THEME['surface_color']};
        border-radius: 6px;
        border: 1px solid {DATADOJO_THEME['primary_color']};
        color: {DATADOJO_THEME['text_color']};
    }}
    
    /* DataFrame dark styling */
    .dataframe {{
        background-color: {DATADOJO_THEME['surface_color']};
        color: {DATADOJO_THEME['text_color']};
        border: 1px solid #404040;
    }}
    
    /* DataFrame styling */
    .dataframe {{
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }}
    
    /* Alert styling */
    .alert {{
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }}
    
    .alert-success {{
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }}
    
    .alert-warning {{
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }}
    
    .alert-error {{
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {DATADOJO_THEME['surface_color']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {DATADOJO_THEME['primary_color']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {DATADOJO_THEME['accent_color']};
    }}
    </style>
    """

def apply_theme():
    """Apply DataDojo theme to Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)

def get_quality_class(score: float) -> str:
    """Get CSS class for quality score."""
    if score >= QUALITY_THRESHOLDS['excellent']:
        return 'quality-excellent'
    elif score >= QUALITY_THRESHOLDS['good']:
        return 'quality-good'
    elif score >= QUALITY_THRESHOLDS['fair']:
        return 'quality-fair'
    else:
        return 'quality-poor'

def format_metric_delta(value: float, format_type: str = 'percent') -> str:
    """Format metric delta with appropriate styling."""
    if format_type == 'percent':
        formatted_value = f"{value:.1%}"
    elif format_type == 'number':
        formatted_value = f"{value:,.0f}"
    elif format_type == 'bytes':
        if value >= 1024**3:
            formatted_value = f"{value / (1024**3):.1f} GB"
        elif value >= 1024**2:
            formatted_value = f"{value / (1024**2):.1f} MB"
        elif value >= 1024:
            formatted_value = f"{value / 1024:.1f} KB"
        else:
            formatted_value = f"{value:.0f} B"
    else:
        formatted_value = str(value)
    
    return formatted_value