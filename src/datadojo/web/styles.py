"""
DataDojo Professional Styles - Clean, Minimal, High-Contrast UI
Focus on readability, professional appearance, and consistent design
"""

def get_modern_css():
    """Return professional CSS with excellent contrast and clean aesthetics."""
    return """
<style>
/* ============================================
   ROOT VARIABLES & THEME - PROFESSIONAL PALETTE
   ============================================ */
:root {
    /* Primary palette - Blue-based for professionalism */
    --primary: #3B82F6;
    --primary-light: #60A5FA;
    --primary-dark: #2563EB;
    --primary-glow: rgba(59, 130, 246, 0.3);
    
    /* Secondary - Teal for accents */
    --secondary: #14B8A6;
    --secondary-light: #2DD4BF;
    --secondary-glow: rgba(20, 184, 166, 0.3);
    
    /* Accent - Purple for highlights */
    --accent: #8B5CF6;
    --accent-light: #A78BFA;
    --accent-glow: rgba(139, 92, 246, 0.3);
    
    /* Status colors */
    --success: #22C55E;
    --warning: #F59E0B;
    --error: #EF4444;
    --info: #06B6D4;
    
    /* Backgrounds - Dark with good contrast */
    --bg-dark: #0F172A;
    --bg-card: #1E293B;
    --bg-card-hover: #334155;
    --bg-elevated: #283548;
    --border-color: #334155;
    --border-light: #475569;
    
    /* Text - High contrast */
    --text-primary: #F8FAFC;
    --text-secondary: #CBD5E1;
    --text-muted: #94A3B8;
    --text-inverse: #0F172A;
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
    --gradient-secondary: linear-gradient(135deg, #14B8A6 0%, #06B6D4 100%);
    --gradient-accent: linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%);
    --gradient-header: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    
    /* Shadows */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 20px var(--primary-glow);
    
    /* Transitions */
    --transition-fast: all 0.15s ease;
    --transition-smooth: all 0.3s ease;
}

/* ============================================
   CLEAN BACKGROUND
   ============================================ */
.stApp {
    background: var(--bg-dark) !important;
    min-height: 100vh;
}

/* Subtle gradient overlay */
.stApp > div {
    position: relative;
}

/* ============================================
   PROFESSIONAL CARDS
   ============================================ */
.glass-card, .card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    transition: var(--transition-smooth);
}

.card:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* ============================================
   HERO HEADER - CLEAN & PROFESSIONAL
   ============================================ */
.hero-header {
    text-align: center;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.75rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.75rem;
    letter-spacing: -0.5px;
}

.hero-title .icon {
    display: inline-block;
    width: 48px;
    height: 48px;
    background: var(--gradient-primary);
    border-radius: 12px;
    margin-right: 12px;
    vertical-align: middle;
    line-height: 48px;
    text-align: center;
    font-size: 1.5rem;
}

@keyframes titleGlow {
    0%, 100% { filter: brightness(1); }
    50% { filter: brightness(1.2); }
}

.hero-subtitle {
    font-size: 1.125rem !important;
    color: var(--text-secondary) !important;
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto !important;
    line-height: 1.6;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
}

/* ============================================
   METRIC CARDS - CLEAN & PROFESSIONAL
   ============================================ */
.metric-card-modern {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    transition: var(--transition-smooth);
}

.metric-card-modern:hover {
    background: var(--bg-card-hover);
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.metric-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: var(--primary);
}

.metric-value {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0.25rem 0;
}

.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ============================================
   NAVIGATION & SIDEBAR
   ============================================ */
.nav-pills {
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem;
    background: var(--bg-card);
    border-radius: 8px;
    border: 1px solid var(--border-color);
}

.nav-pill {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-fast);
    color: var(--text-secondary);
    border: none;
    background: transparent;
}

.nav-pill:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}

.nav-pill.active {
    background: var(--primary);
    color: white;
}

/* ============================================
   BUTTONS - PROFESSIONAL
   ============================================ */
.stButton > button {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.625rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    color: white !important;
    cursor: pointer;
    transition: var(--transition-fast) !important;
}

.stButton > button:hover {
    background: var(--primary-dark) !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary button style */
.btn-secondary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

.btn-secondary:hover {
    border-color: var(--primary) !important;
}

/* ============================================
   FEATURE CARDS - CLEAN DESIGN
   ============================================ */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.5rem;
    transition: var(--transition-smooth);
    height: 100%;
}

.feature-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--primary);
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

.feature-icon {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    margin-bottom: 1rem;
    background: var(--bg-elevated);
    color: var(--primary);
    border: 1px solid var(--border-color);
}

.feature-title {
    font-size: 1.125rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem;
}

.feature-desc {
    color: var(--text-secondary);
    line-height: 1.5;
    font-size: 0.875rem;
}

/* ============================================
   SIDEBAR STYLING
   ============================================ */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* Sidebar logo area */
.sidebar-logo {
    text-align: center;
    padding: 1.5rem 1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border-color);
}

.sidebar-logo h2 {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8rem;
    font-weight: 800;
}

/* ============================================
   SELECT BOXES & INPUTS
   ============================================ */
.stSelectbox > div > div {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: var(--transition-smooth);
}

.stSelectbox > div > div:hover {
    border-color: var(--primary) !important;
    box-shadow: 0 0 15px var(--primary-glow);
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
    transition: var(--transition-smooth);
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 15px var(--primary-glow) !important;
    outline: none !important;
}

/* ============================================
   DATA FRAMES & TABLES
   ============================================ */
.stDataFrame {
    background: var(--bg-glass) !important;
    backdrop-filter: var(--blur-glass);
    border-radius: 16px !important;
    border: 1px solid var(--border-glass) !important;
    overflow: hidden;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    background: transparent !important;
}

/* ============================================
   EXPANDERS
   ============================================ */
.streamlit-expanderHeader {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: var(--transition-smooth);
}

.streamlit-expanderHeader:hover {
    border-color: var(--primary) !important;
    background: rgba(255, 107, 53, 0.1) !important;
}

.streamlit-expanderContent {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

/* ============================================
   PROGRESS BARS
   ============================================ */
.stProgress > div > div {
    background: var(--bg-glass) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

.stProgress > div > div > div {
    background: var(--gradient-primary) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 15px var(--primary-glow);
}

/* ============================================
   ALERTS & MESSAGES
   ============================================ */
.stSuccess {
    background: rgba(16, 185, 129, 0.1) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 12px !important;
    color: #10B981 !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.1) !important;
    border: 1px solid rgba(245, 158, 11, 0.3) !important;
    border-radius: 12px !important;
    color: #F59E0B !important;
}

.stError {
    background: rgba(239, 68, 68, 0.1) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 12px !important;
    color: #EF4444 !important;
}

.stInfo {
    background: rgba(0, 217, 255, 0.1) !important;
    border: 1px solid rgba(0, 217, 255, 0.3) !important;
    border-radius: 12px !important;
    color: #00D9FF !important;
}

/* ============================================
   TABS
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-glass);
    border-radius: 16px;
    padding: 0.5rem;
    gap: 0.5rem;
    border: 1px solid var(--border-glass);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 12px !important;
    color: var(--text-secondary) !important;
    padding: 0.75rem 1.5rem !important;
    transition: var(--transition-smooth);
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 107, 53, 0.1) !important;
    color: var(--primary) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--primary-glow);
}

/* ============================================
   METRICS
   ============================================ */
[data-testid="stMetricValue"] {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ============================================
   SPINNERS & LOADING
   ============================================ */
.stSpinner > div {
    border-color: var(--primary) !important;
    border-right-color: transparent !important;
}

/* ============================================
   CUSTOM SCROLLBAR
   ============================================ */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-dark);
}

::-webkit-scrollbar-thumb {
    background: var(--gradient-primary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* ============================================
   HIDE DEFAULT STREAMLIT ELEMENTS
   ============================================ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============================================
   TEXT COLORS
   ============================================ */
.stMarkdown, .stText, p, span, li {
    color: var(--text-primary) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

/* ============================================
   PLOTLY CHART STYLING
   ============================================ */
.js-plotly-plot .plotly .modebar {
    background: var(--bg-glass) !important;
    border-radius: 8px;
}

/* ============================================
   ANIMATIONS
   ============================================ */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

.animate-fade-in {
    animation: fadeInUp 0.6s ease-out forwards;
}

.animate-pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* ============================================
   RESPONSIVE DESIGN
   ============================================ */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem !important;
    }
    
    .hero-subtitle {
        font-size: 1rem !important;
    }
    
    .metric-value {
        font-size: 1.8rem !important;
    }
    
    .glass-card {
        padding: 1rem;
    }
}

/* ============================================
   BADGE STYLES
   ============================================ */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-primary {
    background: var(--gradient-primary);
    color: white;
}

.badge-secondary {
    background: var(--gradient-secondary);
    color: white;
}

.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* ============================================
   TOOLTIP STYLES
   ============================================ */
.tooltip {
    position: relative;
}

.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-card);
    color: var(--text-primary);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: var(--transition-smooth);
    border: 1px solid var(--border-glass);
}

.tooltip:hover::after {
    opacity: 1;
    visibility: visible;
}

/* ============================================
   DIVIDER
   ============================================ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-glass), transparent);
    margin: 2rem 0;
}

/* ============================================
   STAT GRID
   ============================================ */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

/* ============================================
   FLOATING ACTION BUTTON
   ============================================ */
.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: var(--gradient-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 20px var(--primary-glow);
    cursor: pointer;
    transition: var(--transition-smooth);
    z-index: 1000;
}

.fab:hover {
    transform: scale(1.1);
    box-shadow: 0 8px 30px var(--primary-glow);
}

</style>
"""


def get_animated_background():
    """Return subtle animated background gradient - CSS only, no HTML elements."""
    return """
<style>
/* Subtle animated gradient background applied to main container */
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%) !important;
}

/* Subtle glow orbs using pseudo-elements on existing containers */
.stApp::before {
    content: '';
    position: fixed;
    top: 10%;
    left: 5%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(255, 107, 53, 0.08), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: pulse-glow 8s ease-in-out infinite;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: 20%;
    right: 10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0, 217, 255, 0.06), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: pulse-glow 10s ease-in-out infinite reverse;
}

@keyframes pulse-glow {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}
</style>
"""


def create_hero_header(title: str, subtitle: str = ""):
    """Create an animated hero header."""
    return f"""
<div class="hero-header">
    <h1 class="hero-title">{title}</h1>
    <p class="hero-subtitle">{subtitle}</p>
</div>
"""


def create_metric_card(icon: str, value: str, label: str, color: str = "primary"):
    """Create a modern metric card with glow effect."""
    return f"""
<div class="metric-card-modern">
    <div class="metric-icon">{icon}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>
"""


def create_feature_card(icon: str, title: str, description: str):
    """Create a feature card with hover effects."""
    return f"""
<div class="feature-card">
    <div class="feature-icon">{icon}</div>
    <h3 class="feature-title">{title}</h3>
    <p class="feature-desc">{description}</p>
</div>
"""


def create_glass_card(content: str):
    """Wrap content in a glassmorphism card."""
    return f"""
<div class="glass-card">
    {content}
</div>
"""


def create_badge(text: str, variant: str = "primary"):
    """Create a styled badge."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def create_divider():
    """Create a styled divider."""
    return '<div class="divider"></div>'


def create_stat_card(icon: str, value: str, label: str, trend: str = None):
    """Create a stat card with optional trend indicator."""
    trend_html = ""
    if trend:
        trend_color = "#10B981" if trend.startswith("+") else "#EF4444"
        trend_html = f'<span style="color: {trend_color}; font-size: 0.8rem;">{trend}</span>'
    
    return f"""
<div class="metric-card-modern">
    <div class="metric-icon">{icon}</div>
    <div class="metric-value">{value}</div>
    {trend_html}
    <div class="metric-label">{label}</div>
</div>
"""
