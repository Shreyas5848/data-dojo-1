"""
DataDojo Professional Styles - Modern Data Science Learning Platform
Crafted with care for students and data scientists who appreciate beautiful tools
"""

def get_modern_css():
    """Return sophisticated CSS with a unique data-science aesthetic."""
    return """
<style>
/* ============================================
   FONT IMPORTS - Professional Typography
   ============================================ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ============================================
   ROOT VARIABLES - SIGNATURE DATADOJO PALETTE
   Warm coral meets deep space - unique, memorable, professional
   ============================================ */
:root {
    /* Brand Colors - Coral/Ember Primary */
    --primary: #FF6B6B;
    --primary-light: #FF8E8E;
    --primary-dark: #E85555;
    --primary-glow: rgba(255, 107, 107, 0.25);
    
    /* Electric Mint - Secondary for data viz */
    --secondary: #4ECDC4;
    --secondary-light: #72DDD6;
    --secondary-glow: rgba(78, 205, 196, 0.25);
    
    /* Golden Amber - Highlights and CTAs */
    --accent: #FFD93D;
    --accent-light: #FFE566;
    --accent-glow: rgba(255, 217, 61, 0.25);
    
    /* Lavender - Tertiary accent */
    --tertiary: #C9B1FF;
    --tertiary-glow: rgba(201, 177, 255, 0.25);
    
    /* Status colors - Refined */
    --success: #6BCB77;
    --success-bg: rgba(107, 203, 119, 0.12);
    --warning: #FFB84C;
    --warning-bg: rgba(255, 184, 76, 0.12);
    --error: #FF6B6B;
    --error-bg: rgba(255, 107, 107, 0.12);
    --info: #4ECDC4;
    --info-bg: rgba(78, 205, 196, 0.12);
    
    /* Deep Space Backgrounds */
    --bg-dark: #0D1117;
    --bg-darker: #010409;
    --bg-card: #161B22;
    --bg-card-hover: #1C2128;
    --bg-elevated: #21262D;
    --bg-surface: #0D1117;
    
    /* Border system */
    --border-color: #30363D;
    --border-subtle: #21262D;
    --border-muted: #484F58;
    --border-emphasis: #606771;
    
    /* Text - Carefully calibrated */
    --text-primary: #F0F6FC;
    --text-secondary: #8B949E;
    --text-muted: #6E7681;
    --text-link: #58A6FF;
    --text-inverse: #0D1117;
    
    /* Signature Gradients */
    --gradient-primary: linear-gradient(135deg, #FF6B6B 0%, #FFD93D 100%);
    --gradient-secondary: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
    --gradient-accent: linear-gradient(135deg, #C9B1FF 0%, #FF6B6B 100%);
    --gradient-warm: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 50%, #FFD93D 100%);
    --gradient-cool: linear-gradient(135deg, #4ECDC4 0%, #58A6FF 100%);
    --gradient-mesh: radial-gradient(at 40% 20%, rgba(255, 107, 107, 0.15) 0px, transparent 50%),
                     radial-gradient(at 80% 0%, rgba(78, 205, 196, 0.1) 0px, transparent 50%),
                     radial-gradient(at 0% 50%, rgba(201, 177, 255, 0.1) 0px, transparent 50%),
                     radial-gradient(at 80% 80%, rgba(255, 217, 61, 0.08) 0px, transparent 50%);
    
    /* Glass effects */
    --bg-glass: rgba(22, 27, 34, 0.7);
    --border-glass: rgba(48, 54, 61, 0.6);
    --blur-glass: blur(20px);
    
    /* Shadows - Layered depth */
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.4);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
    --shadow-lg: 0 12px 48px rgba(0, 0, 0, 0.6);
    --shadow-glow-primary: 0 0 40px var(--primary-glow);
    --shadow-glow-secondary: 0 0 40px var(--secondary-glow);
    --shadow-inner: inset 0 1px 0 rgba(255, 255, 255, 0.03);
    
    /* Transitions - Smooth and intentional */
    --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-bounce: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    
    /* Typography scale */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-display: 'Space Grotesk', 'Inter', sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    
    /* Spacing rhythm */
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --space-2xl: 3rem;
    
    /* Border radius - Consistent curves */
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --radius-full: 9999px;
}

/* ============================================
   BASE STYLES & TYPOGRAPHY
   ============================================ */
* {
    font-family: var(--font-sans);
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    font-weight: 600;
    letter-spacing: -0.02em;
}

code, pre, .stCodeBlock {
    font-family: var(--font-mono) !important;
}

/* ============================================
   MAIN APP BACKGROUND - Mesh Gradient
   ============================================ */
.stApp {
    background: var(--bg-dark) !important;
    background-image: var(--gradient-mesh) !important;
    background-attachment: fixed;
    min-height: 100vh;
}

/* ============================================
   PROFESSIONAL CARDS - Elevated Design
   ============================================ */
.glass-card {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    box-shadow: var(--shadow-md);
    transition: var(--transition-smooth);
}

.glass-card:hover {
    border-color: var(--border-muted);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* Card with accent border */
.card-accent {
    border-left: 3px solid var(--primary);
}

/* ============================================
   HERO HEADER - Signature DataDojo Style
   ============================================ */
.hero-header {
    text-align: center;
    padding: var(--space-2xl) var(--space-xl);
    margin-bottom: var(--space-xl);
    position: relative;
    z-index: 1;
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: 2.75rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin-bottom: 1rem !important;
    letter-spacing: -0.02em;
    display: block;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 0.5rem;
    display: inline-block;
}

.hero-subtitle {
    font-family: var(--font-sans) !important;
    font-size: 1.1rem !important;
    color: var(--text-secondary) !important;
    font-weight: 400;
    max-width: 640px;
    margin: 0.5rem auto 0 auto !important;
    line-height: 1.6;
}

/* ============================================
   METRIC CARDS - Data-Focused Design
   ============================================ */
.metric-card-modern {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    text-align: center;
    transition: var(--transition-smooth);
    border-top: 3px solid transparent;
}

.metric-card-modern:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-muted);
    border-top-color: var(--primary);
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}

.metric-icon {
    font-size: 1.75rem;
    margin-bottom: 0.5rem;
    display: inline-block;
    color: var(--primary);
}

.metric-value {
    font-family: var(--font-display) !important;
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0.5rem 0;
    letter-spacing: -0.02em;
}

.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ============================================
   NAVIGATION & SIDEBAR - Refined
   ============================================ */
.nav-pills {
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem;
    background: var(--bg-card);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-color);
}

.nav-pill {
    padding: 0.625rem 1.25rem;
    border-radius: var(--radius-sm);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-fast);
    color: var(--text-secondary);
    border: none;
    background: transparent;
    font-size: 0.9rem;
}

.nav-pill:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}

.nav-pill.active {
    background: var(--gradient-primary);
    color: white;
    box-shadow: 0 2px 8px var(--primary-glow);
}

/* ============================================
   BUTTONS - Polished Interactive Elements
   ============================================ */
.stButton > button {
    background: var(--gradient-primary) !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.75rem 1.75rem !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: white !important;
    cursor: pointer;
    transition: var(--transition-smooth) !important;
    box-shadow: 0 2px 8px var(--primary-glow);
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px var(--primary-glow) !important;
    filter: brightness(1.1);
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px var(--primary-glow) !important;
}

/* Secondary button style */
.btn-secondary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

.btn-secondary:hover {
    border-color: var(--primary) !important;
    background: var(--bg-card-hover) !important;
}

/* Ghost button */
.btn-ghost {
    background: transparent !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

.btn-ghost:hover {
    background: var(--bg-card) !important;
    border-color: var(--primary) !important;
}

/* ============================================
   FEATURE CARDS - Interactive Learning Cards
   ============================================ */
.feature-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
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
    font-size: 1.5rem;
    margin-bottom: 1rem;
    color: var(--primary);
}

.feature-title {
    font-family: var(--font-display) !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.5rem;
}

.feature-desc {
    color: var(--text-secondary);
    line-height: 1.6;
    font-size: 0.9rem;
}

/* ============================================
   SIDEBAR STYLING - Clean Navigation
   ============================================ */
section[data-testid="stSidebar"] {
    background: var(--bg-darker) !important;
    border-right: 1px solid var(--border-subtle);
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
    padding: 1.75rem 1rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border-subtle);
    position: relative;
}

.sidebar-logo::after {
    content: '';
    position: absolute;
    bottom: -1px;
    left: 20%;
    width: 60%;
    height: 1px;
    background: var(--gradient-primary);
}

.sidebar-logo h2 {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Navigation items */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
    transition: var(--transition-fast);
}

/* ============================================
   SELECT BOXES & INPUTS - Refined
   ============================================ */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    transition: var(--transition-smooth);
    font-family: var(--font-sans) !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--border-muted) !important;
}

.stSelectbox > div > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
    transition: var(--transition-smooth);
    font-family: var(--font-sans) !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px var(--primary-glow) !important;
    outline: none !important;
}

/* Placeholder styling */
.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
}

/* ============================================
   DATA FRAMES & TABLES - Data Science Look
   ============================================ */
.stDataFrame {
    background: var(--bg-card) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--border-color) !important;
    overflow: hidden;
    box-shadow: var(--shadow-md);
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    background: transparent !important;
}

/* Table header styling */
.stDataFrame th {
    background: var(--bg-elevated) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ============================================
   EXPANDERS - Collapsible Sections
   ============================================ */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    transition: var(--transition-smooth);
    font-family: var(--font-sans) !important;
}

.streamlit-expanderHeader:hover {
    border-color: var(--border-muted) !important;
    background: var(--bg-card-hover) !important;
}

.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* New Streamlit expander selectors */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

[data-testid="stExpander"] > div:first-child {
    background: var(--bg-card) !important;
}

[data-testid="stExpander"] summary {
    background: var(--bg-card) !important;
    padding: 0.75rem 1rem !important;
    font-family: var(--font-sans) !important;
    color: var(--text-primary) !important;
}

[data-testid="stExpander"] summary:hover {
    background: var(--bg-card-hover) !important;
}

/* Hide any "Key" or internal labels that might overlap */
[data-testid="stExpander"] summary span[data-testid] {
    display: none !important;
}

/* Ensure expander text is visible and not overlapped */
[data-testid="stExpander"] summary > span:first-child,
[data-testid="stExpander"] summary p {
    color: var(--text-primary) !important;
    position: relative !important;
    z-index: 1 !important;
}

/* Fix expander icon positioning */
[data-testid="stExpander"] summary svg {
    color: var(--text-secondary) !important;
}

/* Expander content area */
[data-testid="stExpander"] > div:last-child {
    background: var(--bg-card) !important;
    padding: 1rem !important;
}

/* ============================================
   PROGRESS BARS - Visual Feedback
   ============================================ */
.stProgress > div > div {
    background: var(--bg-elevated) !important;
    border-radius: var(--radius-full) !important;
    overflow: hidden;
    height: 8px !important;
}

.stProgress > div > div > div {
    background: var(--gradient-primary) !important;
    border-radius: var(--radius-full) !important;
    box-shadow: 0 0 12px var(--primary-glow);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ============================================
   ALERTS & MESSAGES - Refined Feedback
   ============================================ */
.stSuccess {
    background: var(--success-bg) !important;
    border: 1px solid rgba(107, 203, 119, 0.3) !important;
    border-radius: var(--radius-md) !important;
    color: var(--success) !important;
    border-left: 3px solid var(--success) !important;
}

.stWarning {
    background: var(--warning-bg) !important;
    border: 1px solid rgba(255, 184, 76, 0.3) !important;
    border-radius: var(--radius-md) !important;
    color: var(--warning) !important;
    border-left: 3px solid var(--warning) !important;
}

.stError {
    background: var(--error-bg) !important;
    border: 1px solid rgba(255, 107, 107, 0.3) !important;
    border-radius: var(--radius-md) !important;
    color: var(--error) !important;
    border-left: 3px solid var(--error) !important;
}

.stInfo {
    background: var(--info-bg) !important;
    border: 1px solid rgba(78, 205, 196, 0.3) !important;
    border-radius: var(--radius-md) !important;
    color: var(--info) !important;
    border-left: 3px solid var(--info) !important;
}

/* ============================================
   TABS - Segment Control Style
   ============================================ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: 0.375rem;
    gap: 0.375rem;
    border: 1px solid var(--border-color);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    padding: 0.625rem 1.25rem !important;
    transition: var(--transition-smooth);
    font-family: var(--font-sans) !important;
    font-weight: 500;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

.stTabs [aria-selected="true"] {
    background: var(--gradient-primary) !important;
    color: white !important;
    box-shadow: 0 2px 8px var(--primary-glow);
}

/* ============================================
   METRICS - Dashboard Style
   ============================================ */
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
}

/* ============================================
   SPINNERS & LOADING
   ============================================ */
.stSpinner > div {
    border-color: var(--primary) !important;
    border-right-color: transparent !important;
}

/* Loading overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(13, 17, 23, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

/* ============================================
   CUSTOM SCROLLBAR - Refined
   ============================================ */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--bg-darker);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb {
    background: var(--bg-elevated);
    border-radius: 5px;
    border: 2px solid var(--bg-darker);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--border-muted);
}

/* Firefox scrollbar */
* {
    scrollbar-width: thin;
    scrollbar-color: var(--bg-elevated) var(--bg-darker);
}

/* ============================================
   HIDE DEFAULT STREAMLIT ELEMENTS
   ============================================ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ============================================
   TYPOGRAPHY COLORS
   ============================================ */
.stMarkdown, .stText, p, span, li {
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}

/* Code blocks */
code {
    font-family: var(--font-mono) !important;
    background: var(--bg-elevated) !important;
    padding: 0.15rem 0.4rem;
    border-radius: var(--radius-sm);
    font-size: 0.875em;
    color: var(--primary-light) !important;
}

pre {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem !important;
}

/* ============================================
   PLOTLY CHART STYLING
   ============================================ */
.js-plotly-plot .plotly .modebar {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
}

.js-plotly-plot .plotly .modebar-btn {
    color: var(--text-secondary) !important;
}

.js-plotly-plot .plotly .modebar-btn:hover {
    color: var(--primary) !important;
}

/* ============================================
   ANIMATIONS - Refined Motion
   ============================================ */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

@keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.animate-fade-in {
    animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.animate-slide-in {
    animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1) forwards;
}

.animate-pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* Stagger children animations */
.stagger-1 { animation-delay: 0.1s; }
.stagger-2 { animation-delay: 0.2s; }
.stagger-3 { animation-delay: 0.3s; }
.stagger-4 { animation-delay: 0.4s; }

/* ============================================
   RESPONSIVE DESIGN
   ============================================ */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.25rem !important;
    }
    
    .hero-subtitle {
        font-size: 1rem !important;
        padding: 0 1rem;
    }
    
    .metric-value {
        font-size: 1.75rem !important;
    }
    
    .glass-card {
        padding: 1.25rem;
    }
    
    .feature-card {
        padding: 1.25rem;
    }
    
    .metric-icon {
        width: 44px;
        height: 44px;
        font-size: 1.25rem;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 1.875rem !important;
    }
    
    .metric-card-modern {
        padding: 1rem;
    }
}

/* ============================================
   BADGE STYLES - Refined Labels
   ============================================ */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.625rem;
    border-radius: var(--radius-full);
    font-family: var(--font-sans);
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.badge-primary {
    background: var(--gradient-primary);
    color: white;
}

.badge-secondary {
    background: var(--gradient-secondary);
    color: var(--text-inverse);
}

.badge-success {
    background: var(--success-bg);
    color: var(--success);
    border: 1px solid rgba(107, 203, 119, 0.3);
}

.badge-warning {
    background: var(--warning-bg);
    color: var(--warning);
    border: 1px solid rgba(255, 184, 76, 0.3);
}

.badge-info {
    background: var(--info-bg);
    color: var(--info);
    border: 1px solid rgba(78, 205, 196, 0.3);
}

.badge-ghost {
    background: var(--bg-elevated);
    color: var(--text-secondary);
    border: 1px solid var(--border-color);
}

/* ============================================
   TOOLTIP STYLES
   ============================================ */
.tooltip {
    position: relative;
    cursor: help;
}

.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: calc(100% + 8px);
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-elevated);
    color: var(--text-primary);
    padding: 0.5rem 0.875rem;
    border-radius: var(--radius-sm);
    font-family: var(--font-sans);
    font-size: 0.8rem;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: var(--transition-fast);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-md);
    z-index: 100;
}

.tooltip::before {
    content: '';
    position: absolute;
    bottom: calc(100% + 2px);
    left: 50%;
    transform: translateX(-50%);
    border: 6px solid transparent;
    border-top-color: var(--border-color);
    opacity: 0;
    visibility: hidden;
    transition: var(--transition-fast);
}

.tooltip:hover::after,
.tooltip:hover::before {
    opacity: 1;
    visibility: visible;
}

/* ============================================
   DIVIDER - Section Separator
   ============================================ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-color), transparent);
    margin: 2.5rem 0;
}

.divider-vertical {
    width: 1px;
    height: 100%;
    background: linear-gradient(180deg, transparent, var(--border-color), transparent);
}

/* ============================================
   STAT GRID - Dashboard Layout
   ============================================ */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.25rem;
    margin: 1.5rem 0;
}

/* ============================================
   FLOATING ACTION BUTTON
   ============================================ */
.fab {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    background: var(--gradient-primary);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px var(--primary-glow);
    cursor: pointer;
    transition: var(--transition-bounce);
    z-index: 1000;
    border: none;
}

.fab:hover {
    transform: scale(1.08) translateY(-2px);
    box-shadow: 0 8px 24px var(--primary-glow);
}

.fab:active {
    transform: scale(0.95);
}

/* ============================================
   ADDITIONAL UTILITY CLASSES
   ============================================ */

/* Text utilities */
.text-gradient {
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.text-muted {
    color: var(--text-muted) !important;
}

.text-secondary {
    color: var(--text-secondary) !important;
}

.text-mono {
    font-family: var(--font-mono) !important;
}

/* Spacing utilities */
.mt-1 { margin-top: var(--space-sm); }
.mt-2 { margin-top: var(--space-md); }
.mt-3 { margin-top: var(--space-lg); }
.mb-1 { margin-bottom: var(--space-sm); }
.mb-2 { margin-bottom: var(--space-md); }
.mb-3 { margin-bottom: var(--space-lg); }

/* Flex utilities */
.flex { display: flex; }
.flex-center { display: flex; align-items: center; justify-content: center; }
.flex-between { display: flex; align-items: center; justify-content: space-between; }
.gap-1 { gap: var(--space-sm); }
.gap-2 { gap: var(--space-md); }

/* Data visualization accent colors for consistency */
.data-viz-1 { color: var(--primary); }
.data-viz-2 { color: var(--secondary); }
.data-viz-3 { color: var(--accent); }
.data-viz-4 { color: var(--tertiary); }

/* Interactive hover states */
.interactive {
    cursor: pointer;
    transition: var(--transition-fast);
}

.interactive:hover {
    opacity: 0.8;
}

/* Focus states for accessibility */
*:focus-visible {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}

/* Selection styling */
::selection {
    background: var(--primary);
    color: white;
}

/* ============================================
   SKELETON LOADERS - Modern Loading States
   ============================================ */
.skeleton {
    background: linear-gradient(90deg, var(--bg-card) 25%, var(--bg-elevated) 50%, var(--bg-card) 75%);
    background-size: 200% 100%;
    animation: skeleton-loading 1.5s ease-in-out infinite;
    border-radius: var(--radius-md);
}

.skeleton-text {
    height: 1rem;
    margin-bottom: 0.5rem;
    width: 100%;
}

.skeleton-text.short {
    width: 60%;
}

.skeleton-title {
    height: 1.5rem;
    width: 40%;
    margin-bottom: 1rem;
}

.skeleton-card {
    height: 120px;
    width: 100%;
}

.skeleton-avatar {
    width: 48px;
    height: 48px;
    border-radius: 50%;
}

@keyframes skeleton-loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* ============================================
   ENHANCED SIDEBAR NAVIGATION
   ============================================ */
.nav-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1rem;
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    text-decoration: none;
    transition: var(--transition-smooth);
    margin-bottom: 0.25rem;
    cursor: pointer;
    border: 1px solid transparent;
}

.nav-item:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border-color: var(--border-color);
}

.nav-item.active {
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 217, 61, 0.1) 100%);
    color: var(--primary);
    border-color: var(--primary);
    font-weight: 500;
}

.nav-item .nav-icon {
    font-size: 1.1rem;
    width: 24px;
    text-align: center;
}

.nav-item .nav-label {
    font-size: 0.9rem;
}

.nav-section-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    padding: 1rem 1rem 0.5rem;
    font-weight: 600;
}

/* Sidebar Quick Stats */
.sidebar-stat {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.625rem 0.875rem;
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
    margin-bottom: 0.5rem;
    border: 1px solid var(--border-subtle);
}

.sidebar-stat .stat-value {
    font-family: var(--font-mono);
    font-weight: 600;
    font-size: 0.95rem;
}

.sidebar-stat .stat-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

.sidebar-stat.primary .stat-value { color: var(--primary); }
.sidebar-stat.secondary .stat-value { color: var(--secondary); }
.sidebar-stat.accent .stat-value { color: var(--accent); }

/* ============================================
   ONBOARDING TOUR COMPONENTS
   ============================================ */
.tour-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 9998;
}

.tour-spotlight {
    position: absolute;
    box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.75);
    border-radius: var(--radius-md);
    z-index: 9999;
}

.tour-tooltip {
    position: absolute;
    background: var(--bg-card);
    border: 1px solid var(--primary);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    max-width: 320px;
    z-index: 10000;
    box-shadow: var(--shadow-lg), 0 0 30px var(--primary-glow);
}

.tour-tooltip h4 {
    color: var(--text-primary);
    margin: 0 0 0.5rem;
    font-size: 1.1rem;
}

.tour-tooltip p {
    color: var(--text-secondary);
    margin: 0 0 1rem;
    font-size: 0.9rem;
    line-height: 1.5;
}

.tour-progress {
    display: flex;
    gap: 0.375rem;
    margin-bottom: 1rem;
}

.tour-progress-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--border-color);
    transition: var(--transition-fast);
}

.tour-progress-dot.active {
    background: var(--primary);
    box-shadow: 0 0 8px var(--primary-glow);
}

.tour-progress-dot.completed {
    background: var(--success);
}

.tour-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.tour-btn {
    padding: 0.5rem 1rem;
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-fast);
}

.tour-btn-skip {
    background: transparent;
    border: none;
    color: var(--text-muted);
}

.tour-btn-skip:hover {
    color: var(--text-primary);
}

.tour-btn-next {
    background: var(--gradient-primary);
    border: none;
    color: white;
}

.tour-btn-next:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px var(--primary-glow);
}

/* ============================================
   TOAST NOTIFICATIONS
   ============================================ */
.toast-container {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 10001;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.toast {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1.25rem;
    background: var(--bg-card);
    border-radius: var(--radius-md);
    border-left: 4px solid var(--primary);
    box-shadow: var(--shadow-lg);
    animation: toast-slide-in 0.3s ease-out;
    max-width: 360px;
}

.toast.success { border-left-color: var(--success); }
.toast.warning { border-left-color: var(--warning); }
.toast.error { border-left-color: var(--error); }
.toast.info { border-left-color: var(--info); }

.toast-icon {
    font-size: 1.25rem;
}

.toast-content {
    flex: 1;
}

.toast-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.9rem;
    margin-bottom: 0.125rem;
}

.toast-message {
    color: var(--text-secondary);
    font-size: 0.8rem;
}

.toast-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0.25rem;
    font-size: 1rem;
    transition: var(--transition-fast);
}

.toast-close:hover {
    color: var(--text-primary);
}

@keyframes toast-slide-in {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* ============================================
   ENHANCED EMPTY STATES
   ============================================ */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--bg-card);
    border: 1px dashed var(--border-color);
    border-radius: var(--radius-lg);
}

.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}

.empty-state-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.empty-state-desc {
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.5;
}

.empty-state-action {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1.5rem;
    background: var(--gradient-primary);
    color: white;
    border: none;
    border-radius: var(--radius-md);
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition-smooth);
}

.empty-state-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px var(--primary-glow);
}

/* ============================================
   MICRO-INTERACTIONS
   ============================================ */
/* Ripple effect on click */
.ripple {
    position: relative;
    overflow: hidden;
}

.ripple::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    top: 0;
    left: 0;
    pointer-events: none;
    background-image: radial-gradient(circle, var(--primary) 10%, transparent 10.01%);
    background-repeat: no-repeat;
    background-position: 50%;
    transform: scale(10, 10);
    opacity: 0;
    transition: transform 0.5s, opacity 0.5s;
}

.ripple:active::after {
    transform: scale(0, 0);
    opacity: 0.3;
    transition: 0s;
}

/* Hover lift effect */
.hover-lift {
    transition: var(--transition-smooth);
}

.hover-lift:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
}

/* Pulse animation for notifications */
.pulse-dot {
    position: relative;
}

.pulse-dot::after {
    content: '';
    position: absolute;
    top: -2px;
    right: -2px;
    width: 10px;
    height: 10px;
    background: var(--primary);
    border-radius: 50%;
    animation: pulse-ring 1.5s ease-out infinite;
}

@keyframes pulse-ring {
    0% {
        transform: scale(1);
        opacity: 1;
    }
    100% {
        transform: scale(2);
        opacity: 0;
    }
}

/* Shake animation for errors */
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}

.shake {
    animation: shake 0.5s ease-in-out;
}

/* Success checkmark animation */
@keyframes checkmark {
    0% {
        stroke-dashoffset: 100;
    }
    100% {
        stroke-dashoffset: 0;
    }
}

.success-checkmark {
    stroke-dasharray: 100;
    stroke-dashoffset: 100;
    animation: checkmark 0.5s ease-out forwards;
}

/* ============================================
   MINI CHARTS / SPARKLINES
   ============================================ */
.sparkline-container {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 32px;
}

.sparkline-bar {
    width: 4px;
    background: var(--primary);
    border-radius: 2px 2px 0 0;
    transition: var(--transition-fast);
    opacity: 0.7;
}

.sparkline-bar:hover {
    opacity: 1;
    transform: scaleY(1.1);
}

/* ============================================
   ENHANCED DATA CARDS
   ============================================ */
.data-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 1.25rem;
    transition: var(--transition-smooth);
}

.data-card:hover {
    border-color: var(--primary);
    box-shadow: 0 0 20px var(--primary-glow);
}

.data-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1rem;
}

.data-card-title {
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.data-card-badge {
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    background: var(--bg-elevated);
    border-radius: var(--radius-full);
    color: var(--text-secondary);
}

.data-card-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.data-card-stat {
    text-align: center;
    padding: 0.5rem;
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
}

.data-card-stat-value {
    font-family: var(--font-mono);
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1rem;
}

.data-card-stat-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.data-card-actions {
    display: flex;
    gap: 0.5rem;
}

.data-card-btn {
    flex: 1;
    padding: 0.5rem 0.75rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-size: 0.8rem;
    cursor: pointer;
    transition: var(--transition-fast);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.375rem;
}

.data-card-btn:hover {
    background: var(--bg-card-hover);
    color: var(--text-primary);
    border-color: var(--primary);
}

.data-card-btn.primary {
    background: var(--gradient-primary);
    border-color: transparent;
    color: white;
}

.data-card-btn.primary:hover {
    box-shadow: 0 4px 12px var(--primary-glow);
}

/* ============================================
   KEYBOARD SHORTCUTS INDICATOR
   ============================================ */
.kbd {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.2rem 0.5rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    box-shadow: 0 2px 0 var(--border-color);
}

/* ============================================
   QUICK ACTION PANEL
   ============================================ */
.quick-actions {
    display: flex;
    gap: 0.75rem;
    padding: 1rem;
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.quick-action-btn {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition-smooth);
}

.quick-action-btn:hover {
    background: var(--bg-card-hover);
    border-color: var(--primary);
    color: var(--primary);
    transform: translateY(-2px);
}

.quick-action-icon {
    font-size: 1.5rem;
}

.quick-action-label {
    font-size: 0.8rem;
    font-weight: 500;
}

</style>
"""


def get_animated_background():
    """Return subtle animated background gradient - CSS only, no HTML elements."""
    return """
<style>
/* Signature mesh gradient background */
.stApp {
    background: var(--bg-dark) !important;
    background-image: var(--gradient-mesh) !important;
    background-attachment: fixed;
}

/* Subtle ambient glow effects */
.stApp::before {
    content: '';
    position: fixed;
    top: 5%;
    left: 10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255, 107, 107, 0.06), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: pulse-glow 8s ease-in-out infinite;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: 15%;
    right: 5%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(78, 205, 196, 0.05), transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: pulse-glow 12s ease-in-out infinite reverse;
}

@keyframes pulse-glow {
    0%, 100% { transform: scale(1); opacity: 0.4; }
    50% { transform: scale(1.05); opacity: 0.7; }
}
</style>
"""


def create_hero_header(title: str, subtitle: str = ""):
    """Create a signature DataDojo hero header with animated underline."""
    return f"""
<div class="hero-header animate-fade-in">
    <h1 class="hero-title">🥋 {title}</h1>
    <p class="hero-subtitle">{subtitle}</p>
</div>
"""


def create_metric_card(icon: str, value: str, label: str, color: str = "primary"):
    """Create a modern metric card with accent bar on hover."""
    return f"""
<div class="metric-card-modern">
    <div class="metric-icon">{icon}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-label">{label}</div>
</div>
"""


def create_feature_card(icon: str, title: str, description: str):
    """Create an interactive feature card with gradient overlay on hover."""
    return f"""
<div class="feature-card">
    <div class="feature-icon">{icon}</div>
    <h3 class="feature-title">{title}</h3>
    <p class="feature-desc">{description}</p>
</div>
"""


def create_glass_card(content: str):
    """Wrap content in a refined card with subtle top highlight."""
    return f"""
<div class="glass-card">
    {content}
</div>
"""


def create_badge(text: str, variant: str = "primary"):
    """Create a styled badge with proper typography."""
    return f'<span class="badge badge-{variant}">{text}</span>'


def create_divider():
    """Create a gradient divider for section separation."""
    return '<div class="divider"></div>'


def create_stat_card(icon: str, value: str, label: str, trend: str | None = None):
    """Create a stat card with optional trend indicator."""
    trend_html = ""
    if trend:
        trend_color = "var(--success)" if trend.startswith("+") else "var(--error)"
        trend_html = f'<span style="color: {trend_color}; font-size: 0.85rem; font-family: var(--font-mono);">{trend}</span>'
    
    return f"""
<div class="metric-card-modern">
    <div class="metric-icon">{icon}</div>
    <div class="metric-value">{value}</div>
    {trend_html}
    <div class="metric-label">{label}</div>
</div>
"""


def create_nav_item(icon: str, label: str, active: bool = False):
    """Create a navigation item with icon."""
    active_class = "active" if active else ""
    return f"""
<div class="nav-item {active_class}">
    <span class="nav-icon">{icon}</span>
    <span class="nav-label">{label}</span>
</div>
"""


def create_sidebar_stat(value: str, label: str, color: str = "primary"):
    """Create a sidebar statistic display."""
    return f"""
<div class="sidebar-stat {color}">
    <span class="stat-value">{value}</span>
    <span class="stat-label">{label}</span>
</div>
"""


def create_skeleton_loader(variant: str = "card"):
    """Create a skeleton loading placeholder."""
    if variant == "card":
        return '<div class="skeleton skeleton-card"></div>'
    elif variant == "text":
        return '''
<div class="skeleton skeleton-title"></div>
<div class="skeleton skeleton-text"></div>
<div class="skeleton skeleton-text short"></div>
'''
    elif variant == "avatar":
        return '<div class="skeleton skeleton-avatar"></div>'
    return '<div class="skeleton skeleton-text"></div>'


def create_empty_state(icon: str, title: str, description: str, action_text: str = "", action_icon: str = ""):
    """Create an empty state component."""
    action_html = ""
    if action_text:
        action_html = f'''
<button class="empty-state-action">
    {action_icon} {action_text}
</button>
'''
    return f"""
<div class="empty-state">
    <div class="empty-state-icon">{icon}</div>
    <h3 class="empty-state-title">{title}</h3>
    <p class="empty-state-desc">{description}</p>
    {action_html}
</div>
"""


def create_toast(message: str, title: str = "", variant: str = "info"):
    """Create a toast notification HTML."""
    icons = {
        "success": "✓",
        "error": "✕",
        "warning": "⚠",
        "info": "ℹ"
    }
    icon = icons.get(variant, "ℹ")
    title_html = f'<div class="toast-title">{title}</div>' if title else ""
    
    return f"""
<div class="toast {variant}">
    <span class="toast-icon">{icon}</span>
    <div class="toast-content">
        {title_html}
        <div class="toast-message">{message}</div>
    </div>
</div>
"""


def create_data_card(name: str, domain: str, rows: int, cols: int, size: str):
    """Create an enhanced data card for datasets."""
    domain_colors = {
        "healthcare": "#4ECDC4",
        "finance": "#FFD93D",
        "ecommerce": "#FF6B6B",
        "e-commerce": "#FF6B6B"
    }
    domain_color = domain_colors.get(domain.lower(), "#C9B1FF")
    
    return f"""
<div class="data-card hover-lift">
    <div class="data-card-header">
        <span class="data-card-title">
            <span style="color: {domain_color};">📊</span> {name}
        </span>
        <span class="data-card-badge">{domain.title()}</span>
    </div>
    <div class="data-card-stats">
        <div class="data-card-stat">
            <div class="data-card-stat-value">{rows:,}</div>
            <div class="data-card-stat-label">Rows</div>
        </div>
        <div class="data-card-stat">
            <div class="data-card-stat-value">{cols}</div>
            <div class="data-card-stat-label">Columns</div>
        </div>
        <div class="data-card-stat">
            <div class="data-card-stat-value">{size}</div>
            <div class="data-card-stat-label">Size</div>
        </div>
    </div>
</div>
"""


def create_quick_actions():
    """Create a quick actions panel."""
    return """
<div class="quick-actions">
    <div class="quick-action-btn">
        <span class="quick-action-icon">📊</span>
        <span class="quick-action-label">New Profile</span>
    </div>
    <div class="quick-action-btn">
        <span class="quick-action-icon">🔧</span>
        <span class="quick-action-label">Generate Data</span>
    </div>
    <div class="quick-action-btn">
        <span class="quick-action-icon">📓</span>
        <span class="quick-action-label">New Notebook</span>
    </div>
    <div class="quick-action-btn">
        <span class="quick-action-icon">📈</span>
        <span class="quick-action-label">View Progress</span>
    </div>
</div>
"""


def create_sparkline(values: list, color: str = "var(--primary)"):
    """Create a mini sparkline chart."""
    if not values:
        return ""
    max_val = max(values) if values else 1
    bars = ""
    for val in values:
        height = max(4, int((val / max_val) * 28))
        bars += f'<div class="sparkline-bar" style="height: {height}px; background: {color};"></div>'
    
    return f'<div class="sparkline-container">{bars}</div>'


def create_onboarding_step(step: int, total: int, title: str, description: str):
    """Create an onboarding tour step."""
    dots = ""
    for i in range(total):
        if i < step:
            dots += '<div class="tour-progress-dot completed"></div>'
        elif i == step:
            dots += '<div class="tour-progress-dot active"></div>'
        else:
            dots += '<div class="tour-progress-dot"></div>'
    
    return f"""
<div class="tour-tooltip">
    <div class="tour-progress">{dots}</div>
    <h4>{title}</h4>
    <p>{description}</p>
    <div class="tour-actions">
        <button class="tour-btn tour-btn-skip">Skip Tour</button>
        <button class="tour-btn tour-btn-next">Next →</button>
    </div>
</div>
"""
