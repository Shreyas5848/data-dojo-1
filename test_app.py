import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(page_title="Test", layout="wide")

st.title("Test App")
st.write("If you see this, the basic app works!")

try:
    from datadojo.web.styles import get_modern_css
    st.markdown(get_modern_css(), unsafe_allow_html=True)
    st.success("✅ CSS loaded successfully")
except Exception as e:
    st.error(f"❌ CSS failed: {e}")

try:
    from datadojo.cli.list_datasets import discover_datasets
    datasets = discover_datasets(['datasets'])
    st.success(f"✅ Found {len(datasets)} datasets")
except Exception as e:
    st.error(f"❌ Dataset discovery failed: {e}")

st.write("App is running!")
