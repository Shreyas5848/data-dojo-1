"""
Quick CLI Demo Script for DataDojo Dataset Exploration
Run this to start the interactive session and explore your datasets.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datadojo.core.dojo import Dojo
from datadojo.cli.interactive_session import DojoSession

def start_dataset_explorer():
    """Start the DataDojo CLI for dataset exploration."""
    
    print("🚀 Starting DataDojo Dataset Explorer...")
    print("=" * 60)
    
    try:
        # Initialize the dojo (you can customize this based on your needs)
        dojo = Dojo()
        
        # Create and start the interactive session
        session = DojoSession(dojo)
        
        print("💡 Quick Commands to Try:")
        print("   • list-datasets                    - Show all your datasets")
        print("   • list-datasets --domain healthcare - Filter by domain")
        print("   • profile-data --file <path>       - Profile a specific dataset")
        print("   • profile-all --domain healthcare  - Profile all healthcare datasets")
        print("   • generate-data --domain finance   - Generate new datasets")
        print("   • help                            - Show all available commands")
        print()
        
        # Start the interactive session
        session.run()
        
    except Exception as e:
        print(f"❌ Error starting DataDojo CLI: {e}")
        print("Make sure you're running this from the project root directory.")
        return False
    
    return True

if __name__ == "__main__":
    start_dataset_explorer()