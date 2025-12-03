"""
ML Pipeline CLI Command - Launch the ML Pipeline Builder
"""

import click
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datadojo.cli.web_launch import launch_web_dashboard


@click.command()
@click.option('--port', default=8505, help='Port to run the web interface')
@click.option('--host', default='localhost', help='Host to bind the web interface')  
@click.option('--no-browser', is_flag=True, help="Don't automatically open browser")
def ml_pipeline(port, host, no_browser):
    """
    Launch the ML Pipeline Builder web interface.
    
    Build machine learning models with a visual, educational interface.
    No coding required - perfect for beginners and experts!
    
    Examples:
        dojo ml-pipeline                    # Launch ML Pipeline Builder
        dojo ml-pipeline --port 8506        # Use custom port
    """
    click.echo("🤖 Launching ML Pipeline Builder...")
    click.echo("")
    click.echo("📚 What you can do:")
    click.echo("   • Build ML models visually")  
    click.echo("   • Learn as you go with explanations")
    click.echo("   • Use pre-built templates")
    click.echo("   • No coding required!")
    click.echo("")
    
    # Launch web dashboard directly to ML page
    launch_web_dashboard(
        port=port, 
        host=host, 
        auto_open=not no_browser
    )


if __name__ == '__main__':
    ml_pipeline()