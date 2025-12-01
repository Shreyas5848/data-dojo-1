"""
Web Interface Launch Command
Automatically start the DataDojo web dashboard with smart port management.
"""

import subprocess
import sys
import webbrowser
import time
import socket
from pathlib import Path
from typing import Optional, Tuple
import threading
import os

from .interface import CLIResult


def find_available_port(start_port: int = 8501, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    
    # Fallback to a random available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]


def check_dependencies() -> Tuple[bool, str]:
    """Check if required web dependencies are installed."""
    try:
        import streamlit
        import plotly
        return True, f"Streamlit {streamlit.__version__} ready"
    except ImportError as e:
        missing = str(e).split("'")[1] if "'" in str(e) else "streamlit"
        return False, f"Missing dependency: {missing}"


def launch_web_dashboard(
    port: Optional[int] = None,
    auto_open: bool = True,
    debug: bool = False
) -> CLIResult:
    """
    Launch the DataDojo web dashboard.
    
    Args:
        port: Specific port to use (auto-detected if None)
        auto_open: Whether to automatically open browser
        debug: Enable debug mode
    
    Returns:
        CLIResult with launch status
    """
    
    # Check dependencies first
    deps_ok, deps_msg = check_dependencies()
    if not deps_ok:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Web dependencies not installed: {deps_msg}\n"
                         f"Install with: pip install -r requirements-web.txt"
        )
    
    # Find the app.py file
    app_path = Path(__file__).parent.parent.parent.parent / "app.py"
    if not app_path.exists():
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Web app not found at {app_path}\n"
                         f"Make sure app.py exists in the project root."
        )
    
    # Determine port
    if port is None:
        port = find_available_port()
    
    # Prepare launch message
    launch_msg = f"""
🚀 LAUNCHING DATADOJO WEB DASHBOARD
{'=' * 50}

📍 Local URL: http://localhost:{port}
🌐 Network URL: http://0.0.0.0:{port}
📂 App Location: {app_path}
🔧 Debug Mode: {'ON' if debug else 'OFF'}

⚡ Features Available:
  • Interactive Dataset Explorer
  • AI-Powered Data Profiler  
  • Synthetic Data Generator
  • Advanced Visualizations

💡 Tips:
  • Use Ctrl+C to stop the server
  • Refresh browser if connection fails
  • Check firewall if network access needed

Starting server..."""
    
    print(launch_msg)
    
    try:
        # Prepare streamlit command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true" if not auto_open else "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        if debug:
            cmd.extend(["--logger.level", "debug"])
        
        # Change to app directory for proper imports
        cwd = str(app_path.parent)
        
        # Start streamlit in background if not interactive
        if auto_open:
            # Start server and wait a bit
            process = subprocess.Popen(
                cmd, 
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Open browser
            url = f"http://localhost:{port}"
            try:
                webbrowser.open(url)
                print(f"\n✅ Browser opened at {url}")
            except Exception as e:
                print(f"\n⚠️  Could not open browser automatically: {e}")
                print(f"   Please open {url} manually")
            
            # Keep process running
            try:
                print(f"\n🔄 Server running on port {port}. Press Ctrl+C to stop...")
                process.wait()
            except KeyboardInterrupt:
                print(f"\n🛑 Stopping DataDojo web server...")
                process.terminate()
                return CLIResult(
                    success=True,
                    output=f"Web dashboard stopped successfully.",
                    exit_code=0
                )
        
        else:
            # Run in foreground
            subprocess.run(cmd, cwd=cwd)
        
        return CLIResult(
            success=True,
            output=f"DataDojo web dashboard launched successfully at http://localhost:{port}",
            exit_code=0
        )
        
    except FileNotFoundError:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message="Streamlit not found. Install with: pip install streamlit"
        )
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to launch web dashboard: {str(e)}"
        )


def check_web_status(port: int = 8501) -> CLIResult:
    """Check if the web dashboard is running."""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}", timeout=2)
        if response.status_code == 200:
            return CLIResult(
                success=True,
                output=f"✅ DataDojo web dashboard is running at http://localhost:{port}",
                exit_code=0
            )
        else:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Web dashboard returned status {response.status_code}"
            )
    except ImportError:
        # Try socket connection instead
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                if result == 0:
                    return CLIResult(
                        success=True,
                        output=f"✅ Service running on port {port} (install requests for full check)",
                        exit_code=0
                    )
                else:
                    return CLIResult(
                        success=False,
                        output="",
                        exit_code=1,
                        error_message=f"No service found on port {port}"
                    )
        except Exception as e:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Connection check failed: {str(e)}"
            )
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Web dashboard not responding: {str(e)}"
        )


def main():
    """Command-line interface for web launch."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch DataDojo Web Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  datadojo web                    # Launch with auto-detected port
  datadojo web --port 8502        # Launch on specific port  
  datadojo web --no-browser       # Launch without opening browser
  datadojo web --status           # Check if running
        """
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        help="Port to run the web server on (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--no-browser", 
        action="store_true",
        help="Don't automatically open browser"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Check if web dashboard is running"
    )
    
    args = parser.parse_args()
    
    if args.status:
        result = check_web_status(args.port or 8501)
    else:
        result = launch_web_dashboard(
            port=args.port,
            auto_open=not args.no_browser,
            debug=args.debug
        )
    
    if result.success:
        if result.output:
            print(result.output)
        sys.exit(0)
    else:
        print(f"❌ Error: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()