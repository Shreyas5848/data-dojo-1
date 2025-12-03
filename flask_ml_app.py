"""
Standalone Flask ML Interface - Complete alternative to Streamlit
No TypedDict, no serialization issues, just pure Python web interface.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import json
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from datadojo.ml.simple_ml_engine import SimpleMLEngine

app = Flask(__name__)
app.secret_key = 'datadojo-ml-interface'

# Global engine instance
ml_engine = SimpleMLEngine()
current_data = None

@app.route('/')
def index():
    """Main ML pipeline interface."""
    return render_template('ml_interface.html')

@app.route('/upload', methods=['POST'])
def upload_data():
    """Handle CSV file upload."""
    global current_data, ml_engine
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Read CSV data
            df = pd.read_csv(file.stream)
            current_data = df
            
            # Reset engine
            ml_engine = SimpleMLEngine()
            ml_engine.create_pipeline()
            
            return jsonify({
                'success': True,
                'rows': len(df),
                'columns': list(df.columns),
                'preview': df.head().to_dict('records')
            })
        else:
            return jsonify({'error': 'Please upload a CSV file'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo-data', methods=['POST'])
def load_demo_data():
    """Load demo data."""
    global current_data, ml_engine
    
    try:
        # Create demo data
        demo_data = pd.DataFrame({
            'age': [25, 35, 45, 30, 50, 28, 40, 33, 55, 29],
            'income': [50000, 75000, 90000, 60000, 120000, 45000, 85000, 70000, 110000, 52000],
            'experience': [2, 8, 15, 5, 20, 1, 12, 6, 18, 3],
            'score': [75, 85, 95, 80, 98, 70, 90, 82, 96, 78],
            'category': ['A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'B', 'A']
        })
        
        current_data = demo_data
        
        # Reset engine
        ml_engine = SimpleMLEngine()
        ml_engine.create_pipeline()
        
        return jsonify({
            'success': True,
            'rows': len(demo_data),
            'columns': list(demo_data.columns),
            'preview': demo_data.head().to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/run-step', methods=['POST'])
def run_step():
    """Execute a pipeline step."""
    global current_data, ml_engine
    
    try:
        if current_data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.get_json()
        step_index = data.get('step_index', 0)
        target_column = data.get('target_column')
        
        if step_index >= len(ml_engine.steps):
            return jsonify({'error': 'Invalid step index'}), 400
        
        step = ml_engine.steps[step_index]
        
        # Prepare kwargs
        kwargs = {}
        if step.name == "train_model" and target_column:
            kwargs['target_col'] = target_column
        
        # Execute step
        result = ml_engine.execute_step(step, current_data, **kwargs)
        
        return jsonify({
            'success': True,
            'step_name': step.name,
            'step_status': step.status,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/pipeline-status')
def pipeline_status():
    """Get current pipeline status."""
    global ml_engine
    
    steps_info = []
    for i, step in enumerate(ml_engine.steps):
        steps_info.append({
            'index': i,
            'name': step.name,
            'description': step.description,
            'status': step.status,
            'result': step.result
        })
    
    return jsonify({
        'steps': steps_info,
        'total_steps': len(ml_engine.steps)
    })

@app.route('/reset', methods=['POST'])
def reset_pipeline():
    """Reset the entire pipeline."""
    global ml_engine, current_data
    
    ml_engine = SimpleMLEngine()
    ml_engine.create_pipeline()
    current_data = None
    
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8508)