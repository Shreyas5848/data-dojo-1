"""
Tests for Notebook Template functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json


class TestNotebookTemplateEngine:
    """Tests for NotebookTemplateEngine."""

    @pytest.fixture
    def template_engine(self):
        """Create template engine instance."""
        from datadojo.notebook.template_engine import NotebookTemplateEngine
        return NotebookTemplateEngine()

    @pytest.fixture
    def sample_profile(self):
        """Create sample profile results."""
        return {
            'total_rows': 100,
            'total_columns': 5,
            'total_missing': 10,
            'duplicate_rows': 2,
            'numeric_columns': ['age', 'income', 'score'],
            'categorical_columns': ['gender', 'category'],
            'columns_with_missing': ['income'],
            'missing_values_summary': {'age': 0, 'income': 5, 'score': 5, 'gender': 0, 'category': 0},
            'overall_quality_score': 85.0
        }

    def test_template_engine_initialization(self, template_engine):
        """Test template engine initializes correctly."""
        assert template_engine is not None

    def test_get_available_templates(self):
        """Test getting available template types."""
        from datadojo.notebook.template_engine import get_available_templates
        
        templates = get_available_templates()
        
        assert isinstance(templates, dict)
        assert len(templates) >= 8  # We have 8 template types
        assert 'exploratory_data_analysis' in templates
        assert 'data_cleaning' in templates
        assert 'classification_analysis' in templates
        assert 'regression_analysis' in templates
        assert 'time_series_analysis' in templates
        assert 'clustering_analysis' in templates
        assert 'dimensionality_reduction' in templates
        assert 'feature_engineering' in templates

    def test_generate_eda_notebook(self, template_engine, sample_profile):
        """Test generating EDA notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'exploratory_data_analysis', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert hasattr(notebook, 'cells')
        assert len(notebook.cells) > 0
        
        # Check for markdown and code cells
        cell_types = [cell.cell_type for cell in notebook.cells]
        assert 'markdown' in cell_types
        assert 'code' in cell_types

    def test_generate_cleaning_notebook(self, template_engine, sample_profile):
        """Test generating data cleaning notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'data_cleaning', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_classification_notebook(self, template_engine, sample_profile):
        """Test generating classification notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'classification_analysis', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_regression_notebook(self, template_engine, sample_profile):
        """Test generating regression notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'regression_analysis', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_time_series_notebook(self, template_engine, sample_profile):
        """Test generating time series notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'time_series_analysis', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_clustering_notebook(self, template_engine, sample_profile):
        """Test generating clustering notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'clustering_analysis', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_dimensionality_reduction_notebook(self, template_engine, sample_profile):
        """Test generating dimensionality reduction notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'dimensionality_reduction', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_generate_feature_engineering_notebook(self, template_engine, sample_profile):
        """Test generating feature engineering notebook."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'feature_engineering', 
            'test_dataset'
        )
        
        assert notebook is not None
        assert len(notebook.cells) > 0

    def test_notebook_has_correct_format(self, template_engine, sample_profile):
        """Test that generated notebook has correct format."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'exploratory_data_analysis', 
            'test_dataset'
        )
        
        # Check notebook metadata
        assert hasattr(notebook, 'metadata')
        assert 'kernelspec' in notebook.metadata
        
        # Check nbformat version
        assert notebook.nbformat >= 4

    def test_notebook_cells_have_source(self, template_engine, sample_profile):
        """Test that all cells have source content."""
        notebook = template_engine.generate_notebook(
            sample_profile, 
            'exploratory_data_analysis', 
            'test_dataset'
        )
        
        for cell in notebook.cells:
            assert hasattr(cell, 'source')
            assert cell.source is not None


class TestProgressTracking:
    """Tests for Progress Tracking functionality."""

    @pytest.fixture
    def temp_progress_file(self):
        """Create temporary progress file."""
        temp_dir = tempfile.mkdtemp()
        progress_file = Path(temp_dir) / "user_progress.json"
        yield progress_file
        shutil.rmtree(temp_dir)

    def test_default_progress_structure(self):
        """Test default progress data structure."""
        from datadojo.web.progress_interface import get_default_progress
        
        progress = get_default_progress()
        
        assert 'user_info' in progress
        assert 'skills' in progress
        assert 'achievements' in progress
        assert 'activities' in progress
        assert 'notebooks_generated' in progress
        assert 'datasets_profiled' in progress
        
        # Check user_info structure
        assert 'level' in progress['user_info']
        assert 'xp' in progress['user_info']
        
        # Check skills structure
        assert len(progress['skills']) == 8
        for skill in progress['skills'].values():
            assert 'level' in skill
            assert 'xp' in skill
            assert 'max_xp' in skill

    def test_calculate_level(self):
        """Test level calculation from XP."""
        from datadojo.web.progress_interface import calculate_level
        
        # Level 1 with 0 XP
        level, current_xp, next_level_xp = calculate_level(0)
        assert level == 1
        assert current_xp == 0
        
        # Level 2 with 100 XP
        level, current_xp, next_level_xp = calculate_level(100)
        assert level == 2
        
        # Level 3 with 300 XP
        level, current_xp, next_level_xp = calculate_level(300)
        assert level == 3

    def test_add_xp(self):
        """Test adding XP to progress."""
        from datadojo.web.progress_interface import add_xp, get_default_progress
        
        progress = get_default_progress()
        
        # Add XP to data_exploration skill
        progress = add_xp(progress, 'data_exploration', 50, 'Test Activity')
        
        assert progress['skills']['data_exploration']['xp'] == 50
        assert progress['user_info']['xp'] == 50
        assert len(progress['activities']) == 1
        assert progress['activities'][0]['name'] == 'Test Activity'

    def test_check_achievements(self):
        """Test achievement checking."""
        from datadojo.web.progress_interface import check_achievements, get_default_progress
        
        progress = get_default_progress()
        progress['notebooks_generated'] = 5
        
        progress = check_achievements(progress)
        
        # Should have "First Notebook" and "Notebook Ninja" achievements
        achievement_names = [a['name'] for a in progress['achievements']]
        assert 'First Notebook' in achievement_names
        assert 'Notebook Ninja' in achievement_names


class TestNotebookInterface:
    """Tests for Notebook Interface functions."""

    def test_generate_profile_summary(self):
        """Test generating profile summary from DataFrame."""
        from datadojo.web.notebook_interface import generate_profile_summary
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'age': [25, 30, 35, None, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        profile = generate_profile_summary(df)
        
        assert profile['total_rows'] == 5
        assert profile['total_columns'] == 3
        assert profile['total_missing'] == 1
        assert 'age' in profile['numeric_columns'] or 'income' in profile['numeric_columns']
        assert 'category' in profile['categorical_columns']

    def test_recommend_template(self):
        """Test template recommendation based on data."""
        from datadojo.web.notebook_interface import recommend_template
        
        # Create profile with high missing values
        profile_with_missing = {
            'total_rows': 100,
            'total_columns': 5,
            'total_missing': 100,  # High missing
            'numeric_columns': ['a', 'b'],
            'categorical_columns': ['c']
        }
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': ['x', 'y']})
        
        recommendation = recommend_template(profile_with_missing, df)
        assert recommendation == 'data_cleaning'

    def test_get_template_features(self):
        """Test getting template features."""
        from datadojo.web.notebook_interface import get_template_features
        
        features = get_template_features('exploratory_data_analysis')
        
        assert isinstance(features, list)
        assert len(features) > 0

    def test_get_template_sections(self):
        """Test getting template sections."""
        from datadojo.web.notebook_interface import get_template_sections
        
        sections = get_template_sections('classification_analysis')
        
        assert isinstance(sections, list)
        assert len(sections) > 0
        assert any('Model' in s for s in sections)

    def test_list_saved_notebooks_empty(self):
        """Test listing saved notebooks when none exist."""
        from datadojo.web.notebook_interface import list_saved_notebooks
        
        notebooks = list_saved_notebooks()
        
        assert isinstance(notebooks, list)


class TestHelpInterface:
    """Tests for Help Interface functions."""

    def test_help_content_structure(self):
        """Test that help content has expected structure."""
        # This tests that the help interface module can be imported
        from datadojo.web.help_interface import render_help_page
        
        # Function should exist and be callable
        assert callable(render_help_page)
