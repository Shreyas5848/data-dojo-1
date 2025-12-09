"""
DataDojo Visualization Components
Advanced visualization utilities for the web dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt


class DataVisualizationEngine:
    """Advanced visualization engine with automated chart recommendations."""
    
    def __init__(self):
        self.color_palette = [
            '#FF9900', '#FFB366', '#FFCC80', '#FFE0B2', 
            '#FF8A65', '#FF7043', '#FF5722', '#E64A19'
        ]
        self.dark_theme_config = {
            'plot_bgcolor': 'rgba(14,17,23,0)',
            'paper_bgcolor': 'rgba(14,17,23,0)',
            'font_color': '#FAFAFA',
            'grid_color': '#404040'
        }
    
    def recommend_visualizations(self, df: pd.DataFrame) -> List[Dict]:
        """Recommend appropriate visualizations based on data characteristics."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        
        # Detect datetime columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    # Try to parse a sample
                    sample = df[col].dropna().head(100)
                    if len(sample) > 0:
                        pd.to_datetime(sample, errors='raise')
                        datetime_cols.append(col)
                except:
                    continue
        
        # Univariate recommendations
        for col in numeric_cols:
            recommendations.append({
                'type': 'histogram',
                'title': f'Distribution of {col}',
                'columns': [col],
                'description': f'Shows the frequency distribution of {col} values'
            })
            
            recommendations.append({
                'type': 'box',
                'title': f'Box Plot of {col}',
                'columns': [col],
                'description': f'Shows outliers and quartiles for {col}'
            })
        
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only for reasonable number of categories
                recommendations.append({
                    'type': 'bar',
                    'title': f'Count of {col}',
                    'columns': [col],
                    'description': f'Shows frequency of each {col} category'
                })
        
        # Bivariate recommendations
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    recommendations.append({
                        'type': 'scatter',
                        'title': f'{col1} vs {col2}',
                        'columns': [col1, col2],
                        'description': f'Shows relationship between {col1} and {col2}'
                    })
        
        # Mixed type recommendations
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                if df[cat_col].nunique() <= 10:
                    recommendations.append({
                        'type': 'box_by_category',
                        'title': f'{num_col} by {cat_col}',
                        'columns': [num_col, cat_col],
                        'description': f'Distribution of {num_col} across {cat_col} categories'
                    })
        
        # Time series recommendations
        for dt_col in datetime_cols:
            for num_col in numeric_cols:
                recommendations.append({
                    'type': 'time_series',
                    'title': f'{num_col} over {dt_col}',
                    'columns': [dt_col, num_col],
                    'description': f'Shows how {num_col} changes over time'
                })
        
        # Correlation heatmap
        if len(numeric_cols) >= 3:
            recommendations.append({
                'type': 'correlation',
                'title': 'Correlation Heatmap',
                'columns': numeric_cols,
                'description': 'Shows correlations between all numeric variables'
            })
        
        return recommendations[:15]  # Limit to top 15 recommendations
    
    def create_visualization(self, df: pd.DataFrame, viz_config: Dict) -> go.Figure:
        """Create visualization based on configuration."""
        viz_type = viz_config['type']
        columns = viz_config['columns']
        title = viz_config['title']
        
        if viz_type == 'histogram':
            return self._create_histogram(df, columns[0], title)
        elif viz_type == 'box':
            return self._create_box_plot(df, columns[0], title)
        elif viz_type == 'bar':
            return self._create_bar_chart(df, columns[0], title)
        elif viz_type == 'scatter':
            return self._create_scatter_plot(df, columns[0], columns[1], title)
        elif viz_type == 'box_by_category':
            return self._create_box_by_category(df, columns[0], columns[1], title)
        elif viz_type == 'time_series':
            return self._create_time_series(df, columns[0], columns[1], title)
        elif viz_type == 'correlation':
            return self._create_correlation_heatmap(df, columns, title)
        else:
            raise ValueError(f"Unknown visualization type: {viz_type}")
    
    def _create_histogram(self, df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """Create histogram visualization."""
        fig = px.histogram(
            df, 
            x=column, 
            title=title,
            color_discrete_sequence=self.color_palette,
            nbins=30
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """Create box plot visualization."""
        fig = px.box(
            df, 
            y=column, 
            title=title,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, column: str, title: str) -> go.Figure:
        """Create bar chart visualization."""
        value_counts = df[column].value_counts()
        
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=title,
            labels={'x': column, 'y': 'Count'},
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
        """Create scatter plot visualization."""
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col, 
            title=title,
            color_discrete_sequence=self.color_palette,
            opacity=0.7
        )
        
        # Add trendline
        try:
            from sklearn.linear_model import LinearRegression
            X = df[[x_col]].dropna()
            y = df[y_col].dropna()
            
            if len(X) > 1 and len(y) > 1:
                # Align X and y
                common_idx = X.index.intersection(y.index)
                X_aligned = X.loc[common_idx]
                y_aligned = y.loc[common_idx]
                
                if len(X_aligned) > 1:
                    model = LinearRegression()
                    model.fit(X_aligned, y_aligned)
                    y_pred = model.predict(X_aligned)
                    
                    fig.add_trace(go.Scatter(
                        x=X_aligned[x_col],
                        y=y_pred,
                        mode='lines',
                        name='Trendline',
                        line=dict(color='red', dash='dash')
                    ))
        except:
            pass  # Skip trendline if sklearn not available
        
        fig.update_layout(
            plot_bgcolor=self.dark_theme_config['plot_bgcolor'],
            paper_bgcolor=self.dark_theme_config['paper_bgcolor'],
            font_color=self.dark_theme_config['font_color'],
            xaxis=dict(gridcolor=self.dark_theme_config['grid_color']),
            yaxis=dict(gridcolor=self.dark_theme_config['grid_color'])
        )
        return fig
    
    def _create_box_by_category(self, df: pd.DataFrame, num_col: str, cat_col: str, title: str) -> go.Figure:
        """Create box plot by category visualization."""
        fig = px.box(
            df, 
            x=cat_col, 
            y=num_col, 
            title=title,
            color=cat_col,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_time_series(self, df: pd.DataFrame, date_col: str, value_col: str, title: str) -> go.Figure:
        """Create time series visualization."""
        # Convert to datetime
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_clean = df_copy.dropna(subset=[date_col, value_col])
        
        # Group by date if needed
        if len(df_clean) > 1000:
            df_clean = df_clean.groupby(df_clean[date_col].dt.date)[value_col].mean().reset_index()
            df_clean.columns = [date_col, value_col]
        
        fig = px.line(
            df_clean.sort_values(date_col), 
            x=date_col, 
            y=value_col, 
            title=title,
            color_discrete_sequence=self.color_palette
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, columns: List[str], title: str) -> go.Figure:
        """Create correlation heatmap visualization."""
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu_r',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 12}
        ))
        
        fig.update_layout(
            title=title,
            plot_bgcolor=self.dark_theme_config['plot_bgcolor'],
            paper_bgcolor=self.dark_theme_config['paper_bgcolor'],
            font_color=self.dark_theme_config['font_color']
        )
        return fig
    
    def create_quality_dashboard(self, profile) -> List[go.Figure]:
        """Create data quality dashboard visualizations."""
        figures = []
        
        # Quality scores radar chart
        categories = ['Overall Quality', 'Completeness', 'Consistency', 'Uniqueness']
        values = [
            profile.overall_quality_score,
            profile.completeness_score,
            profile.consistency_score,
            profile.uniqueness_score
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            marker_color='#FF9900'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Data Quality Scores",
            showlegend=False
        )
        figures.append(fig_radar)
        
        # Missing data by column
        if profile.column_profiles:
            columns = []
            missing_pcts = []
            
            for col_name, col_profile in profile.column_profiles.items():
                columns.append(col_name[:15] + '...' if len(col_name) > 15 else col_name)
                missing_pcts.append(col_profile.null_percentage)
            
            fig_missing = go.Figure(data=go.Bar(
                x=columns,
                y=missing_pcts,
                marker_color='#FF9900'
            ))
            
            fig_missing.update_layout(
                title="Missing Data by Column",
                xaxis_title="Columns",
                yaxis_title="Missing Percentage (%)",
                xaxis_tickangle=-45
            )
            figures.append(fig_missing)
        
        return figures


def create_data_quality_summary_card(profile) -> str:
    """Create HTML for data quality summary card with dark theme."""
    
    quality_color = "#00D084" if profile.overall_quality_score > 0.8 else "#FFB020" if profile.overall_quality_score > 0.6 else "#FF6B6B"
    
    # Build compact HTML to avoid Streamlit rendering issues
    score_pct = f"{profile.overall_quality_score:.1%}"
    completeness_pct = f"{profile.completeness_score:.1%}"
    consistency_pct = f"{profile.consistency_score:.1%}"
    uniqueness_pct = f"{profile.uniqueness_score:.1%}"
    shape_str = f"{profile.shape[0]:,} rows × {profile.shape[1]} columns"
    memory_str = f"{profile.memory_usage_mb:.1f} MB"
    duplicates_str = f"{profile.duplicate_rows:,} ({profile.duplicate_percentage:.1f}%)"
    
    html = f'<div style="border: 1px solid {quality_color}; border-radius: 12px; padding: 20px; margin: 20px 0; background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);">'
    html += f'<div style="text-align: center;">'
    html += f'<h2 style="color: #94A3B8; margin: 0; font-size: 1rem;">Data Quality Score</h2>'
    html += f'<h1 style="color: {quality_color}; font-size: 2.5rem; margin: 10px 0;">{score_pct}</h1>'
    html += '</div>'
    html += '<div style="display: flex; justify-content: space-around; margin-top: 20px;">'
    html += f'<div style="text-align: center;"><h4 style="color: #64748B; margin: 5px 0; font-size: 0.8rem;">Completeness</h4><h3 style="color: {quality_color}; margin: 0;">{completeness_pct}</h3></div>'
    html += f'<div style="text-align: center;"><h4 style="color: #64748B; margin: 5px 0; font-size: 0.8rem;">Consistency</h4><h3 style="color: {quality_color}; margin: 0;">{consistency_pct}</h3></div>'
    html += f'<div style="text-align: center;"><h4 style="color: #64748B; margin: 5px 0; font-size: 0.8rem;">Uniqueness</h4><h3 style="color: {quality_color}; margin: 0;">{uniqueness_pct}</h3></div>'
    html += '</div>'
    html += '<div style="margin-top: 20px; padding: 15px; background: #0F172A; border-radius: 8px; border: 1px solid #334155;">'
    html += '<h4 style="margin: 0 0 10px 0; color: #F8FAFC; font-size: 0.9rem;">Dataset Overview</h4>'
    html += f'<p style="margin: 5px 0; color: #94A3B8; font-size: 0.85rem;"><strong style="color: #F8FAFC;">Shape:</strong> {shape_str}</p>'
    html += f'<p style="margin: 5px 0; color: #94A3B8; font-size: 0.85rem;"><strong style="color: #F8FAFC;">Memory:</strong> {memory_str}</p>'
    html += f'<p style="margin: 5px 0; color: #94A3B8; font-size: 0.85rem;"><strong style="color: #F8FAFC;">Duplicates:</strong> {duplicates_str}</p>'
    html += '</div></div>'
    
    return html