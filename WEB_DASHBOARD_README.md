# DataDojo Web Dashboard 🥋📊

A professional web interface for the DataDojo platform, providing interactive data exploration, profiling, and visualization capabilities.

## Features

### 🏠 Dashboard Home
- **Project Overview**: Key statistics and metrics
- **Dataset Discovery**: Browse available datasets across domains
- **Quick Actions**: Fast access to common tasks
- **Progress Tracking**: Monitor your data science learning journey

### 🔍 Dataset Explorer
- **Smart Filtering**: Filter datasets by domain, size, and quality
- **Preview Capabilities**: Quick data preview with sample rows
- **Metadata Analysis**: File size, shape, and basic statistics
- **Batch Operations**: Profile multiple datasets simultaneously

### 📈 Intelligent Data Profiler
- **Quality Scoring**: AI-powered data quality assessment
- **Visual Analytics**: Automated visualization recommendations
- **Column Analysis**: Detailed profiling of each data column
- **Business Insights**: Actionable recommendations for data improvement
- **Interactive Charts**: Quality dashboards and trend analysis

### 🎲 Synthetic Data Generator
- **Multi-Domain Support**: Healthcare, Finance, E-commerce datasets
- **Configurable Quality**: Control data quality issues for learning
- **Realistic Patterns**: Generate data with real-world characteristics
- **Instant Download**: Export generated datasets immediately

## Installation

### Quick Start
```bash
# Install web dashboard dependencies
pip install -r requirements-web.txt

# Launch the dashboard
streamlit run app.py
```

### Full Installation
```bash
# Clone the repository
git clone <repository-url>
cd data-dojo-1

# Install all dependencies
pip install -e .
pip install -r requirements-web.txt

# Launch the dashboard
python -m streamlit run app.py --server.port 8502
```

## Usage Guide

### 1. Starting the Dashboard
```bash
cd data-dojo-1
streamlit run app.py
```
The dashboard will be available at `http://localhost:8501`

### 2. Exploring Datasets
- Navigate to **Dataset Explorer** from the sidebar
- Use filters to find datasets of interest
- Click **Preview** to see sample data
- Use **Profile Dataset** for detailed analysis

### 3. Data Profiling
- Select a dataset from the profiler page
- Click **🚀 Profile Dataset** to start analysis
- Review quality scores and recommendations
- Explore automated visualizations

### 4. Generating Synthetic Data
- Go to **Data Generator** page
- Choose domain (Healthcare, Finance, E-commerce)
- Configure dataset parameters
- Download generated CSV file

## Advanced Features

### Custom Visualizations
The dashboard automatically recommends appropriate visualizations:
- **Histograms** for numeric distributions
- **Box plots** for outlier detection
- **Scatter plots** for correlations
- **Time series** for temporal data
- **Heatmaps** for correlation analysis

### Quality Assessment
Comprehensive data quality scoring:
- **Completeness**: Missing value analysis
- **Consistency**: Data format validation
- **Uniqueness**: Duplicate detection
- **Overall Quality**: Composite score

### Business Intelligence
AI-generated insights including:
- Data quality issues and solutions
- Column-specific recommendations
- Business context analysis
- Improvement suggestions

## Configuration

### Theme Customization
Edit `src/datadojo/web/config.py` to customize:
- Color schemes
- Layout parameters
- Chart defaults
- Quality thresholds

### Performance Tuning
Adjust settings in config for:
- Maximum file sizes
- Chunk processing
- Cache duration
- Display limits

## API Integration

The web dashboard integrates with DataDojo's CLI components:

```python
# Use profiler programmatically
from datadojo.utils.intelligent_profiler import IntelligentProfiler
from datadojo.web.visualizations import DataVisualizationEngine

profiler = IntelligentProfiler()
viz_engine = DataVisualizationEngine()

# Profile dataset
profile = profiler.profile_dataset(df, "my_dataset")

# Generate visualizations
recommendations = viz_engine.recommend_visualizations(df)
for rec in recommendations:
    fig = viz_engine.create_visualization(df, rec)
```

## Troubleshooting

### Common Issues

**Dashboard won't start:**
- Check Python version (3.8+ required)
- Verify all dependencies installed
- Check port availability

**Visualizations not showing:**
- Ensure plotly is installed
- Check browser compatibility
- Clear browser cache

**Large file processing:**
- Adjust chunk size in config
- Increase memory limits
- Use data sampling for preview

### Performance Tips
- Use dataset filtering for large collections
- Enable caching for repeated operations  
- Limit visualization complexity for large datasets
- Consider data sampling for initial exploration

## Development

### Adding Custom Visualizations
1. Extend `DataVisualizationEngine` class
2. Add new visualization types
3. Update recommendation logic
4. Test with various datasets

### Custom Themes
1. Modify `DATADOJO_THEME` in config
2. Update CSS styles
3. Test color accessibility
4. Verify brand consistency

## Support

For issues and questions:
- Check the main DataDojo documentation
- Review configuration settings
- Test with sample datasets
- Verify dependency versions

---

Built with ❤️ using Streamlit, Plotly, and the DataDojo platform.