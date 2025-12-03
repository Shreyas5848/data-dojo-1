# DataDojo: AI-Powered Data Preparation Learning Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational framework that teaches data preprocessing skills through hands-on learning with real datasets. DataDojo combines production-ready pipeline tools with interactive educational guidance, supporting multiple domains (e-commerce, healthcare, finance) across three difficulty levels (beginner, intermediate, advanced).

## ✨ Features

- 🎓 **Interactive Learning**: Step-by-step guidance with explanations
- 📊 **Real Datasets**: Messy data that reflects actual challenges
- 🔧 **Production Ready**: Toggle between educational and production modes
- 🌐 **Multi-Domain**: E-commerce, healthcare, and finance datasets
- 📈 **Progress Tracking**: Visual dashboards and skill assessments
- ⚡ **High Performance**: <500ms guidance, scalable to 1M+ rows
- 📓 **Notebook Templates**: 8 template types for all data science tasks (NEW!)
- 🎯 **Smart Recommendations**: Auto-detect best analysis for your data (NEW!)
- 🏆 **Gamified Learning**: XP, levels, and achievements (NEW!)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/data-dojo.git
cd data-dojo

# Install in development mode
pip install -e .
```

### Your First Learning Project

```python
from datadojo import create_dojo
from datadojo.contracts.dojo_interface import Domain, Difficulty, OperationType

# Create a DataDojo instance
dojo = create_dojo()

# List available projects
projects = dojo.list_projects(domain=Domain.ECOMMERCE, difficulty=Difficulty.BEGINNER)

# Start a project
project = dojo.start_project(projects[0].id)
print(f"Started: {project.name}")

# Create a data cleaning pipeline
pipeline = project.create_pipeline("my_first_pipeline")

# Add processing steps
pipeline.add_step(
    step_id="clean_data",
    name="Clean Dataset",
    operation_type=OperationType.DATA_CLEANING,
    description="Remove duplicates and handle missing values",
    learned_concepts=["missing_values", "duplicates"]
)

# Get educational guidance
educational = dojo.get_educational_interface()
concept = educational.get_concept_explanation("missing_values")
print(f"{concept.title}:\n{concept.get_summary()}")

# Track your progress
progress = educational.get_progress("student-1", project.id)
progress.complete_step("clean_data")
progress.learn_concept("missing_values")
```

### Command Line Interface

```bash
# List available projects
datadojo list-projects --domain ecommerce

# Start a project
datadojo start-project ecommerce-customer-segmentation

# Explain a concept with full details
datadojo explain missing_values --detail full

# View your progress
datadojo show-progress --student student-1 --project project-id

# Validate a dataset
datadojo validate-data input.csv --checks missing,duplicates,outliers

# Execute a pipeline
datadojo pipeline execute pipeline-id --input data.csv --output results.csv
```

### Web Dashboard (NEW!)

Launch the interactive web interface:

```bash
# Start the Streamlit dashboard
streamlit run app.py

# Or use a specific port
streamlit run app.py --server.port 8501
```

**Dashboard Features:**
- 🏠 **Home**: Overview of datasets and quick stats
- 📁 **Dataset Explorer**: Browse and preview datasets
- 🔍 **Data Profiler**: Intelligent data quality analysis
- ⚡ **Data Generator**: Create synthetic datasets
- 📓 **Notebook Templates**: Generate Jupyter notebooks
- 📊 **Progress Dashboard**: Track your learning journey
- 📚 **Tutorial & Help**: Comprehensive guides and FAQ

## 📚 Learning Paths

### Beginner (Data Cleaning Fundamentals)
- Missing value handling strategies
- Duplicate detection and removal
- Data type validation and conversion
- Basic outlier identification

### Intermediate (Feature Engineering)
- Creating derived features
- Categorical encoding techniques
- Scaling and normalization
- Feature selection methods

### Advanced (Production Pipelines)
- Automated preprocessing pipelines
- Performance optimization
- Error handling and logging
- Integration with ML workflows

## 🏗️ Domains

### E-commerce
- Customer behavior analysis
- Product catalog cleanup
- Sales data preprocessing
- Recommendation system features

### Healthcare
- Patient data anonymization
- Clinical trial preprocessing
- Medical record standardization
- Outcome prediction features

### Finance
- Transaction fraud detection
- Risk assessment features
- Market data preparation
- Portfolio analysis

## 📊 Educational Content

DataDojo includes comprehensive explanations for key concepts:

## 📓 Notebook Templates (NEW!)

Generate professional Jupyter notebooks with a click:

### Available Templates
| Template | Algorithms/Techniques |
|----------|----------------------|
| 📊 EDA | Statistical analysis, distributions, correlations |
| 🧹 Data Cleaning | Missing values, duplicates, outliers |
| 🎯 Classification | 8 algorithms (Logistic Regression → XGBoost) |
| 📉 Regression | 7 algorithms (Linear → Gradient Boosting) |
| 📅 Time Series | ARIMA, Exponential Smoothing, Decomposition |
| 🔮 Clustering | K-Means, DBSCAN, Hierarchical |
| 📐 Dimensionality Reduction | PCA, t-SNE, UMAP |
| 🔧 Feature Engineering | Encoding, scaling, feature selection |

### Smart Features
- **Auto-Detect**: Analyzes your data and recommends best templates
- **Customization**: Select only the sections you need
- **Educational**: Each cell includes explanations for learning

📖 See [NOTEBOOK_TEMPLATES_README.md](NOTEBOOK_TEMPLATES_README.md) for complete documentation.

---

## 📚 Core Concepts

### Data Quality
- **Missing Values**: Strategies for handling gaps in data (imputation, deletion)
- **Outliers**: Detection and handling using IQR, Z-score, isolation forests
- **Duplicates**: Identification and removal strategies
- **Data Types**: Understanding and converting data types properly

### Transformations
- **Normalization**: Scaling features to standard ranges (0-1, -1-1)
- **Standardization**: Transforming to mean=0, std=1
- **Encoding**: Converting categorical variables to numerical (one-hot, label, target)

### Feature Engineering
- **Creating Features**: Interaction terms, polynomial features, date extractions
- **Binning**: Discretizing continuous variables
- **Aggregations**: Creating summary features

### Advanced Topics
- **Imbalanced Data**: SMOTE, undersampling, class weights
- **Dimensionality Reduction**: PCA, t-SNE, feature selection

Each concept includes:
- Clear explanations with analogies
- Code examples
- Related concepts for deeper learning
- Difficulty-adjusted guidance

## 🎯 Progress Tracking & Visualization

Track your learning journey with built-in visualizations:

```python
from datadojo.educational.visualization import create_visualizer

# Create visualizer
visualizer = create_visualizer(use_plotly=False)  # or True for interactive charts

# Visualize your progress
visualizer.plot_progress_timeline(progress)  # Steps completed over time
visualizer.plot_skill_radar(progress)  # Skill assessment radar chart
visualizer.plot_concept_mastery(progress, all_concepts)  # Concept learning status
visualizer.plot_completion_percentage(progress, total_steps)  # Overall completion

# Generate complete dashboard
visualizer.generate_progress_dashboard(
    progress=progress,
    total_steps=10,
    all_concepts=concept_list,
    output_dir="./my_progress"
)
```

## 🏗️ Architecture

DataDojo follows a clean, modular architecture:

```
datadojo/
├── contracts/          # Interface definitions (DojoInterface, ProjectInterface, etc.)
├── core/              # Core implementations (Dojo, Project, Pipeline, Educational)
├── models/            # Data models (LearningProject, Pipeline, ProcessingStep, etc.)
├── services/          # Business logic layer (ProjectService, PipelineService, etc.)
├── educational/       # Educational systems (concepts, guidance, visualization)
├── domains/           # Domain-specific modules (ecommerce, healthcare, finance)
├── storage/           # File-based storage implementations
├── cli/              # Command-line interface commands
├── config/           # Configuration management
└── utils/            # Utilities and custom exceptions
```

## 🧪 Testing

Comprehensive test suite with multiple test types:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/contract/ -v     # Contract tests (interface compliance)
pytest tests/unit/ -v         # Unit tests (models, services, utils)
pytest tests/integration/ -v  # Integration tests
pytest tests/performance/ -v  # Performance benchmarks

# Run only fast tests (exclude slow performance tests)
pytest -m "not slow"

# Run with coverage
pytest --cov=datadojo --cov-report=html
```

### Performance Requirements
- Guidance generation: < 500ms
- Storage operations: < 100ms
- Data processing: Scalable to 1M+ rows
- Concept lookups: < 1ms per lookup

## 📖 Examples

Jupyter notebooks in the `examples/` directory:

1. **01_getting_started.ipynb** - Introduction to DataDojo basics
2. **02_data_cleaning_workflow.ipynb** - Complete data cleaning pipeline
3. **03_progress_tracking.ipynb** - Progress tracking and visualization

## 🔧 Configuration

Configure DataDojo via file or environment variables:

### Configuration File (`~/.datadojo/config.json`)

```json
{
  "version": "0.1.0",
  "debug": false,
  "log_level": "INFO",
  "storage": {
    "base_path": "~/.datadojo",
    "enable_backup": true,
    "max_backups": 10
  },
  "educational": {
    "default_guidance_level": "detailed",
    "default_educational_mode": true,
    "show_hints": true,
    "progress_tracking_enabled": true
  },
  "pipeline": {
    "default_timeout_seconds": 300,
    "enable_caching": true,
    "max_workers": 4
  },
  "performance": {
    "chunk_size": 10000,
    "use_multiprocessing": true,
    "cache_size_mb": 100
  }
}
```

### Environment Variables

- `DATADOJO_STORAGE_PATH`: Base storage path
- `DATADOJO_EDUCATIONAL_MODE`: Enable/disable educational features (true/false)
- `DATADOJO_DEBUG`: Enable debug mode (true/false)
- `DATADOJO_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `DATADOJO_GUIDANCE_LEVEL`: Default guidance level (minimal, detailed, verbose)

## 🛠️ Development

```bash
# Clone repository
git clone https://github.com/yourusername/data-dojo.git
cd data-dojo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=datadojo --cov-report=html

# Format code (if using black)
black src/ tests/

# Type checking (if using mypy)
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by the need for better data preprocessing education
- Built with modern Python best practices
- Designed for both beginners and advanced learners
- Educational content curated from real-world data science workflows

## 📧 Support

- **Issues**: https://github.com/KVSSetty/data-dojo/issues
- **Discussions**: https://github.com/KVSSetty/data-dojo/discussions

---

**Happy Learning! 🥋📊**