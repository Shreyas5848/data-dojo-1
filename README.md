<p align="center">
  <h1 align="center">🥋 DataDojo</h1>
  <p align="center">
    <strong>Master Data Science Through Hands-On Practice</strong>
  </p>
  <p align="center">
    An interactive learning platform for data preprocessing, analysis, and machine learning
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status"></a>
</p>

---

## 📖 Overview

DataDojo is an educational framework designed to teach data science skills through practical, hands-on experience with real-world datasets. It combines an intuitive **web interface** with a powerful **command-line interface (CLI)**, supporting multiple domains including **Healthcare**, **Finance**, and **E-commerce**.

### ✨ Key Highlights

- 🎯 **Interactive Web Dashboard** - Beautiful dark-themed UI built with Streamlit
- 💻 **Powerful CLI** - Full-featured command-line tools for terminal workflows
- 📊 **Intelligent Data Profiling** - Automated data quality analysis with actionable insights
- 📓 **Smart Notebook Generation** - Auto-generate Jupyter notebooks for any analysis task
- 🔬 **Real-World Datasets** - Messy data that reflects actual industry challenges
- 📈 **Progress Tracking** - Gamified learning with XP, levels, and achievements
- 🏭 **Synthetic Data Generator** - Create realistic datasets for practice
- 🎓 **Educational Mode** - Step-by-step guidance with concept explanations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or poetry for package management

### Installation

```bash
# Clone the repository
git clone https://github.com/Shreyas5848/data-dojo-1.git
cd data-dojo-1

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install in development mode (for CLI access)
pip install -e .
```

### Launch the Web Interface

```bash
# Start the Streamlit dashboard
streamlit run app.py --server.port 8530

# The app will open automatically at http://localhost:8530
```

### Using the CLI

```bash
# Start an interactive learning session
datadojo learn

# Or use individual commands
datadojo --help
```

---

## 🌐 Web Dashboard

Launch the interactive web interface for a visual experience:

```bash
streamlit run app.py --server.port 8530
```

### Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview of datasets, quick stats, domain distribution |
| 📁 **Dataset Explorer** | Browse, filter, and preview datasets by domain |
| 🔍 **Data Profiler** | Intelligent data quality analysis with scores and recommendations |
| ⚡ **Data Generator** | Create synthetic datasets for Healthcare, Finance, E-commerce |
| 📓 **Notebook Templates** | Generate Jupyter notebooks for any analysis task |
| 📊 **Progress Dashboard** | Track XP, achievements, and learning streaks |
| 📚 **Tutorial & Help** | Comprehensive guides and FAQ |

---

## 💻 Command Line Interface (CLI)

DataDojo provides a comprehensive CLI for terminal-based workflows.

### Getting Started

```bash
# Show all available commands
datadojo --help

# Start interactive learning session (recommended for beginners)
datadojo learn

# Launch web dashboard from CLI
datadojo web
```

### Available Commands

#### 📋 Project Management

```bash
# List all available learning projects
datadojo list-projects

# Filter by domain and difficulty
datadojo list-projects --domain healthcare --difficulty beginner

# Start a specific project
datadojo start <project_id> --student <your_name>

# Show your learning progress
datadojo progress <student_id>
datadojo progress <student_id> --project <project_id> --format detailed
```

#### 📊 Data Operations

```bash
# List available datasets
datadojo list-datasets

# Profile a dataset (analyze quality, statistics, issues)
datadojo profile datasets/healthcare/patient_demographics.csv

# Validate data quality
datadojo validate data.csv --rules custom_rules.yaml --format detailed

# Generate synthetic data
datadojo generate --domain healthcare --rows 1000 --output synthetic_data.csv
```

#### 🔧 Pipeline Operations

```bash
# Run a preprocessing pipeline
datadojo pipeline --ops "clean,normalize,encode" --input data.csv --output cleaned.csv

# ML Pipeline builder
datadojo ml-pipeline
```

#### 📚 Learning & Education

```bash
# Explain a data science concept
datadojo explain missing_values
datadojo explain outliers --detail detailed --examples

# Practice a specific concept interactively
datadojo practice normalization --student <your_name> --project <project_id>

# Mark a learning step as complete
datadojo complete-step step_id --student <name> --project <project_id>
```

#### 🛠️ Utility Commands

```bash
# Check environment and diagnose issues
datadojo doctor

# Launch web dashboard
datadojo web --port 8530

# Check if web dashboard is running
datadojo web --status
```

### Interactive Mode

Start a guided interactive session:

```bash
datadojo learn
```

This launches an interactive REPL with:
- Guided project selection
- Step-by-step instructions
- Real-time feedback
- Progress tracking

---

## 📓 Notebook Template Generator

Generate professional Jupyter notebooks instantly:

| Template | Description |
|----------|-------------|
| 📊 **EDA** | Exploratory Data Analysis with distributions, correlations, visualizations |
| 🧹 **Data Cleaning** | Missing values, duplicates, outliers handling |
| 🎯 **Classification** | 8 algorithms (Logistic Regression, Random Forest, XGBoost, etc.) |
| 📉 **Regression** | 7 algorithms with cross-validation and feature importance |
| 📅 **Time Series** | ARIMA, Exponential Smoothing, trend/seasonality decomposition |
| 🔮 **Clustering** | K-Means, DBSCAN, Hierarchical clustering |
| 📐 **Dimensionality Reduction** | PCA, t-SNE, UMAP |
| 🔧 **Feature Engineering** | Encoding, scaling, feature selection techniques |

**Smart Features:**
- Auto-detects best template based on your data
- Customizable sections - include only what you need
- Educational comments explaining each step

---

## 🗂️ Datasets

DataDojo includes curated datasets across three domains:

### Healthcare 🏥
| Dataset | Description | Rows | Features |
|---------|-------------|------|----------|
| `patient_demographics.csv` | Patient records with demographics | 525 | 34 |
| `lab_results.csv` | Laboratory test results | 1,500 | 22 |
| `ehr_records.csv` | Electronic health records | 50 | 15 |

### Finance 💰
| Dataset | Description | Rows | Features |
|---------|-------------|------|----------|
| `bank_transactions.csv` | Banking transaction records | 1,000 | 12 |
| `credit_applications.csv` | Loan application data | 500 | 18 |
| `stock_prices.csv` | Historical stock data | 43 | 16 |

### E-commerce 🛒
| Dataset | Description | Rows | Features |
|---------|-------------|------|----------|
| `customers_messy.csv` | Customer data with quality issues | 21 | 8 |
| `transactions.csv` | Purchase transactions | 50 | 10 |
| `user_product_interactions.csv` | User behavior data | 61 | 8 |

---

## 📚 Learning Paths

### 🟢 Beginner (Data Cleaning Fundamentals)
- Missing value handling strategies
- Duplicate detection and removal
- Data type validation and conversion
- Basic outlier identification

### 🟡 Intermediate (Feature Engineering)
- Creating derived features
- Categorical encoding techniques
- Scaling and normalization
- Feature selection methods

### 🔴 Advanced (Production Pipelines)
- Automated preprocessing pipelines
- Performance optimization
- Error handling and logging
- Integration with ML workflows

---

## 📂 Project Structure

```
data-dojo-1/
├── app.py                     # Main Streamlit application
├── datasets/                  # Real-world messy datasets
│   ├── healthcare/           # Patient, lab, EHR data
│   ├── finance/              # Transactions, credit, stocks
│   ├── ecommerce/            # Customers, orders, interactions
│   └── uploads/              # User-uploaded datasets
├── src/datadojo/             # Core library
│   ├── cli/                  # Command-line interface
│   │   ├── __main__.py      # CLI entry point
│   │   ├── list_datasets.py # Dataset discovery
│   │   ├── profile_data.py  # Data profiling
│   │   ├── generate_data.py # Synthetic data generation
│   │   ├── explain_concept.py # Concept explanations
│   │   ├── practice.py      # Interactive practice
│   │   └── ...              # Other CLI commands
│   ├── web/                  # Web interface components
│   ├── notebook/             # Notebook template engine
│   ├── utils/                # Profiler, generators, helpers
│   ├── ml/                   # Machine learning utilities
│   └── educational/          # Learning content
├── generated_notebooks/       # Output directory for notebooks
├── examples/                  # Example notebooks and scripts
├── tests/                     # Test suite
├── requirements.txt          # Production dependencies
└── pyproject.toml            # Poetry configuration
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Custom CSS (Dark Theme) |
| **CLI** | Click, Rich (colored output), Prompt Toolkit |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Notebook Generation** | nbformat |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/datadojo

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t datadojo .

# Run the container
docker run -p 8530:8530 datadojo

# Or use docker-compose
docker-compose up
```

---

## ⚙️ Configuration

### Streamlit Config (`.streamlit/config.toml`)
```toml
[server]
headless = true
port = 8530

[theme]
base = "dark"
primaryColor = "#FF9900"
```

### Environment Variables
```bash
DATADOJO_STORAGE_PATH=~/.datadojo     # Base storage path
DATADOJO_EDUCATIONAL_MODE=true         # Enable educational features
DATADOJO_DEBUG=false                   # Debug mode
DATADOJO_LOG_LEVEL=INFO               # Logging level
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- CLI powered by [Click](https://click.palletsprojects.com/) and [Rich](https://rich.readthedocs.io/)
- Visualization powered by [Plotly](https://plotly.com/)
- Machine learning with [Scikit-learn](https://scikit-learn.org/)

---

<p align="center">
  <strong>Made with ❤️ for data science learners</strong>
</p>

<p align="center">
  <a href="https://github.com/Shreyas5848/data-dojo-1">⭐ Star this repo if you find it helpful!</a>
</p>