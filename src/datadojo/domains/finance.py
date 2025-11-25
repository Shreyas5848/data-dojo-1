"""Finance domain module for DataDojo.

Provides finance-specific datasets, operations, and validation rules
for learning data preparation in banking, trading, and financial analysis contexts.
"""

from typing import List, Dict, Any
from datadojo.models.learning_project import LearningProject, Domain, Difficulty, ValidationRule
from datadojo.models.domain_module import DomainModule, OperationDefinition


def get_finance_module() -> DomainModule:
    """Get the finance domain module configuration.

    Returns:
        DomainModule with finance specific setup
    """
    module = DomainModule(
        domain_name="finance",
        display_name="Finance & Banking Analytics",
        description=(
            "Learn data preparation for financial analysis, risk assessment, fraud "
            "detection, and market predictions using real-world financial datasets."
        ),
        operations=[],
        validation_rules=[]
    )

    # Add domain-specific operations
    module.operations.append(OperationDefinition(
        name="financial_ratios",
        operation_type="feature_engineering",
        description="Calculate financial ratios from balance sheet and income statement",
        parameters_schema={
            "ratio_types": "list",  # e.g., ["ROE", "ROA", "debt_to_equity"]
            "period": "quarterly|annual"
        },
        educational_content=(
            "Financial ratios like ROE (Return on Equity), ROA (Return on Assets), "
            "and debt-to-equity are crucial metrics for assessing company health."
        )
    ))

    module.operations.append(OperationDefinition(
        name="time_series_features",
        operation_type="feature_engineering",
        description="Create lag features, moving averages, and technical indicators",
        parameters_schema={
            "window_sizes": "list",
            "indicators": "list"  # e.g., ["SMA", "EMA", "RSI", "MACD"]
        },
        educational_content=(
            "Time series features like moving averages and RSI (Relative Strength Index) "
            "help identify trends and patterns in financial data."
        )
    ))

    module.operations.append(OperationDefinition(
        name="transaction_anomaly_detection",
        operation_type="validation",
        description="Identify unusual transaction patterns for fraud detection",
        parameters_schema={
            "threshold_method": "statistical|isolation_forest|autoencoder",
            "sensitivity": "float"
        },
        educational_content=(
            "Anomaly detection in financial transactions helps identify fraud, errors, "
            "or unusual patterns that require investigation."
        )
    ))

    module.operations.append(OperationDefinition(
        name="credit_risk_features",
        operation_type="feature_engineering",
        description="Engineer features for credit risk assessment",
        parameters_schema={
            "include_payment_history": "boolean",
            "include_credit_utilization": "boolean"
        },
        educational_content=(
            "Credit risk features include payment history, credit utilization ratio, "
            "length of credit history, and debt-to-income ratio."
        )
    ))

    # Add validation rules
    module.validation_rules.append({
        "name": "positive_amounts",
        "type": "range",
        "description": "Transaction amounts must be positive",
        "column": "amount",
        "min_value": 0.01
    })

    module.validation_rules.append({
        "name": "valid_ticker_format",
        "type": "pattern",
        "description": "Stock ticker symbols must be valid format",
        "column": "ticker",
        "pattern": r"^[A-Z]{1,5}$"
    })

    module.validation_rules.append({
        "name": "account_balance_check",
        "type": "consistency",
        "description": "Closing balance must equal opening balance plus net transactions",
        "columns": ["opening_balance", "transactions", "closing_balance"]
    })

    module.validation_rules.append({
        "name": "date_ordering",
        "type": "temporal",
        "description": "Transaction dates must be in chronological order",
        "column": "transaction_date"
    })

    return module


def get_sample_projects() -> List[LearningProject]:
    """Get sample finance learning projects.

    Returns:
        List of pre-configured finance projects
    """
    projects = []

    # Beginner: Transaction Data Cleaning
    projects.append(LearningProject(
        id="finance_beginner_01",
        name="Banking Transaction Data Cleaning",
        domain=Domain.FINANCE,
        difficulty=Difficulty.BEGINNER,
        description=(
            "Learn to clean and validate banking transaction data. Handle missing values, "
            "detect duplicates, standardize transaction categories, and ensure data quality."
        ),
        dataset_path="datasets/finance/bank_transactions.csv",
        expected_outcomes=[
            "Handle missing transaction descriptions",
            "Identify and remove duplicate transactions",
            "Standardize transaction categories and merchant names",
            "Validate transaction amounts and dates",
            "Ensure account balance consistency"
        ],
        validation_rules=[
            ValidationRule(
                name="no_future_dates",
                description="Transaction dates cannot be in the future",
                check_type="temporal",
                parameters={"column": "transaction_date", "max": "today"}
            ),
            ValidationRule(
                name="valid_amounts",
                description="Transaction amounts must be non-zero",
                check_type="range",
                parameters={"column": "amount", "exclude_zero": True}
            )
        ]
    ))

    # Intermediate: Stock Market Analysis Preparation
    projects.append(LearningProject(
        id="finance_intermediate_01",
        name="Stock Market Data Feature Engineering",
        domain=Domain.FINANCE,
        difficulty=Difficulty.INTERMEDIATE,
        description=(
            "Prepare stock market data for analysis and prediction. Create technical "
            "indicators, handle missing data, and engineer features for trading strategies."
        ),
        dataset_path="datasets/finance/stock_prices.csv",
        expected_outcomes=[
            "Calculate technical indicators (SMA, EMA, RSI, MACD)",
            "Handle missing prices and stock splits",
            "Create volatility and momentum features",
            "Engineer candlestick pattern features",
            "Build time-series features for prediction"
        ],
        validation_rules=[
            ValidationRule(
                name="ohlc_consistency",
                description="Open, High, Low, Close prices must be consistent",
                check_type="consistency",
                parameters={"rule": "Low <= Open,Close <= High"}
            ),
            ValidationRule(
                name="volume_positive",
                description="Trading volume must be non-negative",
                check_type="range",
                parameters={"column": "volume", "min": 0}
            )
        ]
    ))

    # Advanced: Credit Risk Modeling Pipeline
    projects.append(LearningProject(
        id="finance_advanced_01",
        name="Credit Risk Assessment Data Pipeline",
        domain=Domain.FINANCE,
        difficulty=Difficulty.ADVANCED,
        description=(
            "Build a comprehensive preprocessing pipeline for credit risk modeling. "
            "Handle customer financial data, payment histories, and external credit scores "
            "for default prediction."
        ),
        dataset_path="datasets/finance/credit_applications.csv",
        expected_outcomes=[
            "Engineer credit utilization and debt-to-income features",
            "Process payment history and create default indicators",
            "Handle missing credit scores with imputation strategies",
            "Create behavioral scoring features",
            "Implement feature transformations for highly skewed distributions",
            "Build WOE (Weight of Evidence) features for categorical variables"
        ],
        validation_rules=[
            ValidationRule(
                name="credit_score_range",
                description="Credit scores must be between 300 and 850",
                check_type="range",
                parameters={"column": "credit_score", "min": 300, "max": 850}
            ),
            ValidationRule(
                name="income_positive",
                description="Annual income must be positive",
                check_type="range",
                parameters={"column": "annual_income", "min": 0}
            ),
            ValidationRule(
                name="utilization_ratio",
                description="Credit utilization must be between 0 and 1",
                check_type="range",
                parameters={"column": "credit_utilization", "min": 0, "max": 1}
            )
        ]
    ))

    return projects


def get_domain_concepts() -> List[Dict[str, Any]]:
    """Get finance specific data science concepts.

    Returns:
        List of educational concepts for finance domain
    """
    return [
        {
            "concept_id": "technical_indicators",
            "title": "Technical Indicators",
            "explanation": (
                "Technical indicators are mathematical calculations based on price and "
                "volume data. Common indicators include SMA (Simple Moving Average), "
                "EMA (Exponential Moving Average), RSI (Relative Strength Index), and "
                "MACD (Moving Average Convergence Divergence)."
            ),
            "difficulty_level": "intermediate",
            "examples": [
                "# Calculate Simple Moving Average\ndf['SMA_20'] = df['close'].rolling(window=20).mean()\n\n# Calculate RSI\ndelta = df['close'].diff()\ngain = delta.where(delta > 0, 0).rolling(window=14).mean()\nloss = -delta.where(delta < 0, 0).rolling(window=14).mean()\nRS = gain / loss\ndf['RSI'] = 100 - (100 / (1 + RS))"
            ]
        },
        {
            "concept_id": "credit_scoring",
            "title": "Credit Scoring Models",
            "explanation": (
                "Credit scoring predicts the likelihood of loan default. Features include "
                "payment history, credit utilization, length of credit history, types of "
                "credit, and recent credit inquiries. Common models use logistic regression, "
                "scorecards, or machine learning."
            ),
            "difficulty_level": "intermediate",
            "examples": [
                "# Calculate credit utilization\ndf['credit_utilization'] = df['total_balance'] / df['total_credit_limit']\n\n# Create payment history score\ndf['late_payment_score'] = (\n    df['num_late_30'] * 1 + \n    df['num_late_60'] * 2 + \n    df['num_late_90'] * 3\n)"
            ]
        },
        {
            "concept_id": "fraud_detection",
            "title": "Fraud Detection in Financial Transactions",
            "explanation": (
                "Fraud detection identifies suspicious transactions using anomaly detection "
                "and pattern recognition. Features include transaction amount, frequency, "
                "location, time patterns, and deviations from normal behavior."
            ),
            "difficulty_level": "advanced",
            "examples": [
                "# Detect unusual transaction amounts\ndf['z_score'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()\ndf['is_anomaly'] = df['z_score'].abs() > 3\n\n# Time-based features\ndf['hour'] = pd.to_datetime(df['timestamp']).dt.hour\ndf['is_unusual_time'] = df['hour'].isin([0, 1, 2, 3, 4, 5])"
            ]
        },
        {
            "concept_id": "portfolio_optimization",
            "title": "Portfolio Risk and Return Analysis",
            "explanation": (
                "Portfolio optimization balances risk and return. Key concepts include "
                "expected return, volatility (standard deviation), Sharpe ratio, and "
                "correlation between assets. Modern Portfolio Theory uses these to find "
                "optimal asset allocation."
            ),
            "difficulty_level": "advanced",
            "examples": [
                "# Calculate returns and volatility\nreturns = df.pct_change()\nexpected_return = returns.mean() * 252  # Annualized\nvolatility = returns.std() * np.sqrt(252)  # Annualized\n\n# Sharpe ratio (assuming 2% risk-free rate)\nsharpe_ratio = (expected_return - 0.02) / volatility"
            ]
        }
    ]
