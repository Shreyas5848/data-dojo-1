"""E-commerce domain module for DataDojo.

Provides e-commerce specific datasets, operations, and validation rules
for learning data preparation in retail/online shopping contexts.
"""

from typing import List, Dict, Any
from datadojo.models.learning_project import LearningProject, Domain, Difficulty, ValidationRule
from datadojo.models.domain_module import DomainModule, OperationDefinition


def get_ecommerce_module() -> DomainModule:
    """Get the e-commerce domain module configuration.

    Returns:
        DomainModule with e-commerce specific setup
    """
    module = DomainModule(
        domain_name="ecommerce",
        display_name="E-Commerce & Retail Analytics",
        description=(
            "Learn data preparation for e-commerce analytics, customer segmentation, "
            "sales forecasting, and recommendation systems using real-world retail datasets."
        ),
        operations=[],
        validation_rules=[]
    )

    # Add domain-specific operations
    module.operations.append(OperationDefinition(
        name="customer_segmentation_prep",
        operation_type="feature_engineering",
        description="Prepare customer data for segmentation (RFM analysis)",
        parameters_schema={
            "recency_col": "string",
            "frequency_col": "string",
            "monetary_col": "string"
        },
        educational_content=(
            "RFM (Recency, Frequency, Monetary) analysis helps segment customers "
            "based on their purchasing behavior for targeted marketing."
        )
    ))

    module.operations.append(OperationDefinition(
        name="product_category_encoding",
        operation_type="transformation",
        description="Encode product categories for machine learning",
        parameters_schema={
            "category_col": "string",
            "encoding_type": "onehot|label|target"
        },
        educational_content=(
            "Product categories need to be converted to numerical format for "
            "machine learning models while preserving hierarchical relationships."
        )
    ))

    module.operations.append(OperationDefinition(
        name="sales_seasonality_features",
        operation_type="feature_engineering",
        description="Extract seasonal features from sales data",
        parameters_schema={
            "date_col": "string",
            "include_holidays": "boolean"
        },
        educational_content=(
            "Seasonal patterns in sales data can be captured through features "
            "like day of week, month, quarter, and proximity to holidays."
        )
    ))

    # Add validation rules
    module.validation_rules.append({
        "name": "positive_price_check",
        "type": "range",
        "description": "Ensure all prices are positive",
        "column": "price",
        "min_value": 0.01
    })

    module.validation_rules.append({
        "name": "valid_quantity_check",
        "type": "range",
        "description": "Ensure quantities are valid integers",
        "column": "quantity",
        "min_value": 0,
        "max_value": 10000
    })

    module.validation_rules.append({
        "name": "date_order_check",
        "type": "temporal",
        "description": "Ensure order dates are not in the future",
        "column": "order_date",
        "max_date": "today"
    })

    return module


def get_sample_projects() -> List[LearningProject]:
    """Get sample e-commerce learning projects.

    Returns:
        List of pre-configured e-commerce projects
    """
    projects = []

    # Beginner: Customer Data Cleaning
    projects.append(LearningProject(
        id="ecommerce_beginner_01",
        name="Customer Data Cleaning Basics",
        domain=Domain.ECOMMERCE,
        difficulty=Difficulty.BEGINNER,
        description=(
            "Learn fundamental data cleaning with an e-commerce customer dataset. "
            "Handle missing values, remove duplicates, and standardize formats."
        ),
        dataset_path="datasets/ecommerce/customers_messy.csv",
        expected_outcomes=[
            "Handle missing customer contact information",
            "Identify and remove duplicate customer records",
            "Standardize address and phone number formats",
            "Clean and validate email addresses"
        ],
        validation_rules=[
            ValidationRule(
                name="email_format",
                description="Validate email format",
                check_type="pattern",
                parameters={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"}
            ),
            ValidationRule(
                name="no_duplicates",
                description="Ensure no duplicate customer IDs",
                check_type="uniqueness",
                parameters={"column": "customer_id"}
            )
        ]
    ))

    # Intermediate: Sales Analysis Preparation
    projects.append(LearningProject(
        id="ecommerce_intermediate_01",
        name="Sales Data Feature Engineering",
        domain=Domain.ECOMMERCE,
        difficulty=Difficulty.INTERMEDIATE,
        description=(
            "Prepare sales transaction data for analysis. Create features for "
            "customer lifetime value, purchase patterns, and seasonal trends."
        ),
        dataset_path="datasets/ecommerce/transactions.csv",
        expected_outcomes=[
            "Calculate customer lifetime value (CLV)",
            "Create RFM (Recency, Frequency, Monetary) features",
            "Extract seasonal and temporal features",
            "Engineer product affinity features"
        ],
        validation_rules=[
            ValidationRule(
                name="positive_amounts",
                description="All transaction amounts must be positive",
                check_type="range",
                parameters={"column": "amount", "min": 0.01}
            )
        ]
    ))

    # Advanced: Recommendation System Data Prep
    projects.append(LearningProject(
        id="ecommerce_advanced_01",
        name="Product Recommendation Data Pipeline",
        domain=Domain.ECOMMERCE,
        difficulty=Difficulty.ADVANCED,
        description=(
            "Build a complete data preprocessing pipeline for a product recommendation "
            "system. Handle user-item interactions, collaborative filtering features, "
            "and content-based features."
        ),
        dataset_path="datasets/ecommerce/user_product_interactions.csv",
        expected_outcomes=[
            "Create user-item interaction matrix",
            "Engineer collaborative filtering features",
            "Extract product content features",
            "Handle cold-start problem for new users/products",
            "Implement efficient data structures for recommendation"
        ],
        validation_rules=[
            ValidationRule(
                name="rating_range",
                description="Ratings must be between 1 and 5",
                check_type="range",
                parameters={"column": "rating", "min": 1, "max": 5}
            )
        ]
    ))

    return projects


def get_domain_concepts() -> List[Dict[str, Any]]:
    """Get e-commerce specific data science concepts.

    Returns:
        List of educational concepts for e-commerce domain
    """
    return [
        {
            "concept_id": "rfm_analysis",
            "title": "RFM Analysis",
            "explanation": (
                "RFM (Recency, Frequency, Monetary) analysis is a marketing technique "
                "that segments customers based on their purchasing behavior. Recency "
                "measures how recently a customer made a purchase, Frequency counts how "
                "often they buy, and Monetary tracks how much they spend."
            ),
            "difficulty_level": "intermediate",
            "examples": [
                "# Calculate RFM scores\nrecency = (today - df['last_purchase_date']).dt.days\nfrequency = df.groupby('customer_id').size()\nmonetary = df.groupby('customer_id')['amount'].sum()"
            ]
        },
        {
            "concept_id": "customer_lifetime_value",
            "title": "Customer Lifetime Value (CLV)",
            "explanation": (
                "CLV estimates the total revenue a business can expect from a customer "
                "over their entire relationship. It helps prioritize customer acquisition "
                "and retention efforts."
            ),
            "difficulty_level": "intermediate",
            "examples": [
                "# Simple CLV calculation\navg_purchase_value = df.groupby('customer_id')['amount'].mean()\npurchase_frequency = df.groupby('customer_id').size()\nclv = avg_purchase_value * purchase_frequency"
            ]
        },
        {
            "concept_id": "market_basket_analysis",
            "title": "Market Basket Analysis",
            "explanation": (
                "Market basket analysis identifies products that are frequently purchased "
                "together. This helps with product placement, cross-selling, and bundling "
                "strategies."
            ),
            "difficulty_level": "advanced",
            "examples": [
                "# Find frequent item pairs\nfrom mlxtend.frequent_patterns import apriori\nfrequent_items = apriori(basket_data, min_support=0.01)"
            ]
        }
    ]
