"""
Generate Data Command for DataDojo CLI
Provides synthetic data generation capabilities through the command line.
"""

import argparse
from pathlib import Path
from typing import Optional

from ..utils.synthetic_data_generator import SyntheticDataGenerator

class CLIResult:
    def __init__(self, success: bool, output: str, exit_code: int, error_message: Optional[str] = None):
        self.success = success
        self.output = output
        self.exit_code = exit_code
        self.error_message = error_message


def generate_data(
    domain: Optional[str] = None,
    size: str = "medium", 
    output_dir: str = "datasets",
    seed: int = 42
) -> CLIResult:
    """
    Generate synthetic datasets for learning purposes.
    
    Args:
        domain: Specific domain to generate (healthcare, ecommerce, finance)
        size: Dataset size (small, medium, large)
        output_dir: Output directory for generated files
        seed: Random seed for reproducibility
    
    Returns:
        CLIResult: Success/failure status with details
    """
    try:
        generator = SyntheticDataGenerator(seed=seed)
        
        # Size configurations
        size_configs = {
            "small": {"patients": 500, "transactions": 2000, "bank_txns": 1000, "credit_apps": 500},
            "medium": {"patients": 1000, "transactions": 5000, "bank_txns": 3000, "credit_apps": 1000},
            "large": {"patients": 2000, "transactions": 10000, "bank_txns": 8000, "credit_apps": 2000}
        }
        
        if size not in size_configs:
            return CLIResult(
                success=False,
                output="",
                exit_code=1,
                error_message=f"Invalid size '{size}'. Choose from: {', '.join(size_configs.keys())}"
            )
        
        config = size_configs[size]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        if domain:
            if domain == "healthcare":
                patients = generator.generate_patient_demographics(config["patients"])
                lab_results = generator.generate_lab_results(config["patients"] * 3, patients['patient_id'].tolist())
                
                healthcare_path = output_path / "healthcare"
                healthcare_path.mkdir(exist_ok=True)
                
                patients_file = healthcare_path / "patient_demographics.csv"
                lab_file = healthcare_path / "lab_results.csv"
                
                patients.to_csv(patients_file, index=False)
                lab_results.to_csv(lab_file, index=False)
                
                generated_files.extend([patients_file, lab_file])
                
            elif domain == "ecommerce":
                customers = generator.generate_customers(config["patients"])
                transactions = generator.generate_transactions(config["transactions"], customers['customer_id'].tolist())
                
                ecommerce_path = output_path / "ecommerce"
                ecommerce_path.mkdir(exist_ok=True)
                
                customers_file = ecommerce_path / "customers_messy.csv"
                transactions_file = ecommerce_path / "transactions.csv"
                
                customers.to_csv(customers_file, index=False)
                transactions.to_csv(transactions_file, index=False)
                
                generated_files.extend([customers_file, transactions_file])
                
            elif domain == "finance":
                bank_txns = generator.generate_bank_transactions(config["bank_txns"])
                credit_apps = generator.generate_credit_applications(config["credit_apps"])
                
                finance_path = output_path / "finance"
                finance_path.mkdir(exist_ok=True)
                
                bank_file = finance_path / "bank_transactions.csv"
                credit_file = finance_path / "credit_applications.csv"
                
                bank_txns.to_csv(bank_file, index=False)
                credit_apps.to_csv(credit_file, index=False)
                
                generated_files.extend([bank_file, credit_file])
                
            else:
                return CLIResult(
                    success=False,
                    output="",
                    exit_code=1,
                    error_message=f"Invalid domain '{domain}'. Choose from: healthcare, ecommerce, finance"
                )
        
        else:
            # Generate all domains
            datasets = generator.generate_all_datasets()
            
            for domain_name, domain_datasets in datasets.items():
                domain_path = output_path / domain_name
                domain_path.mkdir(parents=True, exist_ok=True)
                
                for name, df in domain_datasets.items():
                    filepath = domain_path / f"{name}.csv"
                    df.to_csv(filepath, index=False)
                    generated_files.append(filepath)
        
        # Create summary message
        total_files = len(generated_files)
        total_records = sum(len(pd.read_csv(f)) for f in generated_files)
        
        summary = f"✅ Generated {total_files} datasets with {total_records:,} total records\n"
        summary += f"📁 Output directory: {output_path.absolute()}\n\n"
        summary += "Generated files:\n"
        for file_path in generated_files:
            df_size = len(pd.read_csv(file_path))
            summary += f"  • {file_path.name}: {df_size:,} records\n"
        
        return CLIResult(
            success=True,
            output=summary,
            exit_code=0
        )
        
    except Exception as e:
        return CLIResult(
            success=False,
            output="",
            exit_code=1,
            error_message=f"Failed to generate data: {str(e)}"
        )


# Import pandas for file size calculation
import pandas as pd