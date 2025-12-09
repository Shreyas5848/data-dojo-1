"""
Advanced Synthetic Data Generator for DataDojo
Generates realistic, large-scale datasets for all domains with proper data relationships and quality issues.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import string
from pathlib import Path


class SyntheticDataGenerator:
    """Generate realistic synthetic datasets for learning purposes."""
    
    def __init__(self, seed: int = 42, error_rate: float = 0.15, duplicate_rate: float = 0.05, missing_rate: float = 0.10):
        """
        Initialize the synthetic data generator.
        
        Args:
            seed: Random seed for reproducibility
            error_rate: Percentage of records with data quality issues (0.0-1.0)
            duplicate_rate: Percentage of duplicate records to add (0.0-1.0)
            missing_rate: Percentage of values to make missing (0.0-1.0)
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Store messiness parameters
        self.error_rate = error_rate
        self.duplicate_rate = duplicate_rate
        self.missing_rate = missing_rate
        
        # Common data pools
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
            "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
            "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
            "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
            "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
            "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
            "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
            "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
            "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
        ]
        
        self.cities = [
            ("New York", "NY", "10001"), ("Los Angeles", "CA", "90001"),
            ("Chicago", "IL", "60601"), ("Houston", "TX", "77001"),
            ("Phoenix", "AZ", "85001"), ("Philadelphia", "PA", "19101"),
            ("San Antonio", "TX", "78201"), ("San Diego", "CA", "92101"),
            ("Dallas", "TX", "75201"), ("San Jose", "CA", "95101"),
            ("Austin", "TX", "78701"), ("Jacksonville", "FL", "32201"),
            ("Boston", "MA", "02101"), ("Seattle", "WA", "98101"),
            ("Denver", "CO", "80201"), ("Atlanta", "GA", "30301"),
            ("Miami", "FL", "33101"), ("Detroit", "MI", "48201")
        ]
        
    def _generate_email(self, first_name: str, last_name: str) -> str:
        """Generate realistic email address."""
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"]
        formats = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name[0].lower()}{last_name.lower()}",
            f"{first_name.lower()}{random.randint(1, 99)}"
        ]
        return f"{random.choice(formats)}@{random.choice(domains)}"
    
    def _generate_phone(self) -> str:
        """Generate US phone number."""
        area_code = random.randint(200, 999)
        exchange = random.randint(200, 999)
        subscriber = random.randint(1000, 9999)
        return f"+1-{area_code}-{exchange}-{subscriber}"
    
    def _generate_date_range(self, start_date: datetime, end_date: datetime) -> datetime:
        """Generate random date within range."""
        delta = end_date - start_date
        random_days = random.randint(0, delta.days)
        return start_date + timedelta(days=random_days)
    
    # ==================== HEALTHCARE DOMAIN ====================
    
    def generate_patient_demographics(self, num_patients: int = 1000) -> pd.DataFrame:
        """Generate realistic patient demographics data with quality issues."""
        
        blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        blood_type_weights = [0.34, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.07]
        
        races = ["White", "Black or African American", "Asian", "Hispanic or Latino", 
                 "American Indian", "Pacific Islander", "Two or More Races"]
        race_weights = [0.60, 0.13, 0.06, 0.18, 0.01, 0.01, 0.01]
        
        marital_statuses = ["Single", "Married", "Divorced", "Widowed", "Separated"]
        marital_weights = [0.30, 0.48, 0.12, 0.07, 0.03]
        
        insurance_providers = [
            "Blue Cross Blue Shield", "UnitedHealthcare", "Aetna", "Cigna", 
            "Humana", "Kaiser Permanente", "Medicare", "Medicaid", "Self-Pay"
        ]
        
        smoking_statuses = ["Never", "Former", "Current", "Unknown"]
        smoking_weights = [0.55, 0.22, 0.14, 0.09]
        
        physicians = [
            "Dr. Emily Chen", "Dr. Michael Roberts", "Dr. Sarah Williams",
            "Dr. James Wilson", "Dr. Lisa Anderson", "Dr. Robert Kim",
            "Dr. Jennifer Park", "Dr. David Lee", "Dr. Michelle Torres"
        ]
        
        chronic_conditions_pool = [
            "Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Coronary Artery Disease",
            "COPD", "Asthma", "Chronic Kidney Disease", "Atrial Fibrillation",
            "Heart Failure", "Osteoarthritis", "Depression", "Anxiety",
            "Hypothyroidism", "GERD", "Obesity", "Sleep Apnea", "Stroke History"
        ]
        
        allergies_pool = [
            "Penicillin", "Sulfa drugs", "Aspirin", "NSAIDs", "Codeine",
            "Latex", "Iodine contrast", "Shellfish", "Peanuts", "Eggs",
            "No Known Allergies"
        ]
        
        patients = []
        
        for i in range(1, num_patients + 1):
            gender = random.choice(["Male", "Female", "Other"])
            
            # Age distribution (weighted toward middle-aged and elderly for healthcare)
            age = int(np.random.choice(
                range(18, 95),
                p=self._age_distribution_healthcare()
            ))
            
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            
            dob = datetime.now() - timedelta(days=age*365 + random.randint(0, 364))
            city, state, zip_base = random.choice(self.cities)
            
            # Generate chronic conditions (more likely with age)
            num_conditions = min(int(np.random.exponential(age/30)), 6)
            conditions = random.sample(chronic_conditions_pool, min(num_conditions, len(chronic_conditions_pool)))
            
            # Generate allergies
            num_allergies = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.2, 0.05])
            allergies = random.sample(allergies_pool, num_allergies) if num_allergies > 0 else ["No Known Allergies"]
            
            # BMI calculation (correlated with conditions)
            if "Obesity" in conditions:
                bmi = round(random.uniform(30, 45), 1)
            elif "Type 2 Diabetes" in conditions:
                bmi = round(random.uniform(26, 38), 1)
            else:
                bmi = round(random.uniform(18.5, 32), 1)
            
            # Height and weight based on gender and BMI
            if gender == "Male":
                height = random.randint(64, 76)
            else:
                height = random.randint(58, 70)
            weight = round(bmi * (height ** 2) / 703, 1)
            
            # Registration and visit dates
            reg_date = self._generate_date_range(
                datetime.now() - timedelta(days=3650),
                datetime.now() - timedelta(days=30)
            )
            last_visit = self._generate_date_range(reg_date, datetime.now())
            
            patient = {
                "patient_id": f"P{i:06d}",
                "mrn": f"MRN{10000 + i}",
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": dob.strftime("%Y-%m-%d"),
                "age": age,
                "gender": gender,
                "race": np.random.choice(races, p=race_weights),
                "ethnicity": random.choice(["Hispanic or Latino", "Not Hispanic or Latino"]),
                "marital_status": np.random.choice(marital_statuses, p=marital_weights),
                "language": random.choice(["English", "Spanish", "Mandarin", "Vietnamese", "Korean"]),
                "ssn_last_four": f"{random.randint(1000, 9999)}",
                "address": f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Park'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr'])}",
                "city": city,
                "state": state,
                "zip_code": f"{zip_base[:3]}{random.randint(10, 99)}",
                "phone": self._generate_phone(),
                "email": self._generate_email(first_name, last_name) if random.random() > 0.10 else "",
                "emergency_contact_name": f"{random.choice(self.first_names)} {last_name}",
                "emergency_contact_phone": self._generate_phone(),
                "emergency_contact_relationship": random.choice(["Spouse", "Parent", "Child", "Sibling", "Friend"]),
                "insurance_provider": random.choice(insurance_providers),
                "insurance_id": f"INS{random.randint(100000000, 999999999)}",
                "primary_care_physician": random.choice(physicians),
                "registration_date": reg_date.strftime("%Y-%m-%d"),
                "last_visit_date": last_visit.strftime("%Y-%m-%d"),
                "blood_type": np.random.choice(blood_types, p=blood_type_weights),
                "height_inches": height,
                "weight_lbs": weight,
                "bmi": bmi,
                "smoking_status": np.random.choice(smoking_statuses, p=smoking_weights),
                "alcohol_use": random.choice(["None", "Social", "Moderate", "Heavy"]),
                "allergies": ";".join(allergies),
                "chronic_conditions": ";".join(conditions) if conditions else "None"
            }
            
            # Introduce data quality issues based on error_rate
            if random.random() < self.error_rate:
                issue_type = random.choice([
                    "missing_email", "missing_phone", "invalid_age", 
                    "duplicate_prone", "inconsistent_case", "whitespace",
                    "missing_bmi", "missing_insurance", "typo_state",
                    "outlier_bmi", "invalid_date", "mixed_formats"
                ])
                
                if issue_type == "missing_email":
                    patient["email"] = "" if random.random() < 0.7 else np.nan
                elif issue_type == "missing_phone":
                    patient["phone"] = "" if random.random() < 0.5 else np.nan
                elif issue_type == "invalid_age":
                    patient["age"] = random.choice([-1, 0, 150, 999, None])
                elif issue_type == "missing_bmi":
                    patient["bmi"] = np.nan
                elif issue_type == "missing_insurance":
                    patient["insurance_provider"] = np.nan
                    patient["insurance_id"] = ""
                elif issue_type == "typo_state":
                    patient["state"] = patient["state"][0] if len(patient["state"]) > 1 else patient["state"] + "X"
                elif issue_type == "outlier_bmi":
                    patient["bmi"] = random.choice([5.5, 70.2, 99.9])
                elif issue_type == "invalid_date":
                    patient["last_visit_date"] = "2099-12-31"  # Future date
                elif issue_type == "mixed_formats":
                    patient["phone"] = patient["phone"].replace("+1-", "").replace("-", "") if random.random() < 0.5 else patient["phone"]
                elif issue_type == "inconsistent_case":
                    patient["first_name"] = patient["first_name"].upper() if random.random() < 0.5 else patient["first_name"].lower()
                    patient["city"] = patient["city"].lower() if random.random() < 0.5 else patient["city"].upper()
                elif issue_type == "whitespace":
                    patient["first_name"] = f"  {patient['first_name']}  "
                    patient["last_name"] = f" {patient['last_name']} "
                    patient["city"] = f"\t{patient['city']}\t"
                    
            patients.append(patient)
        
        # Add exact duplicates based on duplicate_rate
        num_duplicates = int(num_patients * self.duplicate_rate)
        for _ in range(num_duplicates):
            dup = random.choice(patients).copy()
            dup["patient_id"] = f"P{len(patients)+1:06d}"
            # Sometimes duplicate with slight variations
            if random.random() < 0.3:
                dup["phone"] = self._generate_phone()  # Same person, different phone
            patients.append(dup)
        
        return pd.DataFrame(patients)
    
    def _age_distribution_healthcare(self) -> List[float]:
        """Generate age distribution weighted toward healthcare demographics."""
        ages = range(18, 95)
        weights = []
        for age in ages:
            if age < 30:
                w = 0.8
            elif age < 45:
                w = 1.2
            elif age < 60:
                w = 1.8
            elif age < 75:
                w = 2.2
            else:
                w = 1.5
            weights.append(w)
        total = sum(weights)
        return [w/total for w in weights]
    
    def generate_lab_results(self, num_results: int = 5000, patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate realistic lab results data."""
        
        if patient_ids is None:
            patient_ids = [f"P{i:06d}" for i in range(1, 1001)]
        
        lab_tests = [
            # (test_code, test_name, category, unit, ref_low, ref_high, typical_mean, typical_std)
            ("CBC-WBC", "White Blood Cell Count", "Hematology", "10^3/uL", 4.5, 11.0, 7.5, 2.0),
            ("CBC-RBC", "Red Blood Cell Count", "Hematology", "10^6/uL", 4.2, 5.4, 4.8, 0.4),
            ("CBC-HGB", "Hemoglobin", "Hematology", "g/dL", 12.0, 17.5, 14.5, 1.5),
            ("CBC-HCT", "Hematocrit", "Hematology", "%", 36.0, 50.0, 43.0, 4.0),
            ("CBC-PLT", "Platelet Count", "Hematology", "10^3/uL", 150.0, 400.0, 250.0, 60.0),
            ("BMP-GLU", "Glucose", "Chemistry", "mg/dL", 70.0, 100.0, 95.0, 25.0),
            ("BMP-BUN", "Blood Urea Nitrogen", "Chemistry", "mg/dL", 7.0, 20.0, 15.0, 5.0),
            ("BMP-CR", "Creatinine", "Chemistry", "mg/dL", 0.7, 1.3, 1.0, 0.3),
            ("BMP-NA", "Sodium", "Chemistry", "mEq/L", 136.0, 145.0, 140.0, 3.0),
            ("BMP-K", "Potassium", "Chemistry", "mEq/L", 3.5, 5.0, 4.2, 0.4),
            ("LIP-CHOL", "Total Cholesterol", "Lipid Panel", "mg/dL", 0.0, 200.0, 195.0, 40.0),
            ("LIP-TG", "Triglycerides", "Lipid Panel", "mg/dL", 0.0, 150.0, 130.0, 60.0),
            ("LIP-HDL", "HDL Cholesterol", "Lipid Panel", "mg/dL", 40.0, 60.0, 50.0, 15.0),
            ("LIP-LDL", "LDL Cholesterol", "Lipid Panel", "mg/dL", 0.0, 100.0, 110.0, 35.0),
            ("LFT-ALT", "Alanine Aminotransferase", "Liver Function", "U/L", 7.0, 56.0, 30.0, 15.0),
            ("LFT-AST", "Aspartate Aminotransferase", "Liver Function", "U/L", 10.0, 40.0, 25.0, 12.0),
            ("TSH", "Thyroid Stimulating Hormone", "Thyroid", "mIU/L", 0.4, 4.0, 2.0, 1.0),
            ("HBA1C", "Hemoglobin A1c", "Diabetes", "%", 4.0, 5.6, 5.8, 1.2),
            ("VITD", "Vitamin D 25-Hydroxy", "Nutrition", "ng/mL", 30.0, 100.0, 35.0, 15.0)
        ]
        
        physicians = [
            "Dr. Emily Chen", "Dr. Michael Roberts", "Dr. Sarah Williams",
            "Dr. James Wilson", "Dr. Lisa Anderson", "Dr. Robert Kim"
        ]
        
        labs = ["Quest Diagnostics", "LabCorp", "Hospital Lab", "Regional Medical Lab"]
        specimen_types = ["Blood", "Serum", "Plasma", "Urine"]
        
        results = []
        
        for i in range(1, num_results + 1):
            patient_id = random.choice(patient_ids)
            test = random.choice(lab_tests)
            test_code, test_name, category, unit, ref_low, ref_high, mean, std = test
            
            # Generate value with some abnormal results (25% abnormal)
            if random.random() < 0.25:
                if random.random() < 0.5:
                    value = round(ref_low - abs(np.random.normal(0, std)), 2)
                else:
                    value = round(ref_high + abs(np.random.normal(0, std)), 2)
            else:
                value = round(np.random.normal(mean, std), 2)
            
            # Determine flags
            if value < ref_low:
                flag = "Low"
                critical = "Yes" if value < ref_low * 0.7 else "No"
            elif value > ref_high:
                flag = "High"
                critical = "Yes" if value > ref_high * 1.3 else "No"
            else:
                flag = "Normal"
                critical = "No"
            
            test_date = self._generate_date_range(
                datetime.now() - timedelta(days=730),
                datetime.now()
            )
            
            collection_time = f"{random.randint(6, 18):02d}:{random.randint(0, 59):02d}:00"
            received_time = f"{int(collection_time[:2]) + random.randint(0, 2):02d}:{random.randint(0, 59):02d}:00"
            result_time = f"{int(received_time[:2]) + random.randint(1, 4):02d}:{random.randint(0, 59):02d}:00"
            
            result = {
                "result_id": f"LR{i:06d}",
                "patient_id": patient_id,
                "order_id": f"ORD{10000 + i}",
                "ordering_physician": random.choice(physicians),
                "test_date": test_date.strftime("%Y-%m-%d"),
                "collection_time": collection_time,
                "received_time": received_time,
                "result_time": result_time,
                "test_code": test_code,
                "test_name": test_name,
                "test_category": category,
                "value": value,
                "unit": unit,
                "reference_range_low": ref_low,
                "reference_range_high": ref_high,
                "flag": flag,
                "critical_flag": critical,
                "specimen_type": random.choice(specimen_types),
                "fasting_status": random.choice(["Yes", "No", "Unknown"]),
                "lab_location": random.choice(labs),
                "performing_lab": random.choice(labs),
                "notes": "" if random.random() > 0.1 else random.choice([
                    "Sample hemolyzed", "Repeat recommended", "Patient fasting confirmed",
                    "Lipemic sample", "Quantity not sufficient"
                ])
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    # ==================== E-COMMERCE DOMAIN ====================
    
    def generate_customers(self, num_customers: int = 2000) -> pd.DataFrame:
        """Generate customer data with intentional quality issues for learning."""
        
        customers = []
        
        for i in range(1, num_customers + 1):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            city, state, zip_code = random.choice(self.cities)
            
            age = random.randint(18, 75)
            reg_date = self._generate_date_range(
                datetime.now() - timedelta(days=1825),
                datetime.now()
            )
            
            # Calculate realistic spending based on age and demographics
            base_spending = random.uniform(100, 5000)
            if age > 40:
                base_spending *= 1.5
            if age > 60:
                base_spending *= 0.8
            
            customer = {
                "customer_id": i,
                "first_name": first_name,
                "last_name": last_name,
                "email": self._generate_email(first_name, last_name),
                "age": age,
                "gender": random.choice(["Male", "Female", "Non-binary", "Prefer not to say"]),
                "city": city,
                "state": state,
                "zip_code": zip_code,
                "registration_date": reg_date.strftime("%Y-%m-%d"),
                "phone": self._generate_phone(),
                "loyalty_tier": random.choices(
                    ["Bronze", "Silver", "Gold", "Platinum"],
                    weights=[0.4, 0.3, 0.2, 0.1]
                )[0],
                "total_purchases": random.randint(0, 100),
                "total_spent": round(base_spending, 2),
                "preferred_category": random.choice([
                    "Electronics", "Clothing", "Home & Garden", "Sports", 
                    "Books", "Beauty", "Automotive", "Health"
                ]),
                "account_status": random.choices(
                    ["Active", "Inactive", "Suspended"],
                    weights=[0.85, 0.12, 0.03]
                )[0]
            }
            
            # Introduce data quality issues based on error_rate
            if random.random() < self.error_rate:
                issue = random.choice([
                    "missing_email", "missing_age", "invalid_phone", 
                    "duplicate_name", "inconsistent_case", "whitespace",
                    "invalid_zip", "future_date", "negative_spending",
                    "invalid_tier", "special_chars", "missing_state"
                ])
                
                if issue == "missing_email":
                    customer["email"] = random.choice(["", None, "N/A", "null"])
                elif issue == "missing_age":
                    customer["age"] = random.choice([None, 0, -1])
                elif issue == "missing_state":
                    customer["state"] = random.choice(["", None, "XX"])
                elif issue == "invalid_phone":
                    customer["phone"] = random.choice(["N/A", "unknown", "555-FAKE", "", "123456", "(000) 000-0000"])
                elif issue == "negative_spending":
                    customer["total_spent"] = -abs(customer["total_spent"])
                elif issue == "invalid_tier":
                    customer["loyalty_tier"] = random.choice(["", None, "Unknown", "diamond", "BRONZE"])  # Mixed case
                elif issue == "special_chars":
                    customer["first_name"] = customer["first_name"] + "@#"
                    customer["last_name"] = "O'" + customer["last_name"]  # Common name issue
                elif issue == "inconsistent_case":
                    customer["first_name"] = customer["first_name"].upper() if random.random() < 0.5 else customer["first_name"].lower()
                    customer["city"] = customer["city"].lower() if random.random() < 0.5 else customer["city"].upper()
                elif issue == "whitespace":
                    customer["first_name"] = f"  {customer['first_name']}  "
                    customer["city"] = f" {customer['city']} "
                    customer["email"] = f" {customer['email']} "
                elif issue == "invalid_zip":
                    customer["zip_code"] = random.choice(["00000", "99999", "ABCDE", ""])
                elif issue == "future_date":
                    customer["registration_date"] = (datetime.now() + timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
            
            customers.append(customer)
        
        # Add exact duplicates based on duplicate_rate
        num_duplicates = int(num_customers * self.duplicate_rate)
        for _ in range(num_duplicates):
            dup = random.choice(customers).copy()
            dup["customer_id"] = len(customers) + 1
            # Sometimes with slight variations
            if random.random() < 0.4:
                dup["email"] = self._generate_email(dup["first_name"], dup["last_name"])
            customers.append(dup)
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, num_transactions: int = 10000, customer_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """Generate e-commerce transaction data."""
        
        if customer_ids is None:
            customer_ids = list(range(1, 2001))
        
        products = [
            (101, "Wireless Bluetooth Headphones", "Electronics", 79.99),
            (102, "Organic Green Tea (50 bags)", "Groceries", 12.49),
            (103, "Running Shoes - Nike Air", "Sports", 129.99),
            (104, "Python Programming Book", "Books", 45.00),
            (105, "Stainless Steel Water Bottle", "Home & Kitchen", 24.99),
            (106, "Yoga Mat Premium", "Sports", 35.99),
            (107, "LED Desk Lamp", "Electronics", 42.99),
            (108, "Coffee Maker 12-Cup", "Home & Kitchen", 89.99),
            (109, "Wireless Mouse", "Electronics", 29.99),
            (110, "Fitness Tracker Watch", "Electronics", 149.99),
            (111, "Organic Protein Powder", "Health", 54.99),
            (112, "Mechanical Keyboard", "Electronics", 119.99),
            (113, "Air Purifier HEPA", "Home & Kitchen", 199.99),
            (114, "Resistance Bands Set", "Sports", 19.99),
            (115, "Noise Cancelling Earbuds", "Electronics", 199.99),
            (116, "Smart Home Speaker", "Electronics", 99.99),
            (117, "Portable Charger 20000mAh", "Electronics", 39.99),
            (118, "Hiking Backpack 40L", "Sports", 79.99),
            (119, "Instant Pot 6-Quart", "Home & Kitchen", 89.99),
            (120, "Data Science Handbook", "Books", 59.99)
        ]
        
        payment_methods = ["Credit Card", "Debit Card", "PayPal", "Apple Pay", "Google Pay"]
        statuses = ["Completed", "Completed", "Completed", "Completed", "Pending", "Refunded", "Cancelled"]
        
        transactions = []
        
        for i in range(1, num_transactions + 1):
            product = random.choice(products)
            product_id, product_name, category, base_price = product
            
            # Apply seasonal and random discounts
            discount = random.choices([0, 0.1, 0.15, 0.2, 0.25], weights=[0.6, 0.2, 0.1, 0.07, 0.03])[0]
            price = round(base_price * (1 - discount), 2)
            quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
            
            trans_date = self._generate_date_range(
                datetime.now() - timedelta(days=730),
                datetime.now()
            )
            
            transaction = {
                "transaction_id": f"TXN{i:06d}",
                "customer_id": random.choice(customer_ids),
                "product_id": product_id,
                "product_name": product_name,
                "category": category,
                "unit_price": price,
                "quantity": quantity,
                "total_amount": round(price * quantity, 2),
                "discount_percent": int(discount * 100),
                "transaction_date": trans_date.strftime("%Y-%m-%d %H:%M:%S"),
                "payment_method": random.choice(payment_methods),
                "status": random.choice(statuses),
                "shipping_address_state": random.choice(self.cities)[1],
                "device": random.choices(["Desktop", "Mobile", "Tablet"], weights=[0.4, 0.5, 0.1])[0],
                "session_id": f"SES{random.randint(100000, 999999)}",
                "revenue": round(price * quantity * 0.3, 2),  # 30% margin
                "shipping_cost": round(random.uniform(0, 15), 2)
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    # ==================== FINANCE DOMAIN ====================
    
    def generate_bank_transactions(self, num_transactions: int = 8000) -> pd.DataFrame:
        """Generate realistic bank transaction data."""
        
        categories = {
            "credit": ["Salary", "Bonus", "Refund", "Transfer In", "Interest", "Dividend", "Freelance"],
            "debit": ["Groceries", "Utilities", "Rent/Mortgage", "Entertainment", "Dining", 
                     "Transportation", "Healthcare", "Shopping", "Insurance", "Subscription",
                     "ATM Withdrawal", "Online Purchase", "Gas Station", "Pharmacy"]
        }
        
        merchants = {
            "Groceries": ["Whole Foods", "Trader Joe's", "Walmart", "Costco", "Kroger", "Safeway"],
            "Utilities": ["Con Edison", "National Grid", "Spectrum", "Verizon", "AT&T"],
            "Entertainment": ["Netflix", "Spotify", "AMC Theatres", "Steam", "PlayStation Store"],
            "Dining": ["Starbucks", "Chipotle", "McDonald's", "Uber Eats", "DoorDash"],
            "Transportation": ["Uber", "Lyft", "Shell", "ExxonMobil", "MTA", "Airport Parking"],
            "Shopping": ["Amazon", "Target", "Best Buy", "Nordstrom", "Macy's", "Apple Store"],
            "Healthcare": ["CVS Pharmacy", "Walgreens", "Hospital Copay", "Quest Diagnostics"]
        }
        
        # Generate account holders
        num_accounts = 500
        accounts = []
        for i in range(1, num_accounts + 1):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            accounts.append({
                "account_id": f"ACC{1000 + i}",
                "customer_name": f"{first_name} {last_name}",
                "account_type": random.choice(["Checking", "Savings", "Business"]),
                "opening_balance": round(random.uniform(1000, 50000), 2)
            })
        
        transactions = []
        
        for i in range(1, num_transactions + 1):
            account = random.choice(accounts)
            
            # Determine transaction type (more debits than credits)
            if random.random() < 0.20:  # 20% credits
                trans_type = "credit"
                category = random.choice(categories["credit"])
                if category == "Salary":
                    amount = round(random.uniform(2000, 8000), 2)
                elif category == "Bonus":
                    amount = round(random.uniform(500, 5000), 2)
                elif category == "Freelance":
                    amount = round(random.uniform(300, 2000), 2)
                else:
                    amount = round(random.uniform(10, 500), 2)
                merchant = category
            else:
                trans_type = "debit"
                category = random.choice(categories["debit"])
                
                if category == "Rent/Mortgage":
                    amount = round(random.uniform(1000, 3500), 2)
                elif category == "Utilities":
                    amount = round(random.uniform(50, 300), 2)
                elif category == "Groceries":
                    amount = round(random.uniform(30, 250), 2)
                elif category == "Entertainment":
                    amount = round(random.uniform(10, 150), 2)
                elif category == "ATM Withdrawal":
                    amount = round(random.choice([20, 40, 60, 80, 100, 120, 200]), 2)
                else:
                    amount = round(random.uniform(5, 500), 2)
                
                amount = -amount  # Negative for debits
                merchant = random.choice(merchants.get(category, [category]))
            
            trans_date = self._generate_date_range(
                datetime.now() - timedelta(days=365),
                datetime.now()
            )
            
            # Calculate balance (simplified)
            balance_change = random.uniform(-500, 1000)  # Simulate account balance changes
            
            transaction = {
                "transaction_id": f"TXN{i:06d}",
                "account_id": account["account_id"],
                "customer_name": account["customer_name"],
                "account_type": account["account_type"],
                "amount": amount,
                "balance_after": round(account["opening_balance"] + balance_change, 2),
                "transaction_type": trans_type,
                "category": category,
                "transaction_date": trans_date.strftime("%Y-%m-%d %H:%M:%S"),
                "description": f"{category} - {merchant}",
                "merchant": merchant,
                "location": f"{random.choice(self.cities)[0]}, {random.choice(self.cities)[1]}",
                "is_recurring": category in ["Salary", "Rent/Mortgage", "Utilities", "Subscription"],
                "flagged_suspicious": random.random() < 0.015,  # 1.5% flagged
                "channel": random.choices(["ATM", "Online", "Mobile", "Branch", "Phone"], weights=[0.25, 0.35, 0.25, 0.1, 0.05])[0]
            }
            
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def generate_credit_applications(self, num_applications: int = 1500) -> pd.DataFrame:
        """Generate realistic credit application data for ML classification."""
        
        education_levels = ["High School", "Some College", "Bachelor's", "Master's", "Doctorate"]
        employment_types = ["Full-time", "Part-time", "Self-employed", "Unemployed", "Retired", "Student"]
        loan_purposes = ["Home Purchase", "Auto Loan", "Debt Consolidation", "Home Improvement", 
                        "Business", "Education", "Medical", "Personal", "Vacation"]
        
        applications = []
        
        for i in range(1, num_applications + 1):
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            
            age = random.randint(18, 75)
            employment = random.choices(employment_types, weights=[0.6, 0.1, 0.15, 0.05, 0.08, 0.02])[0]
            
            # Income based on employment and age
            if employment == "Full-time":
                if age < 25:
                    income = random.uniform(25000, 50000)
                elif age < 35:
                    income = random.uniform(40000, 80000)
                elif age < 50:
                    income = random.uniform(60000, 150000)
                else:
                    income = random.uniform(50000, 120000)
            elif employment == "Part-time":
                income = random.uniform(15000, 35000)
            elif employment == "Self-employed":
                income = random.uniform(30000, 200000)  # High variance
            elif employment == "Retired":
                income = random.uniform(20000, 60000)
            elif employment == "Student":
                income = random.uniform(0, 25000)
            else:  # Unemployed
                income = random.uniform(0, 15000)
            
            income = round(income, 2)
            
            # Credit score correlated with income, age, and employment
            base_score = 580
            if income > 100000:
                base_score += 80
            elif income > 50000:
                base_score += 40
            elif income > 30000:
                base_score += 20
            
            if employment == "Full-time":
                base_score += 40
            elif employment == "Self-employed":
                base_score += 20
            elif employment == "Unemployed":
                base_score -= 60
            
            if age > 30:
                base_score += 20
            if age > 50:
                base_score += 10
            
            credit_score = min(850, max(300, int(np.random.normal(base_score, 60))))
            
            # Loan amount and term
            purpose = random.choice(loan_purposes)
            if purpose == "Home Purchase":
                loan_amount = round(random.uniform(100000, 800000), 2)
                term_months = random.choice([180, 240, 300, 360])
            elif purpose == "Auto Loan":
                loan_amount = round(random.uniform(15000, 80000), 2)
                term_months = random.choice([36, 48, 60, 72])
            elif purpose in ["Business", "Education"]:
                loan_amount = round(random.uniform(25000, 200000), 2)
                term_months = random.choice([60, 84, 120, 180])
            else:
                loan_amount = round(random.uniform(5000, 50000), 2)
                term_months = random.choice([12, 24, 36, 48, 60])
            
            # Existing debt and DTI
            existing_debt = round(random.uniform(0, income * 0.5), 2)
            monthly_income = income / 12
            monthly_debt_payment = existing_debt / 36  # Assume 3-year average payoff
            new_monthly_payment = loan_amount / term_months
            total_monthly_debt = monthly_debt_payment + new_monthly_payment
            dti = round((total_monthly_debt / monthly_income) * 100, 2) if monthly_income > 0 else 999
            
            # Approval logic (more sophisticated)
            approval_score = 0
            if credit_score >= 750:
                approval_score += 40
            elif credit_score >= 700:
                approval_score += 30
            elif credit_score >= 650:
                approval_score += 20
            elif credit_score >= 600:
                approval_score += 10
            
            if dti < 36:
                approval_score += 25
            elif dti < 43:
                approval_score += 15
            elif dti < 50:
                approval_score += 5
            
            if employment in ["Full-time", "Self-employed"]:
                approval_score += 15
            elif employment == "Part-time":
                approval_score += 5
            
            if income > 50000:
                approval_score += 10
            
            # Random factors
            approval_score += random.randint(-10, 10)
            
            approved = approval_score >= 50 and employment != "Unemployed"
            
            # Interest rate based on credit score and approval
            if approved:
                if credit_score >= 750:
                    interest_rate = round(random.uniform(3.5, 6.0), 2)
                elif credit_score >= 700:
                    interest_rate = round(random.uniform(6.0, 9.0), 2)
                elif credit_score >= 650:
                    interest_rate = round(random.uniform(9.0, 14.0), 2)
                else:
                    interest_rate = round(random.uniform(14.0, 22.0), 2)
            else:
                interest_rate = None
            
            city, state, _ = random.choice(self.cities)
            
            application = {
                "application_id": f"APP{i:05d}",
                "customer_id": f"CUST{10000 + i}",
                "application_date": self._generate_date_range(
                    datetime.now() - timedelta(days=365),
                    datetime.now()
                ).strftime("%Y-%m-%d"),
                "first_name": first_name,
                "last_name": last_name,
                "age": age,
                "gender": random.choice(["Male", "Female", "Other"]),
                "marital_status": random.choices(
                    ["Single", "Married", "Divorced", "Widowed"],
                    weights=[0.4, 0.45, 0.1, 0.05]
                )[0],
                "dependents": max(0, int(np.random.poisson(1.2))),
                "education": random.choices(education_levels, weights=[0.2, 0.25, 0.35, 0.15, 0.05])[0],
                "employment_status": employment,
                "employer": f"{random.choice(['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Government'])} Corp" if employment in ["Full-time", "Part-time"] else "",
                "years_employed": max(0, random.randint(0, min(30, age-18))) if employment not in ["Unemployed", "Student"] else 0,
                "annual_income": income,
                "monthly_expenses": round(income * random.uniform(0.4, 0.8) / 12, 2),
                "existing_debt": existing_debt,
                "credit_score": credit_score,
                "credit_history_years": max(0, min(age - 18, random.randint(1, 25))),
                "num_credit_accounts": max(1, int(np.random.poisson(4))),
                "num_late_payments": max(0, int(np.random.poisson(2)) if credit_score < 650 else int(np.random.poisson(0.5))),
                "bankruptcies": 1 if credit_score < 500 and random.random() < 0.3 else 0,
                "loan_purpose": purpose,
                "loan_amount_requested": loan_amount,
                "loan_term_months": term_months,
                "property_ownership": random.choices(["Own", "Rent", "Other"], weights=[0.4, 0.55, 0.05])[0],
                "state": state,
                "debt_to_income_ratio": dti,
                "approved": approved,
                "approval_score": approval_score,
                "interest_rate_offered": interest_rate,
                "rejection_reason": None if approved else random.choice([
                    "Low credit score", "High DTI ratio", "Insufficient income", 
                    "Employment status", "Incomplete application", "Too many recent inquiries"
                ])
            }
            
            applications.append(application)
        
        return pd.DataFrame(applications)
    
    # ==================== UTILITY METHODS ====================
    
    def generate_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Generate all datasets for all domains."""
        
        print("🏥 Generating Healthcare datasets...")
        patient_demographics = self.generate_patient_demographics(1000)
        patient_ids = patient_demographics['patient_id'].tolist()
        lab_results = self.generate_lab_results(5000, patient_ids)
        
        print("🛒 Generating E-commerce datasets...")
        customers = self.generate_customers(2000)
        customer_ids = customers['customer_id'].tolist()
        transactions = self.generate_transactions(10000, customer_ids)
        
        print("💰 Generating Finance datasets...")
        bank_transactions = self.generate_bank_transactions(8000)
        credit_applications = self.generate_credit_applications(1500)
        
        return {
            "healthcare": {
                "patient_demographics": patient_demographics,
                "lab_results": lab_results
            },
            "ecommerce": {
                "customers_messy": customers,
                "transactions": transactions
            },
            "finance": {
                "bank_transactions": bank_transactions,
                "credit_applications": credit_applications
            }
        }
    
    def save_datasets(self, output_dir: str = "datasets"):
        """Generate and save all datasets to CSV files."""
        datasets = self.generate_all_datasets()
        
        for domain, domain_datasets in datasets.items():
            domain_path = Path(output_dir) / domain
            domain_path.mkdir(parents=True, exist_ok=True)
            
            for name, df in domain_datasets.items():
                filepath = domain_path / f"{name}.csv"
                df.to_csv(filepath, index=False)
                print(f"✅ Saved {filepath} ({len(df):,} records)")
        
        print(f"\n🎉 All datasets generated successfully!")
        print(f"📁 Total datasets: {sum(len(d) for d in datasets.values())}")
        print(f"📊 Total records: {sum(len(df) for domain in datasets.values() for df in domain.values()):,}")


# CLI entry point
if __name__ == "__main__":
    generator = SyntheticDataGenerator(seed=42)
    generator.save_datasets("datasets")