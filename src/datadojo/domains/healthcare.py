"""Healthcare domain module for DataDojo.

Provides healthcare-specific datasets, operations, and validation rules
for learning data preparation in medical/clinical contexts with emphasis
on privacy and regulatory compliance.
"""

from typing import List, Dict, Any
from ..models.learning_project import LearningProject, Domain, Difficulty, ValidationRule
from ..models.domain_module import DomainModule, OperationDefinition


def get_healthcare_module() -> DomainModule:
    """Get the healthcare domain module configuration.

    Returns:
        DomainModule with healthcare specific setup
    """
    module = DomainModule(
        domain_name="healthcare",
        display_name="Healthcare & Medical Analytics",
        description=(
            "Learn data preparation for healthcare analytics, patient records processing, "
            "clinical decision support, and medical research with HIPAA-compliant practices."
        )
    )

    # Add domain-specific operations
    module.add_operation(OperationDefinition(
        name="phi_deidentification",
        operation_type="data_cleaning",
        description="Remove or anonymize Protected Health Information (PHI)",
        parameters_schema={
            "phi_columns": "list",
            "method": "remove|hash|generalize"
        },
        educational_content=(
            "Protected Health Information (PHI) includes any information that can "
            "identify a patient. HIPAA requires proper de-identification for data sharing."
        )
    ))

    module.add_operation(OperationDefinition(
        name="icd_code_standardization",
        operation_type="transformation",
        description="Standardize ICD (International Classification of Diseases) codes",
        parameters_schema={
            "icd_version": "9|10",
            "normalize": "boolean"
        },
        educational_content=(
            "ICD codes are standardized medical diagnosis codes. ICD-10 is more "
            "detailed than ICD-9, and proper mapping is crucial for analysis."
        )
    ))

    module.add_operation(OperationDefinition(
        name="vitals_normalization",
        operation_type="transformation",
        description="Normalize vital signs to standard ranges",
        parameters_schema={
            "vitals": "list",
            "age_adjusted": "boolean"
        },
        educational_content=(
            "Vital signs like blood pressure, heart rate, and temperature need "
            "normalization for different age groups and medical conditions."
        )
    ))

    module.add_operation(OperationDefinition(
        name="lab_results_preprocessing",
        operation_type="feature_engineering",
        description="Process and normalize laboratory test results",
        parameters_schema={
            "test_types": "list",
            "handle_abnormal_flags": "boolean"
        },
        educational_content=(
            "Lab results often come with different units and reference ranges. "
            "Standardization is essential for meaningful comparisons."
        )
    ))

    # Add validation rules
    module.add_validation_rule({
        "name": "valid_age_range",
        "type": "range",
        "description": "Age must be between 0 and 120",
        "column": "age",
        "min_value": 0,
        "max_value": 120
    })

    module.add_validation_rule({
        "name": "valid_blood_pressure",
        "type": "composite",
        "description": "Systolic must be higher than diastolic",
        "condition": "systolic_bp > diastolic_bp"
    })

    module.add_validation_rule({
        "name": "phi_check",
        "type": "privacy",
        "description": "Ensure no direct patient identifiers in exported data",
        "prohibited_columns": ["ssn", "medical_record_number", "full_name"]
    })

    return module


def get_sample_projects() -> List[LearningProject]:
    """Get sample healthcare learning projects.

    Returns:
        List of pre-configured healthcare projects
    """
    projects = []

    # Beginner: Patient Demographics Cleaning
    projects.append(LearningProject(
        id="healthcare_beginner_01",
        name="Patient Demographics Data Cleaning",
        domain=Domain.HEALTHCARE,
        difficulty=Difficulty.BEGINNER,
        description=(
            "Learn to clean and standardize patient demographic data while maintaining "
            "privacy compliance. Handle missing values, standardize formats, and ensure "
            "data quality for healthcare analytics."
        ),
        dataset_path="datasets/healthcare/patient_demographics.csv",
        expected_outcomes=[
            "Handle missing demographic information appropriately",
            "Standardize gender, race, and ethnicity coding",
            "Validate age and date of birth consistency",
            "Remove or anonymize PHI (Protected Health Information)",
            "Ensure HIPAA compliance in data handling"
        ],
        validation_rules=[
            ValidationRule(
                name="age_dob_consistency",
                description="Age must match date of birth",
                check_type="consistency",
                parameters={"columns": ["age", "date_of_birth"]}
            ),
            ValidationRule(
                name="no_phi",
                description="No direct identifiers present",
                check_type="privacy",
                parameters={"check_columns": ["ssn", "mrn"]}
            )
        ]
    ))

    # Intermediate: Clinical Lab Results Processing
    projects.append(LearningProject(
        id="healthcare_intermediate_01",
        name="Laboratory Results Analysis Preparation",
        domain=Domain.HEALTHCARE,
        difficulty=Difficulty.INTERMEDIATE,
        description=(
            "Process and normalize clinical laboratory test results. Handle different "
            "units, reference ranges, and abnormal flags for medical research."
        ),
        dataset_path="datasets/healthcare/lab_results.csv",
        expected_outcomes=[
            "Standardize lab test units across different systems",
            "Normalize values to common reference ranges",
            "Handle critical and abnormal value flags",
            "Create features for longitudinal trend analysis",
            "Manage missing or cancelled test results"
        ],
        validation_rules=[
            ValidationRule(
                name="valid_test_values",
                description="Lab values must be within possible physiological ranges",
                check_type="range",
                parameters={"column": "value", "context_aware": True}
            )
        ]
    ))

    # Advanced: Electronic Health Records (EHR) Pipeline
    projects.append(LearningProject(
        id="healthcare_advanced_01",
        name="EHR Data Integration Pipeline",
        domain=Domain.HEALTHCARE,
        difficulty=Difficulty.ADVANCED,
        description=(
            "Build a comprehensive preprocessing pipeline for Electronic Health Records. "
            "Integrate data from multiple sources, handle temporal sequences, extract "
            "features for predictive modeling while maintaining privacy."
        ),
        dataset_path="datasets/healthcare/ehr_records.csv",
        expected_outcomes=[
            "Integrate data from multiple EHR systems",
            "Create temporal features from patient timelines",
            "Extract medication and procedure features",
            "Handle hierarchical diagnosis codes (ICD-10)",
            "Implement privacy-preserving transformations",
            "Engineer features for readmission prediction"
        ],
        validation_rules=[
            ValidationRule(
                name="temporal_consistency",
                description="Event timestamps must be in logical order",
                check_type="temporal",
                parameters={"columns": ["admission_date", "discharge_date"]}
            ),
            ValidationRule(
                name="hipaa_compliance",
                description="Data must meet HIPAA de-identification standards",
                check_type="privacy",
                parameters={"standard": "safe_harbor"}
            )
        ]
    ))

    return projects


def get_domain_concepts() -> List[Dict[str, Any]]:
    """Get healthcare specific data science concepts.

    Returns:
        List of educational concepts for healthcare domain
    """
    return [
        {
            "concept_id": "phi_hipaa",
            "title": "Protected Health Information (PHI) and HIPAA",
            "explanation": (
                "PHI is any health information that can identify an individual. HIPAA "
                "(Health Insurance Portability and Accountability Act) requires proper "
                "de-identification before sharing medical data. There are two methods: "
                "Safe Harbor (removing 18 types of identifiers) and Expert Determination."
            ),
            "difficulty_level": "beginner",
            "examples": [
                "# Remove direct identifiers\ndf_deidentified = df.drop(columns=['ssn', 'name', 'address', 'mrn'])\n\n# Hash indirect identifiers\nimport hashlib\ndf['patient_hash'] = df['patient_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())"
            ]
        },
        {
            "concept_id": "icd_codes",
            "title": "ICD Code Processing",
            "explanation": (
                "ICD (International Classification of Diseases) codes are standardized "
                "diagnosis codes used globally. ICD-10 has over 70,000 codes organized "
                "hierarchically. Proper mapping and grouping is essential for analysis."
            ),
            "difficulty_level": "intermediate",
            "examples": [
                "# Extract ICD-10 category (first 3 characters)\ndf['icd_category'] = df['icd10_code'].str[:3]\n\n# Map to broader disease groups\nicd_groups = {'E10-E14': 'Diabetes', 'I20-I25': 'Ischemic Heart Disease'}"
            ]
        },
        {
            "concept_id": "readmission_prediction",
            "title": "Hospital Readmission Prediction",
            "explanation": (
                "Predicting patient readmissions within 30 days is a key healthcare "
                "analytics task. Features include demographics, diagnosis codes, "
                "procedures, medications, and previous admission patterns."
            ),
            "difficulty_level": "advanced",
            "examples": [
                "# Create readmission features\ndf['length_of_stay'] = (df['discharge_date'] - df['admission_date']).dt.days\ndf['num_previous_admissions'] = df.groupby('patient_id').cumcount()\ndf['days_since_last_admission'] = df.groupby('patient_id')['admission_date'].diff().dt.days"
            ]
        },
        {
            "concept_id": "clinical_notes_processing",
            "title": "Clinical Notes Text Processing",
            "explanation": (
                "Clinical notes contain valuable unstructured information. Processing "
                "involves medical NER (Named Entity Recognition), extracting symptoms, "
                "diagnoses, medications, and negation detection."
            ),
            "difficulty_level": "advanced",
            "examples": [
                "# Extract medical entities (conceptual)\nimport spacy\nnlp = spacy.load('en_core_sci_sm')  # SciSpacy model\ndoc = nlp(clinical_note)\nmedications = [ent.text for ent in doc.ents if ent.label_ == 'MEDICATION']"
            ]
        }
    ]
