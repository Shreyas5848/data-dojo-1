# 🧹 Messy Data Guide - DataDojo Datasets

## Why Messy Data?

Real-world datasets are **NEVER** clean. DataDojo intentionally generates datasets with quality issues so you can practice data cleaning skills that employers actually need.

---

## 📊 Data Quality Issues by Type

### Healthcare Datasets (`patient_demographics.csv`, `lab_results.csv`)

**Intentional Issues (30% of records):**
- ❌ **Missing Values**: Empty emails, phones, BMI, insurance
- ❌ **Invalid Data**: Ages = -1, 0, 150, 999
- ❌ **Outlier BMI**: Values like 5.5 or 99.9
- ❌ **Future Dates**: Last visit dates in 2099
- ❌ **Inconsistent Formatting**: Phone numbers with/without "+1-"
- ❌ **Case Issues**: Names in ALL CAPS or lowercase
- ❌ **Whitespace**: `"  John  "` or `"\tBoston\t"`
- ❌ **Typos**: State codes like "N" instead of "NY"
- ❌ **Duplicates**: 8% exact or near-duplicate records

**Learning Opportunities:**
- Handle missing values (imputation vs deletion)
- Detect and fix outliers
- Validate date ranges
- Standardize formatting
- Remove duplicates
- Data type conversions

---

### E-Commerce Datasets (`customers_messy.csv`, `transactions.csv`)

**Intentional Issues (30% of records):**
- ❌ **Missing Data**: Empty emails, ages, states
- ❌ **Invalid Values**: Negative spending, phone = "555-FAKE"
- ❌ **Zip Code Issues**: "00000", "ABCDE", or empty
- ❌ **Future Registration**: Dates in the future
- ❌ **Mixed Case**: "BRONZE" vs "Bronze" loyalty tiers
- ❌ **Special Characters**: Names with "@#" or "O'"
- ❌ **Whitespace**: Spaces around values
- ❌ **Duplicates**: 7% duplicate customers

**Learning Opportunities:**
- Clean categorical data
- Validate business rules (no negative amounts)
- Standardize formats
- Handle special characters
- Deduplication strategies

---

### Finance Datasets (`bank_transactions.csv`, `credit_applications.csv`)

**Intentional Issues (25% of records):**
- ❌ **Missing Values**: Empty merchant names, categories
- ❌ **Invalid Amounts**: Negative values, outliers
- ❌ **Inconsistent Categories**: Mixed case, typos
- ❌ **Date Issues**: Future dates, invalid formats
- ❌ **Duplicates**: Transaction duplicates
- ❌ **Fraud Indicators**: Suspicious patterns to detect

**Learning Opportunities:**
- Anomaly detection
- Time series data cleaning
- Fraud pattern recognition
- Financial data validation

---

## 🎯 What You'll Learn

### Beginner Skills
1. **Missing Value Detection**: `df.isnull().sum()`
2. **Duplicate Removal**: `df.drop_duplicates()`
3. **Data Type Conversion**: `pd.to_numeric()`, `astype()`
4. **Basic Cleaning**: `.strip()`, `.lower()`, `.upper()`

### Intermediate Skills
1. **Imputation Strategies**: Mean, median, mode, forward-fill
2. **Outlier Detection**: IQR, Z-score methods
3. **Feature Engineering**: Create derived columns
4. **Data Validation**: Business rule checks
5. **Standardization**: Consistent formats

### Advanced Skills
1. **Fuzzy Matching**: Find near-duplicates
2. **Advanced Imputation**: KNN, MICE
3. **Anomaly Detection**: Isolation forests
4. **Pipeline Building**: Automated cleaning workflows

---

## 📈 Quality Score Breakdown

When you profile datasets, you'll see:
- **Overall Quality Score**: 65-85% (intentionally reduced)
- **Completeness**: Missing value percentage
- **Validity**: Invalid/outlier detection
- **Consistency**: Format/case issues
- **Uniqueness**: Duplicate percentage

---

## 🚀 Workflow

### 1. Generate Messy Data
```bash
generate-data --domain healthcare --size medium
```

### 2. Profile the Data
```bash
profile-data --file datasets/healthcare/patient_demographics.csv
```

### 3. Generate Cleaning Notebook
```bash
# In web dashboard:
# Upload dataset → Profile → Generate "Data Cleaning" notebook
```

### 4. Clean the Data
Work through the notebook:
- Handle missing values
- Remove duplicates
- Fix data types
- Standardize formats
- Validate ranges

### 5. Re-profile
Check your quality score improved!

---

## 💡 Pro Tips

1. **Start with EDA**: Always profile first to understand issues
2. **Document Decisions**: Why did you impute vs delete?
3. **Validate Business Rules**: Age > 0, spending >= 0, etc.
4. **Check Correlations**: Before/after cleaning
5. **Create Pipelines**: Make cleaning reproducible

---

## 🎓 Learning Projects

Each project includes datasets with domain-specific issues:

- **Healthcare**: HIPAA compliance, medical coding standards
- **E-Commerce**: Customer segmentation with dirty data
- **Finance**: Fraud detection with noisy transactions

Start with Beginner projects (pre-guided) → Intermediate (hints) → Advanced (independent).

---

## 📚 Additional Resources

- **Notebook Templates**: Auto-generated cleaning workflows
- **Concept Explanations**: `explain missing_values`, `explain outliers`
- **Progress Tracking**: Earn XP for cleaning datasets
- **Achievements**: "Data Janitor", "Clean Sweep", "Quality Guardian"

Happy cleaning! 🧹✨
