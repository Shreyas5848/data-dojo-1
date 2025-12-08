# Notebook Template Generation Fix - Custom Datasets

## Problem
When users uploaded their own CSV files to generate notebook templates, the generated notebooks contained a placeholder path `'your_file.csv'` instead of the actual dataset path. This meant users had to manually edit the notebook to fix the data loading path before they could run it.

## Solution
The fix involved three key changes:

### 1. Enhanced Template Engine (`template_engine.py`)
- Added `dataset_path: Optional[str]` parameter to `generate_notebook()` method
- Updated all 8 template methods to accept and use the dataset path:
  - `_create_eda_template`
  - `_create_cleaning_template`
  - `_create_classification_template`
  - `_create_regression_template`
  - `_create_timeseries_template`
  - `_create_clustering_template`
  - `_create_dimensionality_template`
  - `_create_feature_engineering_template`

- Each template now dynamically determines the data loading path:
```python
# Determine the dataset path to use
if dataset_path:
    data_load_path = dataset_path
else:
    data_load_path = f"{dataset_name}.csv"  # Default fallback
```

- The generated notebooks now use the actual path:
```python
df = pd.read_csv('datasets/uploads/your_actual_file.csv')
```

### 2. File Persistence (`notebook_interface.py`)
When users upload a CSV file or use demo data:
- The file is saved to a permanent location:
  - Uploaded files: `datasets/uploads/`
  - Demo datasets: `demo_datasets/`
- The path is stored in session state: `st.session_state.dataset_path`
- This ensures the notebook can reference a valid file path

### 3. Path Propagation
- The notebook generation call now passes the dataset path:
```python
dataset_path = st.session_state.get('dataset_path', None)
notebook = st.session_state.notebook_engine.generate_notebook(
    profile_results, selected_template, dataset_name, dataset_path
)
```

## Benefits
1. **Zero Manual Configuration**: Users can download and run notebooks immediately
2. **Correct Paths**: Data loading code uses actual file locations
3. **Persistent Storage**: Uploaded files are saved for notebook execution
4. **Backward Compatible**: Falls back to `{dataset_name}.csv` if no path provided

## Testing
To test the fix:
1. Go to web dashboard: `python app.py` or `datadojo web`
2. Navigate to "Notebook Templates" page
3. Upload a CSV file (e.g., `bank_transactions.csv`)
4. Generate a notebook template (any type)
5. Download the notebook
6. Open in Jupyter - verify the `pd.read_csv()` line has the correct path
7. Run all cells - should work without manual path editing

## Files Modified
- `src/datadojo/notebook/template_engine.py` - Added dataset_path parameter
- `src/datadojo/web/notebook_interface.py` - File saving and path tracking
- `app.py` - Already using verbose=False for profiler (no changes needed)

## Example Generated Code
**Before Fix:**
```python
df = pd.read_csv('your_file.csv')  # User had to manually edit this
```

**After Fix:**
```python
df = pd.read_csv('datasets/uploads/bank_transactions.csv')  # Works immediately!
```
