# Notebook Template Generation - Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER UPLOADS CSV FILE                            │
│                   (via Streamlit interface)                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               notebook_interface.py (Line 298-320)                  │
│  1. Save file to: datasets/uploads/<filename>.csv                   │
│  2. Store in session: st.session_state.dataset_path                 │
│  3. Load into DataFrame: st.session_state.current_df                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│            User clicks "Generate Notebook" button                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│               notebook_interface.py (Line 542-544)                  │
│  Retrieve path: dataset_path = session_state.get('dataset_path')   │
│  Call: notebook_engine.generate_notebook(                          │
│           profile_results, template_type, name, dataset_path)       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│             template_engine.py - generate_notebook()                │
│  Pass dataset_path to appropriate template method                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│        Template Method (e.g., _create_eda_template)                │
│  1. Check if dataset_path provided:                                │
│     if dataset_path:                                               │
│         data_load_path = dataset_path                              │
│     else:                                                          │
│         data_load_path = f"{dataset_name}.csv"                     │
│                                                                    │
│  2. Generate notebook with actual path:                            │
│     df = pd.read_csv('{data_load_path}')                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  NOTEBOOK GENERATED ✓                               │
│  Contains: df = pd.read_csv('datasets/uploads/file.csv')          │
│  Ready to run immediately without manual path editing!             │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. File Persistence Layer
- **Location**: `notebook_interface.py` lines 298-320
- **Purpose**: Save uploaded files permanently
- **Output**: Physical file at `datasets/uploads/<filename>.csv`

### 2. Session State Tracking
- **Variables**:
  - `st.session_state.current_df` - Loaded DataFrame
  - `st.session_state.dataset_name` - Display name
  - `st.session_state.dataset_path` - File path for notebooks
  - `st.session_state.profile_results` - Data analysis

### 3. Path Propagation
- **Location**: `notebook_interface.py` line 542
- **Purpose**: Pass saved path to template engine
- **Flow**: Session state → generate_notebook() call

### 4. Template Generation
- **Location**: `template_engine.py` all template methods
- **Purpose**: Use actual path in generated notebook code
- **Logic**: 
  ```python
  if dataset_path:
      data_load_path = dataset_path  # Use provided path
  else:
      data_load_path = f"{dataset_name}.csv"  # Fallback
  ```

## Example Paths

| Source | Saved Location | Used in Notebook |
|--------|----------------|------------------|
| Uploaded: `transactions.csv` | `datasets/uploads/transactions.csv` | `df = pd.read_csv('datasets/uploads/transactions.csv')` |
| Demo data | `demo_datasets/customer_demo.csv` | `df = pd.read_csv('demo_datasets/customer_demo.csv')` |
| Legacy/Manual | N/A | `df = pd.read_csv('dataset_name.csv')` (fallback) |

## Testing Checklist

- [ ] Upload CSV file
- [ ] Verify file saved to `datasets/uploads/`
- [ ] Generate EDA notebook
- [ ] Download notebook
- [ ] Check pd.read_csv() line has correct path
- [ ] Open in Jupyter Notebook
- [ ] Run first code cell - should load data successfully
- [ ] Verify no manual path editing needed
