"""
Intelligent Data Profiler for DataDojo
AI-powered analysis with actionable recommendations and automated insights.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


@dataclass
class ColumnProfile:
    """Comprehensive profile for a single column."""
    name: str
    dtype: str
    null_count: int = 0
    null_percentage: float = 0.0
    unique_count: int = 0
    unique_percentage: float = 0.0
    memory_usage: int = 0
    
    # Numeric-specific
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # String-specific
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None
    
    # Categorical
    top_values: Dict[str, int] = field(default_factory=dict)
    
    # Quality Issues
    quality_issues: List[str] = field(default_factory=list)
    data_quality_score: float = 1.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Patterns
    patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetProfile:
    """Complete profile for a dataset."""
    name: str
    shape: Tuple[int, int]
    memory_usage_mb: float
    column_profiles: Dict[str, ColumnProfile]
    
    # Cross-column analysis
    correlations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    duplicate_rows: int = 0
    duplicate_percentage: float = 0.0
    
    # Quality metrics
    overall_quality_score: float = 1.0
    completeness_score: float = 1.0
    consistency_score: float = 1.0
    uniqueness_score: float = 1.0
    
    # Business insights
    business_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Temporal analysis (if applicable)
    temporal_columns: List[str] = field(default_factory=list)
    temporal_insights: List[str] = field(default_factory=list)


class IntelligentProfiler:
    """AI-powered data profiler with actionable insights."""
    
    def __init__(self):
        self.profiles = {}
        
        # Pattern definitions for data validation
        self.patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^(\+?1[-.\s]?)?(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})$',
            'ssn': r'^\d{3}-?\d{2}-?\d{4}$',
            'zipcode': r'^\d{5}(-\d{4})?$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            'currency': r'^\$?\d{1,3}(,\d{3})*(\.\d{2})?$',
            'percentage': r'^\d{1,3}(\.\d{1,2})?%?$'
        }
        
        # Common data quality thresholds
        self.quality_thresholds = {
            'missing_data': 0.05,  # 5% threshold for missing data concern
            'high_cardinality': 0.95,  # 95% unique values suggests potential ID column
            'low_variance': 0.01,  # Very low variance in numeric data
            'skewness': 2.0,  # High skewness threshold
            'outlier_zscore': 3.0  # Z-score for outlier detection
        }
    
    def profile_dataset(self, df: pd.DataFrame, dataset_name: str = "Dataset", verbose: bool = True) -> DatasetProfile:
        """Create comprehensive profile of a dataset.
        
        Args:
            df: DataFrame to profile
            dataset_name: Name for the dataset
            verbose: If True, print progress messages. Set to False in web environments.
        """
        if verbose:
            try:
                print(f"Profiling dataset: {dataset_name}")
                print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Fallback for systems that can't handle emojis
                print(f"Profiling dataset: {dataset_name}")
                print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        
        # Basic dataset metrics
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Profile each column
        column_profiles = {}
        for col in df.columns:
            if verbose:
                try:
                    print(f"   Analyzing column: {col}")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    print(f"   Analyzing column: {col}")
            column_profiles[col] = self._profile_column(df, col)
        
        # Cross-column analysis
        correlations = self._analyze_correlations(df)
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(df)) * 100
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(df, column_profiles)
        
        # Business insights
        business_insights = self._generate_business_insights(df, column_profiles)
        recommendations = self._generate_dataset_recommendations(df, column_profiles)
        
        # Temporal analysis
        temporal_columns = self._identify_temporal_columns(df)
        temporal_insights = self._analyze_temporal_patterns(df, temporal_columns)
        
        profile = DatasetProfile(
            name=dataset_name,
            shape=df.shape,
            memory_usage_mb=memory_usage_mb,
            column_profiles=column_profiles,
            correlations=correlations,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_percentage,
            overall_quality_score=quality_scores['overall'],
            completeness_score=quality_scores['completeness'],
            consistency_score=quality_scores['consistency'],
            uniqueness_score=quality_scores['uniqueness'],
            business_insights=business_insights,
            recommendations=recommendations,
            temporal_columns=temporal_columns,
            temporal_insights=temporal_insights
        )
        
        self.profiles[dataset_name] = profile
        return profile
    
    def _profile_column(self, df: pd.DataFrame, column: str) -> ColumnProfile:
        """Create detailed profile for a single column."""
        series = df[column]
        
        profile = ColumnProfile(
            name=column,
            dtype=str(series.dtype),
            null_count=series.isnull().sum(),
            null_percentage=(series.isnull().sum() / len(series)) * 100,
            unique_count=series.nunique(),
            unique_percentage=(series.nunique() / len(series)) * 100,
            memory_usage=series.memory_usage(deep=True)
        )
        
        # Numeric analysis
        if pd.api.types.is_numeric_dtype(series):
            profile = self._analyze_numeric_column(series, profile)
        
        # String analysis
        elif pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            profile = self._analyze_string_column(series, profile)
        
        # Datetime analysis
        elif pd.api.types.is_datetime64_any_dtype(series):
            profile = self._analyze_datetime_column(series, profile)
        
        # Top values for categorical data
        if profile.unique_count <= 50:  # Treat as categorical if <= 50 unique values
            value_counts = series.value_counts().head(10)
            profile.top_values = value_counts.to_dict()
        
        # Quality analysis
        profile.quality_issues = self._identify_quality_issues(series, profile)
        profile.data_quality_score = self._calculate_column_quality_score(series, profile)
        profile.recommendations = self._generate_column_recommendations(series, profile)
        
        return profile
    
    def _analyze_numeric_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Analyze numeric column specifics."""
        # Skip boolean columns - they should be treated as categorical
        if series.dtype == 'bool':
            return profile
            
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()
        
        if len(numeric_series) > 0:
            profile.min_value = float(numeric_series.min())
            profile.max_value = float(numeric_series.max())
            profile.mean = float(numeric_series.mean())
            profile.median = float(numeric_series.median())
            profile.std = float(numeric_series.std())
            profile.q25 = float(numeric_series.quantile(0.25))
            profile.q75 = float(numeric_series.quantile(0.75))
            
            # Calculate skewness and kurtosis if enough data
            if len(numeric_series) >= 3:
                try:
                    profile.skewness = float(numeric_series.skew())
                    profile.kurtosis = float(numeric_series.kurtosis())
                except:
                    pass
            
            # Identify patterns
            outliers_count = 0
            if numeric_series.std() > 0:
                try:
                    z_scores = abs((numeric_series - numeric_series.mean()) / numeric_series.std())
                    outliers_count = int((z_scores > self.quality_thresholds['outlier_zscore']).sum())
                except:
                    outliers_count = 0
            
            profile.patterns = {
                'has_negatives': bool((numeric_series < 0).any()),
                'has_zeros': bool((numeric_series == 0).any()),
                'is_integer_like': bool(numeric_series.apply(lambda x: x == int(x)).all()) if len(numeric_series) > 0 else False,
                'outliers_count': outliers_count
            }
        
        return profile
    
    def _analyze_string_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Analyze string column specifics."""
        string_series = series.dropna().astype(str)
        
        if len(string_series) > 0:
            lengths = string_series.str.len()
            profile.min_length = int(lengths.min())
            profile.max_length = int(lengths.max())
            profile.avg_length = float(lengths.mean())
            
            # Pattern matching
            profile.patterns = {}
            for pattern_name, pattern in self.patterns.items():
                matches = string_series.str.match(pattern, na=False).sum()
                if matches > 0:
                    profile.patterns[f'{pattern_name}_matches'] = matches
                    profile.patterns[f'{pattern_name}_percentage'] = (matches / len(string_series)) * 100
            
            # Common string issues
            profile.patterns.update({
                'has_leading_trailing_spaces': string_series.str.strip().ne(string_series).any(),
                'has_mixed_case': string_series.str.lower().ne(string_series).any() and string_series.str.upper().ne(string_series).any(),
                'empty_strings': (string_series == '').sum(),
                'only_whitespace': string_series.str.strip().eq('').sum(),
                'contains_special_chars': string_series.str.contains(r'[^a-zA-Z0-9\s]', regex=True, na=False).sum()
            })
        
        return profile
    
    def _analyze_datetime_column(self, series: pd.Series, profile: ColumnProfile) -> ColumnProfile:
        """Analyze datetime column specifics."""
        datetime_series = pd.to_datetime(series, errors='coerce').dropna()
        
        if len(datetime_series) > 0:
            profile.min_value = datetime_series.min().isoformat()
            profile.max_value = datetime_series.max().isoformat()
            
            # Temporal patterns
            date_range_days = (datetime_series.max() - datetime_series.min()).days
            future_dates_count = int((datetime_series > pd.Timestamp.now()).sum())
            weekend_count = int(datetime_series.dt.weekday.isin([5, 6]).sum())
            
            # Check if datetime has time component (not just date)
            has_time = False
            try:
                has_time = bool((datetime_series.dt.time != pd.Timestamp('2000-01-01').time()).any())
            except:
                has_time = False
            
            # Business hours (9-17) - only if has time component
            business_hours_count = 0
            if has_time:
                try:
                    hours = datetime_series.dt.hour
                    business_hours_count = int(((hours >= 9) & (hours <= 17)).sum())
                except:
                    business_hours_count = 0
            
            profile.patterns = {
                'date_range_days': date_range_days,
                'future_dates': future_dates_count,
                'weekend_dates': weekend_count,
                'business_hours': business_hours_count,
                'has_time_component': has_time
            }
        
        return profile
    
    def _identify_quality_issues(self, series: pd.Series, profile: ColumnProfile) -> List[str]:
        """Identify data quality issues in a column."""
        issues = []
        
        # Missing data
        if profile.null_percentage > self.quality_thresholds['missing_data'] * 100:
            issues.append(f"High missing data: {profile.null_percentage:.1f}%")
        
        # High cardinality (potential ID column)
        if profile.unique_percentage > self.quality_thresholds['high_cardinality'] * 100:
            issues.append(f"Very high cardinality: {profile.unique_percentage:.1f}% unique values")
        
        # Low variance for numeric columns
        if pd.api.types.is_numeric_dtype(series) and profile.std is not None:
            if profile.std < self.quality_thresholds['low_variance']:
                issues.append("Very low variance - column may be constant")
        
        # High skewness
        if profile.skewness is not None and abs(profile.skewness) > self.quality_thresholds['skewness']:
            issues.append(f"High skewness: {profile.skewness:.2f}")
        
        # Outliers
        if 'outliers_count' in profile.patterns and profile.patterns['outliers_count'] > 0:
            outlier_percentage = (profile.patterns['outliers_count'] / len(series.dropna())) * 100
            if outlier_percentage > 5:  # More than 5% outliers
                issues.append(f"Many outliers detected: {profile.patterns['outliers_count']} ({outlier_percentage:.1f}%)")
        
        # String-specific issues
        if pd.api.types.is_string_dtype(series) or series.dtype == 'object':
            if profile.patterns.get('has_leading_trailing_spaces', False):
                issues.append("Contains leading/trailing whitespace")
            
            if profile.patterns.get('empty_strings', 0) > 0:
                issues.append(f"Contains {profile.patterns['empty_strings']} empty strings")
            
            if profile.patterns.get('only_whitespace', 0) > 0:
                issues.append(f"Contains {profile.patterns['only_whitespace']} whitespace-only strings")
        
        # Datetime-specific issues
        if 'future_dates' in profile.patterns and profile.patterns['future_dates'] > 0:
            issues.append(f"Contains {profile.patterns['future_dates']} future dates")
        
        return issues
    
    def _calculate_column_quality_score(self, series: pd.Series, profile: ColumnProfile) -> float:
        """Calculate quality score for a column (0-1 scale)."""
        score = 1.0
        
        # Penalize missing data
        score -= (profile.null_percentage / 100) * 0.3
        
        # Penalize quality issues
        issue_penalty = len(profile.quality_issues) * 0.1
        score -= min(issue_penalty, 0.5)  # Cap at 50% penalty
        
        return max(0.0, score)
    
    def _generate_column_recommendations(self, series: pd.Series, profile: ColumnProfile) -> List[str]:
        """Generate actionable recommendations for a column."""
        recommendations = []
        
        # Missing data recommendations
        if profile.null_percentage > 5:
            if profile.null_percentage > 50:
                recommendations.append("Consider dropping this column due to excessive missing data")
            elif pd.api.types.is_numeric_dtype(series):
                recommendations.append("Consider imputing missing values with median or mean")
            else:
                recommendations.append("Consider imputing missing values with mode or 'Unknown'")
        
        # High cardinality recommendations
        if profile.unique_percentage > 95:
            recommendations.append("This appears to be an ID column - consider using for indexing or dropping if not needed")
        
        # String cleaning recommendations
        if profile.patterns.get('has_leading_trailing_spaces', False):
            recommendations.append("Apply .str.strip() to remove leading/trailing whitespace")
        
        if profile.patterns.get('has_mixed_case', False):
            recommendations.append("Consider standardizing case with .str.lower() or .str.title()")
        
        # Numeric recommendations
        if profile.skewness is not None:
            if abs(profile.skewness) > 2:
                recommendations.append("Consider log transformation to reduce skewness")
        
        if 'outliers_count' in profile.patterns and profile.patterns['outliers_count'] > 0:
            recommendations.append("Investigate and potentially handle outliers")
        
        # Data type optimization
        if pd.api.types.is_numeric_dtype(series):
            if profile.patterns.get('is_integer_like', False) and 'float' in str(series.dtype):
                recommendations.append("Consider converting to integer type for memory efficiency")
        
        # Pattern-based recommendations
        email_matches = profile.patterns.get('email_percentage', 0)
        if email_matches > 80:
            recommendations.append("This appears to be an email column - validate email format")
        
        phone_matches = profile.patterns.get('phone_percentage', 0)
        if phone_matches > 80:
            recommendations.append("This appears to be a phone column - standardize phone number format")
        
        return recommendations
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find high correlations (> 0.7 or < -0.7)
        high_correlations = {}
        for col1 in numeric_cols:
            high_corr = {}
            for col2 in numeric_cols:
                if col1 != col2:
                    corr_value = corr_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.7:
                        high_corr[col2] = round(corr_value, 3)
            if high_corr:
                high_correlations[col1] = high_corr
        
        return high_correlations
    
    def _calculate_quality_scores(self, df: pd.DataFrame, column_profiles: Dict[str, ColumnProfile]) -> Dict[str, float]:
        """Calculate overall quality scores for the dataset."""
        
        # Completeness score (based on missing data)
        total_values = df.shape[0] * df.shape[1]
        missing_values = sum(profile.null_count for profile in column_profiles.values())
        completeness_score = 1.0 - (missing_values / total_values)
        
        # Consistency score (based on data types and patterns)
        consistency_issues = sum(len(profile.quality_issues) for profile in column_profiles.values())
        total_columns = len(column_profiles)
        consistency_score = max(0.0, 1.0 - (consistency_issues / (total_columns * 5)))  # Assume max 5 issues per column
        
        # Uniqueness score (based on duplicates)
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        uniqueness_score = max(0.0, 1.0 - (duplicate_percentage / 100))
        
        # Overall score (weighted average)
        overall_score = (
            completeness_score * 0.4 +
            consistency_score * 0.4 +
            uniqueness_score * 0.2
        )
        
        return {
            'overall': round(overall_score, 3),
            'completeness': round(completeness_score, 3),
            'consistency': round(consistency_score, 3),
            'uniqueness': round(uniqueness_score, 3)
        }
    
    def _generate_business_insights(self, df: pd.DataFrame, column_profiles: Dict[str, ColumnProfile]) -> List[str]:
        """Generate business-relevant insights from the data."""
        insights = []
        
        # Dataset size insights
        if df.shape[0] > 100000:
            insights.append(f"Large dataset with {df.shape[0]:,} records - consider sampling for initial analysis")
        elif df.shape[0] < 100:
            insights.append(f"Small dataset with only {df.shape[0]} records - may need more data for reliable analysis")
        
        # Missing data patterns
        high_missing_cols = [name for name, profile in column_profiles.items() if profile.null_percentage > 20]
        if high_missing_cols:
            insights.append(f"Columns with significant missing data: {', '.join(high_missing_cols)}")
        
        # Potential ID columns
        id_cols = [name for name, profile in column_profiles.items() if profile.unique_percentage > 95]
        if id_cols:
            insights.append(f"Potential identifier columns detected: {', '.join(id_cols)}")
        
        # Categorical vs numeric balance
        categorical_cols = len([p for p in column_profiles.values() if p.unique_count <= 50])
        numeric_cols = len([p for p in column_profiles.values() if pd.api.types.is_numeric_dtype])
        insights.append(f"Data composition: {categorical_cols} categorical, {numeric_cols} numeric columns")
        
        # Memory usage insights
        total_memory_mb = sum(profile.memory_usage for profile in column_profiles.values()) / (1024 * 1024)
        if total_memory_mb > 100:
            insights.append(f"High memory usage ({total_memory_mb:.1f} MB) - consider data type optimization")
        
        return insights
    
    def _generate_dataset_recommendations(self, df: pd.DataFrame, column_profiles: Dict[str, ColumnProfile]) -> List[str]:
        """Generate dataset-level recommendations."""
        recommendations = []
        
        # Duplicate handling
        if df.duplicated().sum() > 0:
            recommendations.append(f"Remove {df.duplicated().sum()} duplicate rows")
        
        # Column dropping suggestions
        constant_cols = [name for name, profile in column_profiles.items() 
                        if profile.unique_count == 1 and profile.null_count == 0]
        if constant_cols:
            recommendations.append(f"Consider dropping constant columns: {', '.join(constant_cols)}")
        
        high_missing_cols = [name for name, profile in column_profiles.items() if profile.null_percentage > 90]
        if high_missing_cols:
            recommendations.append(f"Consider dropping columns with >90% missing data: {', '.join(high_missing_cols)}")
        
        # Data type optimizations
        float_to_int_cols = [name for name, profile in column_profiles.items() 
                           if profile.patterns.get('is_integer_like', False) and 'float' in profile.dtype]
        if float_to_int_cols:
            recommendations.append(f"Convert to integer type: {', '.join(float_to_int_cols)}")
        
        # Correlation warnings
        high_corr_pairs = []
        for col, correlations in self._analyze_correlations(df).items():
            for corr_col, corr_val in correlations.items():
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append(f"{col}-{corr_col} ({corr_val:.2f})")
        
        if high_corr_pairs:
            recommendations.append(f"High correlation detected - consider removing redundant features: {', '.join(high_corr_pairs)}")
        
        return recommendations
    
    def _identify_temporal_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify datetime and date-like columns."""
        temporal_cols = []
        
        for col in df.columns:
            try:
                # Direct datetime columns
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    temporal_cols.append(col)
                # String columns that might be dates - only check columns with date-like names
                elif df[col].dtype == 'object' and any(kw in col.lower() for kw in ['date', 'time', 'timestamp', 'dt', 'created', 'updated']):
                    sample = df[col].dropna().astype(str).head(20)  # Reduced sample size
                    if len(sample) > 0:
                        date_matches = sum(1 for val in sample if pd.to_datetime(val, errors='coerce') is not pd.NaT)
                        if date_matches > len(sample) * 0.8:  # 80% of samples are valid dates
                            temporal_cols.append(col)
            except Exception:
                continue  # Skip problematic columns
        
        return temporal_cols
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame, temporal_columns: List[str]) -> List[str]:
        """Analyze patterns in temporal data."""
        insights = []
        
        for col in temporal_columns:
            try:
                dt_series = pd.to_datetime(df[col], errors='coerce').dropna()
                if len(dt_series) == 0:
                    continue
                
                # Date range analysis
                date_range = (dt_series.max() - dt_series.min()).days
                insights.append(f"{col}: Spans {date_range} days from {dt_series.min().date()} to {dt_series.max().date()}")
                
                # Frequency analysis
                freq_analysis = dt_series.dt.date.value_counts()
                if len(freq_analysis) > 1:
                    most_common_date = freq_analysis.index[0]
                    most_common_count = freq_analysis.iloc[0]
                    insights.append(f"{col}: Most common date is {most_common_date} ({most_common_count} occurrences)")
                
                # Seasonal patterns (if enough data)
                if date_range > 365:
                    monthly_counts = dt_series.dt.month.value_counts().sort_index()
                    peak_month = monthly_counts.idxmax()
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    insights.append(f"{col}: Peak activity in {month_names[peak_month-1]}")
                
            except Exception as e:
                insights.append(f"{col}: Could not analyze temporal patterns - {str(e)}")
        
        return insights
    
    def generate_report(self, profile: DatasetProfile, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive text report."""
        
        report = []
        report.append("="*80)
        report.append(f"📊 DATA PROFILE REPORT: {profile.name}")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset Overview
        report.append("📈 DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Shape: {profile.shape[0]:,} rows × {profile.shape[1]} columns")
        report.append(f"Memory Usage: {profile.memory_usage_mb:.1f} MB")
        report.append(f"Duplicate Rows: {profile.duplicate_rows:,} ({profile.duplicate_percentage:.1f}%)")
        report.append("")
        
        # Quality Scores
        report.append("🎯 DATA QUALITY SCORES")
        report.append("-" * 40)
        report.append(f"Overall Quality: {profile.overall_quality_score:.1%} {'🟢' if profile.overall_quality_score > 0.8 else '🟡' if profile.overall_quality_score > 0.6 else '🔴'}")
        report.append(f"Completeness: {profile.completeness_score:.1%}")
        report.append(f"Consistency: {profile.consistency_score:.1%}")
        report.append(f"Uniqueness: {profile.uniqueness_score:.1%}")
        report.append("")
        
        # Column Analysis
        report.append("📋 COLUMN ANALYSIS")
        report.append("-" * 40)
        
        for col_name, col_profile in profile.column_profiles.items():
            report.append(f"\n🔸 {col_name} ({col_profile.dtype})")
            report.append(f"   • Missing: {col_profile.null_count:,} ({col_profile.null_percentage:.1f}%)")
            report.append(f"   • Unique: {col_profile.unique_count:,} ({col_profile.unique_percentage:.1f}%)")
            report.append(f"   • Quality Score: {col_profile.data_quality_score:.1%}")
            
            # Numeric details
            if col_profile.mean is not None:
                report.append(f"   • Range: {col_profile.min_value:.2f} - {col_profile.max_value:.2f}")
                report.append(f"   • Mean: {col_profile.mean:.2f}, Median: {col_profile.median:.2f}")
                if col_profile.skewness is not None:
                    report.append(f"   • Skewness: {col_profile.skewness:.2f}")
            
            # String details
            if col_profile.avg_length is not None:
                report.append(f"   • Length: {col_profile.min_length}-{col_profile.max_length} chars (avg: {col_profile.avg_length:.1f})")
            
            # Quality issues
            if col_profile.quality_issues:
                report.append(f"   ⚠️  Issues: {'; '.join(col_profile.quality_issues)}")
            
            # Top values for categorical
            if col_profile.top_values:
                top_vals = list(col_profile.top_values.items())[:3]
                top_str = ", ".join([f"{k}: {v}" for k, v in top_vals])
                report.append(f"   • Top values: {top_str}")
        
        # Correlations
        if profile.correlations:
            report.append("\n🔗 HIGH CORRELATIONS")
            report.append("-" * 40)
            for col, corrs in profile.correlations.items():
                for corr_col, corr_val in corrs.items():
                    report.append(f"   • {col} ↔ {corr_col}: {corr_val:.3f}")
        
        # Temporal Insights
        if profile.temporal_insights:
            report.append("\n⏰ TEMPORAL ANALYSIS")
            report.append("-" * 40)
            for insight in profile.temporal_insights:
                report.append(f"   • {insight}")
        
        # Business Insights
        if profile.business_insights:
            report.append("\n💡 BUSINESS INSIGHTS")
            report.append("-" * 40)
            for insight in profile.business_insights:
                report.append(f"   • {insight}")
        
        # Recommendations
        if profile.recommendations:
            report.append("\n🚀 RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(profile.recommendations, 1):
                report.append(f"   {i}. {rec}")
        
        # Column Recommendations
        col_recs = []
        for col_name, col_profile in profile.column_profiles.items():
            if col_profile.recommendations:
                col_recs.extend([f"{col_name}: {rec}" for rec in col_profile.recommendations])
        
        if col_recs:
            report.append("\n🔧 COLUMN-SPECIFIC RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(col_recs, 1):
                report.append(f"   {i}. {rec}")
        
        report.append("\n" + "="*80)
        
        report_text = "\n".join(report)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_path}")
        
        return report_text
    
    def export_profile_json(self, profile: DatasetProfile, output_path: str):
        """Export profile data as JSON for programmatic use."""
        
        def serialize_profile(obj):
            """Custom serializer for profile objects."""
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        profile_dict = {
            'metadata': {
                'name': profile.name,
                'generated_at': datetime.now().isoformat(),
                'shape': profile.shape,
                'memory_usage_mb': profile.memory_usage_mb
            },
            'quality_scores': {
                'overall': profile.overall_quality_score,
                'completeness': profile.completeness_score,
                'consistency': profile.consistency_score,
                'uniqueness': profile.uniqueness_score
            },
            'columns': {name: serialize_profile(col_profile) for name, col_profile in profile.column_profiles.items()},
            'correlations': profile.correlations,
            'insights': {
                'business': profile.business_insights,
                'temporal': profile.temporal_insights,
                'recommendations': profile.recommendations
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profile_dict, f, indent=2, default=str)
        
        print(f"📊 Profile exported to JSON: {output_path}")
    
    def compare_profiles(self, profile1: DatasetProfile, profile2: DatasetProfile) -> str:
        """Compare two dataset profiles and highlight differences."""
        
        comparison = []
        comparison.append("="*80)
        comparison.append(f"📊 PROFILE COMPARISON")
        comparison.append("="*80)
        comparison.append(f"Dataset 1: {profile1.name}")
        comparison.append(f"Dataset 2: {profile2.name}")
        comparison.append("")
        
        # Basic comparison
        comparison.append("📈 BASIC METRICS")
        comparison.append("-" * 40)
        comparison.append(f"Rows: {profile1.shape[0]:,} vs {profile2.shape[0]:,} ({profile2.shape[0] - profile1.shape[0]:+,})")
        comparison.append(f"Columns: {profile1.shape[1]} vs {profile2.shape[1]} ({profile2.shape[1] - profile1.shape[1]:+})")
        comparison.append(f"Memory: {profile1.memory_usage_mb:.1f} MB vs {profile2.memory_usage_mb:.1f} MB")
        comparison.append("")
        
        # Quality comparison
        comparison.append("🎯 QUALITY SCORES")
        comparison.append("-" * 40)
        comparison.append(f"Overall: {profile1.overall_quality_score:.1%} vs {profile2.overall_quality_score:.1%}")
        comparison.append(f"Completeness: {profile1.completeness_score:.1%} vs {profile2.completeness_score:.1%}")
        comparison.append(f"Consistency: {profile1.consistency_score:.1%} vs {profile2.consistency_score:.1%}")
        comparison.append("")
        
        # Column changes
        cols1 = set(profile1.column_profiles.keys())
        cols2 = set(profile2.column_profiles.keys())
        
        added_cols = cols2 - cols1
        removed_cols = cols1 - cols2
        common_cols = cols1 & cols2
        
        if added_cols:
            comparison.append(f"➕ Added columns: {', '.join(added_cols)}")
        if removed_cols:
            comparison.append(f"➖ Removed columns: {', '.join(removed_cols)}")
        
        # Quality changes in common columns
        quality_changes = []
        for col in common_cols:
            q1 = profile1.column_profiles[col].data_quality_score
            q2 = profile2.column_profiles[col].data_quality_score
            diff = q2 - q1
            if abs(diff) > 0.1:  # Significant change
                direction = "↗️" if diff > 0 else "↘️"
                quality_changes.append(f"{col}: {q1:.1%} → {q2:.1%} {direction}")
        
        if quality_changes:
            comparison.append("\n📊 SIGNIFICANT QUALITY CHANGES")
            comparison.append("-" * 40)
            for change in quality_changes:
                comparison.append(f"   • {change}")
        
        comparison.append("\n" + "="*80)
        
        return "\n".join(comparison)


# Convenience function for quick profiling
def quick_profile(df: pd.DataFrame, dataset_name: str = "Dataset") -> DatasetProfile:
    """Quick profiling function for immediate use."""
    profiler = IntelligentProfiler()
    return profiler.profile_dataset(df, dataset_name)


# CLI entry point
if __name__ == "__main__":
    # Example usage
    print("🚀 Intelligent Data Profiler for DataDojo")
    print("This module provides comprehensive data profiling capabilities.")
    print("Import and use: from datadojo.utils.intelligent_profiler import quick_profile")