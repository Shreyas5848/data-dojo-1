"""Concept explanations database for DataDojo.

Pre-loaded educational concepts covering data preprocessing, machine learning,
and data science fundamentals.
"""

from typing import Dict, List, Optional
from ..models.educational_content import EducationalContent, DifficultyLevel


class ConceptDatabase:
    """Database of educational concepts for data science learning."""

    def __init__(self):
        """Initialize the concept database."""
        self._concepts: Dict[str, EducationalContent] = {}
        self._load_default_concepts()

    def _load_default_concepts(self) -> None:
        """Load default educational concepts."""
        # Missing data concepts
        self.add_concept(EducationalContent(
            concept_id="missing_values",
            title="Handling Missing Values",
            explanation=(
                "Missing values are gaps in your dataset where no data was recorded. "
                "They can occur due to data collection errors, sensor failures, or "
                "intentional non-responses. Common approaches include: deletion (remove "
                "rows/columns), imputation (fill with mean/median/mode), or using "
                "algorithms that handle missing values natively."
            ),
            analogies=[
                "Like blank spaces in a form that need to be filled in",
                "Similar to incomplete answers in a survey"
            ],
            examples=[
                "# Check for missing values\ndf.isnull().sum()\n\n# Fill with mean\ndf['age'].fillna(df['age'].mean(), inplace=True)\n\n# Drop rows with any missing values\ndf.dropna(inplace=True)"
            ],
            difficulty_level=DifficultyLevel.BEGINNER
        ))

        self.add_concept(EducationalContent(
            concept_id="outliers",
            title="Detecting and Handling Outliers",
            explanation=(
                "Outliers are data points that differ significantly from other observations. "
                "They can be caused by measurement errors, data entry mistakes, or represent "
                "genuine extreme values. Detection methods include IQR (Interquartile Range), "
                "Z-score, and isolation forests. Handling strategies include removal, "
                "transformation, or robust modeling techniques."
            ),
            analogies=[
                "Like finding a person 7 feet tall in a crowd of average-height people",
                "Similar to one extremely expensive item in a list of regular purchases"
            ],
            examples=[
                "# Detect outliers using IQR\nQ1 = df['price'].quantile(0.25)\nQ3 = df['price'].quantile(0.75)\nIQR = Q3 - Q1\nlower = Q1 - 1.5 * IQR\nupper = Q3 + 1.5 * IQR\noutliers = df[(df['price'] < lower) | (df['price'] > upper)]"
            ],
            difficulty_level=DifficultyLevel.BEGINNER,
            related_concepts=["data_quality", "data_cleaning"]
        ))

        # Data types and transformation
        self.add_concept(EducationalContent(
            concept_id="data_types",
            title="Understanding Data Types",
            explanation=(
                "Data types define what kind of values a variable can hold and what operations "
                "can be performed. Common types include: numeric (int, float), categorical "
                "(string, category), datetime, and boolean. Proper type handling is crucial "
                "for correct analysis and memory efficiency."
            ),
            analogies=[
                "Like different containers for different things: boxes for solids, bottles for liquids"
            ],
            examples=[
                "# Check data types\nprint(df.dtypes)\n\n# Convert to numeric\ndf['age'] = pd.to_numeric(df['age'], errors='coerce')\n\n# Convert to category\ndf['gender'] = df['gender'].astype('category')"
            ],
            difficulty_level=DifficultyLevel.BEGINNER,
            related_concepts=["type_conversion", "data_validation"]
        ))

        self.add_concept(EducationalContent(
            concept_id="normalization",
            title="Data Normalization and Scaling",
            explanation=(
                "Normalization scales numerical features to a common range, typically [0,1] or "
                "[-1,1]. Standardization transforms data to have mean=0 and std=1. These are "
                "important for algorithms sensitive to feature scales like neural networks, "
                "KNN, and gradient descent-based methods."
            ),
            analogies=[
                "Like converting different currencies to a common unit",
                "Similar to grading on a curve in school"
            ],
            examples=[
                "from sklearn.preprocessing import MinMaxScaler, StandardScaler\n\n# Min-Max normalization (0-1)\nscaler = MinMaxScaler()\ndf['price_normalized'] = scaler.fit_transform(df[['price']])\n\n# Standardization (mean=0, std=1)\nscaler = StandardScaler()\ndf['price_standardized'] = scaler.fit_transform(df[['price']])"
            ],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            related_concepts=["feature_engineering", "preprocessing"]
        ))

        # Feature engineering
        self.add_concept(EducationalContent(
            concept_id="feature_engineering",
            title="Feature Engineering",
            explanation=(
                "Feature engineering creates new features from existing data to improve model "
                "performance. Techniques include: creating interaction terms, polynomial features, "
                "binning continuous variables, extracting date components, aggregating data, "
                "and domain-specific transformations."
            ),
            analogies=[
                "Like creating new tools from existing materials",
                "Similar to deriving insights from raw ingredients"
            ],
            examples=[
                "# Create interaction features\ndf['age_income'] = df['age'] * df['income']\n\n# Extract date features\ndf['year'] = pd.to_datetime(df['date']).dt.year\ndf['month'] = pd.to_datetime(df['date']).dt.month\ndf['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek\n\n# Binning\ndf['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['child', 'young_adult', 'adult', 'senior'])"
            ],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            related_concepts=["feature_selection", "domain_knowledge"]
        ))

        self.add_concept(EducationalContent(
            concept_id="encoding_categorical",
            title="Encoding Categorical Variables",
            explanation=(
                "Machine learning algorithms require numerical input, so categorical variables "
                "must be converted to numbers. Methods include: Label Encoding (ordinal), "
                "One-Hot Encoding (nominal), Target Encoding (using target statistics), and "
                "Frequency Encoding. Choice depends on the variable's nature and cardinality."
            ),
            analogies=[
                "Like assigning numbers to colors: red=1, blue=2, green=3",
                "Similar to converting text answers into numerical scores"
            ],
            examples=[
                "# Label encoding (ordinal)\nfrom sklearn.preprocessing import LabelEncoder\nle = LabelEncoder()\ndf['size_encoded'] = le.fit_transform(df['size'])  # S, M, L, XL -> 0,1,2,3\n\n# One-hot encoding (nominal)\ndf_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')\n\n# Target encoding\nmean_target = df.groupby('category')['target'].mean()\ndf['category_encoded'] = df['category'].map(mean_target)"
            ],
            difficulty_level=DifficultyLevel.INTERMEDIATE,
            related_concepts=["categorical_data", "feature_engineering"]
        ))

        # Data quality
        self.add_concept(EducationalContent(
            concept_id="data_quality",
            title="Data Quality Assessment",
            explanation=(
                "Data quality encompasses completeness, accuracy, consistency, and validity. "
                "Key checks include: missing value analysis, duplicate detection, constraint "
                "validation, distribution analysis, and consistency checks across related fields."
            ),
            analogies=[
                "Like quality control in manufacturing",
                "Similar to proofreading a document for errors"
            ],
            examples=[
                "# Completeness check\nmissing_pct = df.isnull().sum() / len(df) * 100\n\n# Duplicates\nduplicates = df.duplicated().sum()\n\n# Value ranges\nassert df['age'].between(0, 120).all(), 'Invalid ages found'\n\n# Consistency\nassert (df['start_date'] <= df['end_date']).all(), 'Date inconsistency'"
            ],
            difficulty_level=DifficultyLevel.BEGINNER,
            related_concepts=["data_validation", "data_cleaning"]
        ))

        # Advanced concepts
        self.add_concept(EducationalContent(
            concept_id="imbalanced_data",
            title="Handling Imbalanced Datasets",
            explanation=(
                "Imbalanced data occurs when classes are not equally represented, common in "
                "fraud detection, disease diagnosis, etc. Techniques include: oversampling "
                "(SMOTE), undersampling, class weights, ensemble methods, and using appropriate "
                "evaluation metrics (precision, recall, F1, AUC-ROC instead of accuracy)."
            ),
            analogies=[
                "Like trying to find a needle in a haystack",
                "Similar to detecting rare events in a sea of normal occurrences"
            ],
            examples=[
                "from imblearn.over_sampling import SMOTE\nfrom sklearn.utils.class_weight import compute_class_weight\n\n# SMOTE oversampling\nsmote = SMOTE(random_state=42)\nX_resampled, y_resampled = smote.fit_resample(X, y)\n\n# Class weights\nclass_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)\nmodel = RandomForestClassifier(class_weight='balanced')"
            ],
            difficulty_level=DifficultyLevel.ADVANCED,
            related_concepts=["classification", "sampling", "evaluation_metrics"]
        ))

        self.add_concept(EducationalContent(
            concept_id="dimensionality_reduction",
            title="Dimensionality Reduction",
            explanation=(
                "Dimensionality reduction reduces the number of features while preserving "
                "important information. Methods include PCA (Principal Component Analysis), "
                "t-SNE, UMAP, and feature selection. Benefits: reduced overfitting, faster "
                "training, easier visualization, and lower storage requirements."
            ),
            analogies=[
                "Like summarizing a book into key points",
                "Similar to creating a 2D map from a 3D world"
            ],
            examples=[
                "from sklearn.decomposition import PCA\nfrom sklearn.manifold import TSNE\n\n# PCA\npca = PCA(n_components=10)\nX_reduced = pca.fit_transform(X)\nprint(f'Variance explained: {pca.explained_variance_ratio_.sum():.2%}')\n\n# t-SNE for visualization\ntsne = TSNE(n_components=2, random_state=42)\nX_2d = tsne.fit_transform(X)"
            ],
            difficulty_level=DifficultyLevel.ADVANCED,
            related_concepts=["feature_selection", "pca", "visualization"]
        ))

    def add_concept(self, concept: EducationalContent) -> None:
        """Add a concept to the database.

        Args:
            concept: EducationalContent to add
        """
        self._concepts[concept.concept_id] = concept

    def get_concept(self, concept_id: str) -> Optional[EducationalContent]:
        """Get a concept by ID.

        Args:
            concept_id: Concept identifier

        Returns:
            EducationalContent if found, None otherwise
        """
        return self._concepts.get(concept_id)

    def list_concepts(
        self,
        difficulty: Optional[DifficultyLevel] = None
    ) -> List[EducationalContent]:
        """List all concepts, optionally filtered by difficulty.

        Args:
            difficulty: Filter by difficulty level

        Returns:
            List of EducationalContent
        """
        concepts = list(self._concepts.values())
        if difficulty:
            concepts = [c for c in concepts if c.difficulty_level == difficulty]
        return concepts

    def search_concepts(self, keyword: str) -> List[EducationalContent]:
        """Search concepts by keyword in title or explanation.

        Args:
            keyword: Search keyword

        Returns:
            List of matching EducationalContent
        """
        keyword = keyword.lower()
        results = []
        for concept in self._concepts.values():
            if (keyword in concept.title.lower() or
                keyword in concept.explanation.lower()):
                results.append(concept)
        return results

    def get_related_concepts(self, concept_id: str) -> List[EducationalContent]:
        """Get concepts related to a given concept.

        Args:
            concept_id: Concept identifier

        Returns:
            List of related EducationalContent
        """
        concept = self.get_concept(concept_id)
        if not concept:
            return []

        related = []
        for related_id in concept.related_concepts:
            related_concept = self.get_concept(related_id)
            if related_concept:
                related.append(related_concept)
        return related


# Global concept database instance
_concept_db = None


def get_concept_database() -> ConceptDatabase:
    """Get the global concept database instance.

    Returns:
        ConceptDatabase singleton
    """
    global _concept_db
    if _concept_db is None:
        _concept_db = ConceptDatabase()
    return _concept_db
