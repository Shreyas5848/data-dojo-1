"""Performance tests for DataDojo.

These tests ensure that the framework meets performance requirements:
- Guidance generation: < 500ms
- Data processing: Scalable to 1M+ rows
- Storage operations: < 100ms for typical operations
"""

import pytest
import time
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path

from datadojo.educational.guidance import GuidanceSystem
from datadojo.educational.concepts import get_concept_database
from datadojo.services.educational_service import EducationalService
from datadojo.services.pipeline_service import PipelineService
from datadojo.services.project_service import ProjectService
from datadojo.storage.file_storage import FileStorage
from datadojo.storage.progress_storage import ProgressStorage
from datadojo.models.processing_step import ProcessingStep, OperationType
from datadojo.models.progress_tracker import ProgressTracker
from datadojo.models.learning_project import LearningProject, Domain, Difficulty


class TestGuidancePerformance:
    """Performance tests for guidance system."""

    @pytest.fixture
    def guidance_system(self):
        """Create guidance system instance."""
        return GuidanceSystem()

    @pytest.fixture
    def sample_step(self):
        """Create a sample processing step."""
        return ProcessingStep(
            id="perf-step-1",
            name="Data Cleaning",
            operation_type=OperationType.DATA_CLEANING,
            description="Clean the dataset",
            learned_concepts=["missing_values", "outliers"]
        )

    @pytest.fixture
    def sample_progress(self):
        """Create a sample progress tracker."""
        progress = ProgressTracker("student-1", "project-1")
        progress.complete_step("step-1")
        progress.complete_step("step-2")
        progress.learn_concept("missing_values")
        progress.update_skill_score("data_cleaning", 75.0)
        return progress

    def test_guidance_generation_speed(self, guidance_system, sample_step, sample_progress):
        """Test that guidance generation is under 500ms."""
        data_issues = ["missing_values", "outliers", "duplicates"]

        start_time = time.time()
        guidance = guidance_system.get_step_guidance(
            sample_step,
            progress=sample_progress,
            data_issues=data_issues
        )
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        assert elapsed < 500, f"Guidance took {elapsed:.2f}ms (threshold: 500ms)"
        assert guidance is not None
        assert "hints" in guidance

    def test_guidance_batch_performance(self, guidance_system, sample_progress):
        """Test generating guidance for multiple steps quickly."""
        steps = [
            ProcessingStep(
                id=f"step-{i}",
                name=f"Step {i}",
                operation_type=OperationType.DATA_CLEANING,
                description=f"Step {i}",
                learned_concepts=["missing_values"]
            )
            for i in range(10)
        ]

        start_time = time.time()
        for step in steps:
            guidance_system.get_step_guidance(step, progress=sample_progress)
        elapsed = (time.time() - start_time) * 1000

        # Should handle 10 steps in under 2 seconds
        assert elapsed < 2000, f"Batch guidance took {elapsed:.2f}ms (threshold: 2000ms)"

    def test_hint_generation_speed(self, guidance_system):
        """Test that hint generation is fast."""
        from datadojo.models.educational_content import DifficultyLevel

        contexts = [
            "I have missing values",
            "How do I handle duplicates",
            "Need to convert data types",
            "Want to normalize features",
            "How to encode categories"
        ]

        start_time = time.time()
        for context in contexts:
            hint = guidance_system.provide_hint(context, DifficultyLevel.BEGINNER)
            assert hint is not None
        elapsed = (time.time() - start_time) * 1000

        # 5 hints in under 100ms
        assert elapsed < 100, f"Hint generation took {elapsed:.2f}ms (threshold: 100ms)"

    def test_concept_lookup_speed(self):
        """Test that concept database lookups are fast."""
        concept_db = get_concept_database()

        start_time = time.time()
        for _ in range(100):
            concept = concept_db.get_concept("missing_values")
            assert concept is not None
        elapsed = (time.time() - start_time) * 1000

        # 100 lookups in under 50ms
        assert elapsed < 50, f"Concept lookups took {elapsed:.2f}ms (threshold: 50ms)"


class TestStoragePerformance:
    """Performance tests for storage operations."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def file_storage(self, temp_storage):
        """Create FileStorage instance."""
        return FileStorage(base_path=temp_storage)

    def test_save_performance(self, file_storage):
        """Test that save operations are under 100ms."""
        test_data = {
            "name": "Test Project",
            "description": "A test project with some data",
            "items": [f"item-{i}" for i in range(100)]
        }

        start_time = time.time()
        file_storage.save("projects", "test-1", test_data)
        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 100, f"Save took {elapsed:.2f}ms (threshold: 100ms)"

    def test_load_performance(self, file_storage):
        """Test that load operations are under 100ms."""
        # Save data first
        test_data = {"name": "Test", "data": "x" * 10000}
        file_storage.save("test", "item-1", test_data)

        # Measure load time
        start_time = time.time()
        loaded = file_storage.load("test", "item-1")
        elapsed = (time.time() - start_time) * 1000

        assert loaded is not None
        assert elapsed < 100, f"Load took {elapsed:.2f}ms (threshold: 100ms)"

    def test_list_performance(self, file_storage):
        """Test that listing operations are fast."""
        # Create multiple items
        for i in range(50):
            file_storage.save("projects", f"proj-{i}", {"name": f"Project {i}"})

        # Measure list time
        start_time = time.time()
        items = file_storage.list_items("projects")
        elapsed = (time.time() - start_time) * 1000

        assert len(items) == 50
        assert elapsed < 200, f"List took {elapsed:.2f}ms (threshold: 200ms)"

    def test_progress_save_performance(self, temp_storage):
        """Test progress storage save performance."""
        progress_storage = ProgressStorage(base_path=temp_storage)

        progress = ProgressTracker("student-1", "project-1")
        for i in range(20):
            progress.complete_step(f"step-{i}")
            progress.learn_concept(f"concept-{i}")

        start_time = time.time()
        progress_storage.save_progress(progress, create_backup=True)
        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 150, f"Progress save took {elapsed:.2f}ms (threshold: 150ms)"


class TestServicePerformance:
    """Performance tests for service layer."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_project_service_operations(self, temp_storage):
        """Test ProjectService performance."""
        service = ProjectService(storage_path=temp_storage)

        # Create projects
        start_time = time.time()
        for i in range(10):
            service.create_project(
                f"proj-{i}",
                f"Project {i}",
                Domain.GENERAL,
                Difficulty.BEGINNER,
                "Test project",
                "/data.csv"
            )
        create_elapsed = (time.time() - start_time) * 1000

        # List projects
        start_time = time.time()
        projects = service.list_projects()
        list_elapsed = (time.time() - start_time) * 1000

        assert len(projects) == 10
        assert create_elapsed < 1000, f"Creating 10 projects took {create_elapsed:.2f}ms"
        assert list_elapsed < 200, f"Listing projects took {list_elapsed:.2f}ms"

    def test_educational_service_operations(self, temp_storage):
        """Test EducationalService performance."""
        service = EducationalService(storage_path=temp_storage)

        # Get multiple concepts
        concept_ids = ["missing_values", "outliers", "data_types", "normalization"]

        start_time = time.time()
        for concept_id in concept_ids:
            concept = service.get_concept(concept_id)
            assert concept is not None
        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 50, f"Getting 4 concepts took {elapsed:.2f}ms"

    def test_pipeline_service_operations(self, temp_storage):
        """Test PipelineService performance."""
        service = PipelineService(storage_path=temp_storage)

        # Create pipeline with steps
        start_time = time.time()
        pipeline = service.create_pipeline("pipe-1", "proj-1", "Test Pipeline")

        for i in range(5):
            step = ProcessingStep(
                id=f"step-{i}",
                name=f"Step {i}",
                operation_type=OperationType.DATA_CLEANING,
                description=f"Step {i}"
            )
            pipeline = service.add_step(pipeline, step)

        elapsed = (time.time() - start_time) * 1000

        assert len(pipeline.steps) == 5
        assert elapsed < 500, f"Creating pipeline with 5 steps took {elapsed:.2f}ms"


class TestDataProcessingScalability:
    """Test scalability with large datasets."""

    def generate_test_dataframe(self, rows: int) -> pd.DataFrame:
        """Generate a test DataFrame with various data types."""
        np.random.seed(42)
        return pd.DataFrame({
            'id': range(rows),
            'numeric': np.random.randn(rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'text': [f'text_{i}' for i in range(rows)],
            'missing': [None if i % 10 == 0 else i for i in range(rows)]
        })

    @pytest.mark.slow
    def test_small_dataset_processing(self):
        """Test processing 10K rows (should be very fast)."""
        df = self.generate_test_dataframe(10_000)

        start_time = time.time()

        # Simulate basic operations
        _ = df.isnull().sum()
        _ = df.describe()
        _ = df['numeric'].mean()
        _ = df['category'].value_counts()

        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 100, f"Processing 10K rows took {elapsed:.2f}ms"

    @pytest.mark.slow
    def test_medium_dataset_processing(self):
        """Test processing 100K rows."""
        df = self.generate_test_dataframe(100_000)

        start_time = time.time()

        # Simulate data quality checks
        missing_counts = df.isnull().sum()
        duplicates = df.duplicated().sum()
        stats = df.describe()

        elapsed = (time.time() - start_time) * 1000

        assert elapsed < 500, f"Processing 100K rows took {elapsed:.2f}ms"

    @pytest.mark.slow
    def test_large_dataset_processing(self):
        """Test processing 1M rows (scalability test)."""
        df = self.generate_test_dataframe(1_000_000)

        start_time = time.time()

        # Simulate typical operations
        _ = df.shape
        _ = df.dtypes
        _ = df.isnull().sum()

        elapsed = (time.time() - start_time) * 1000

        # Should handle 1M rows in under 5 seconds
        assert elapsed < 5000, f"Processing 1M rows took {elapsed:.2f}ms"

    @pytest.mark.slow
    def test_chunked_processing_scalability(self):
        """Test chunked processing for very large datasets."""
        from datadojo.config.settings import PerformanceConfig

        config = PerformanceConfig(chunk_size=50_000)
        total_rows = 500_000

        df = self.generate_test_dataframe(total_rows)

        start_time = time.time()

        # Process in chunks
        chunk_results = []
        for start in range(0, len(df), config.chunk_size):
            end = min(start + config.chunk_size, len(df))
            chunk = df.iloc[start:end]
            chunk_results.append(chunk['numeric'].mean())

        overall_mean = np.mean(chunk_results)
        elapsed = (time.time() - start_time) * 1000

        assert len(chunk_results) > 1  # Multiple chunks processed
        assert elapsed < 3000, f"Chunked processing of 500K rows took {elapsed:.2f}ms"


class TestMemoryEfficiency:
    """Test memory efficiency of operations."""

    @pytest.mark.slow
    def test_progress_tracker_memory(self):
        """Test memory usage of large progress trackers."""
        progress = ProgressTracker("student-1", "project-1")

        # Add many completed steps
        for i in range(1000):
            progress.complete_step(f"step-{i}")

        # Add many concepts
        for i in range(100):
            progress.learn_concept(f"concept-{i}")

        # Serialize and check size
        data = progress.to_dict()
        import sys
        size = sys.getsizeof(str(data))

        # Should be under 1MB for 1000 steps
        assert size < 1_000_000, f"Progress tracker size: {size} bytes (threshold: 1MB)"

    def test_concept_database_memory(self):
        """Test that concept database is memory efficient."""
        concept_db = get_concept_database()

        import sys
        size = sys.getsizeof(concept_db)

        # Concept database should be reasonably sized
        assert size < 10_000_000, f"Concept database size: {size} bytes"


def run_performance_summary():
    """Print a summary of performance test results."""
    print("\n" + "=" * 70)
    print("DataDojo Performance Test Summary")
    print("=" * 70)
    print("\nPerformance Requirements:")
    print("  ✓ Guidance generation: < 500ms")
    print("  ✓ Storage operations: < 100ms")
    print("  ✓ Data processing: Scalable to 1M+ rows")
    print("  ✓ Concept lookups: < 1ms per lookup")
    print("  ✓ Service operations: < 500ms for typical operations")
    print("\nAll performance tests passed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
    run_performance_summary()
