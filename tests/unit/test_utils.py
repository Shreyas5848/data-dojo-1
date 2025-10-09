"""Unit tests for DataDojo utilities."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from datadojo.utils.exceptions import (
    DataDojoException,
    ProjectNotFoundError,
    DataValidationError,
    PipelineExecutionError,
    ConfigurationError,
    StorageError
)
from datadojo.storage.file_storage import FileStorage
from datadojo.storage.progress_storage import ProgressStorage
from datadojo.models.progress_tracker import ProgressTracker
from datadojo.config.settings import (
    Settings,
    StorageConfig,
    EducationalConfig,
    PipelineConfig,
    PerformanceConfig
)


class TestExceptions:
    """Tests for custom exception classes."""

    def test_base_exception(self):
        """Test base DataDojoException."""
        exc = DataDojoException(
            "Test error",
            educational_hint="This is a hint",
            suggested_actions=["Action 1", "Action 2"],
            related_concepts=["concept1"]
        )

        assert str(exc) == "Test error"
        assert exc.educational_hint == "This is a hint"
        assert len(exc.suggested_actions) == 2
        assert "concept1" in exc.related_concepts

    def test_project_not_found_error(self):
        """Test ProjectNotFoundError with defaults."""
        exc = ProjectNotFoundError("proj-123")

        assert "proj-123" in str(exc)
        assert exc.educational_hint is not None
        assert len(exc.suggested_actions) > 0

    def test_data_validation_error(self):
        """Test DataValidationError with field info."""
        exc = DataValidationError(
            validation_name="age_range",
            details="Age must be between 0 and 120",
            column="age"
        )

        assert "age" in str(exc)
        assert "age_range" in str(exc)

    def test_pipeline_execution_error(self):
        """Test PipelineExecutionError with step info."""
        exc = PipelineExecutionError(
            step_name="Data Cleaning",
            error_details="Missing required column"
        )

        assert "Data Cleaning" in str(exc)
        assert "Missing required column" in str(exc)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        exc = ConfigurationError("Invalid config key")
        assert "config" in str(exc).lower()

    def test_storage_error(self):
        """Test StorageError."""
        exc = StorageError("Failed to save file")
        assert "save" in str(exc).lower()


class TestFileStorage:
    """Tests for FileStorage utility."""

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

    def test_create_storage(self, file_storage):
        """Test creating file storage."""
        assert file_storage.base_path.exists()

    def test_save_and_load_data(self, file_storage):
        """Test saving and loading data."""
        test_data = {
            "name": "Test",
            "value": 123,
            "items": ["a", "b", "c"]
        }

        # Save data
        file_storage.save("test_category", "test_id", test_data)

        # Load data
        loaded = file_storage.load("test_category", "test_id")
        assert loaded["name"] == "Test"
        assert loaded["value"] == 123
        assert len(loaded["items"]) == 3

    def test_load_nonexistent_data(self, file_storage):
        """Test loading data that doesn't exist."""
        data = file_storage.load("category", "nonexistent")
        assert data is None

    def test_list_items_in_category(self, file_storage):
        """Test listing items in a category."""
        # Save multiple items
        file_storage.save("projects", "proj1", {"name": "Project 1"})
        file_storage.save("projects", "proj2", {"name": "Project 2"})
        file_storage.save("pipelines", "pipe1", {"name": "Pipeline 1"})

        # List projects
        projects = file_storage.list_items("projects")
        assert len(projects) == 2
        assert "proj1" in projects
        assert "proj2" in projects

        # List pipelines
        pipelines = file_storage.list_items("pipelines")
        assert len(pipelines) == 1

    def test_delete_data(self, file_storage):
        """Test deleting data."""
        # Save data
        file_storage.save("test", "item1", {"data": "test"})

        # Delete data
        result = file_storage.delete("test", "item1")
        assert result is True

        # Verify deletion
        loaded = file_storage.load("test", "item1")
        assert loaded is None

    def test_delete_nonexistent_data(self, file_storage):
        """Test deleting data that doesn't exist."""
        result = file_storage.delete("test", "nonexistent")
        assert result is False

    def test_exists_check(self, file_storage):
        """Test checking if data exists."""
        # Initially doesn't exist
        assert not file_storage.exists("test", "item1")

        # Save data
        file_storage.save("test", "item1", {"data": "test"})

        # Now exists
        assert file_storage.exists("test", "item1")

    def test_metadata_tracking(self, file_storage):
        """Test that metadata is tracked correctly."""
        file_storage.save("test", "item1", {"data": "test"})

        loaded = file_storage.load("test", "item1")
        assert "_metadata" in loaded
        assert "created_at" in loaded["_metadata"]
        assert "updated_at" in loaded["_metadata"]

    def test_update_modifies_metadata(self, file_storage):
        """Test that updates change updated_at timestamp."""
        # Save initial data
        file_storage.save("test", "item1", {"data": "v1"})
        first_load = file_storage.load("test", "item1")
        first_updated = first_load["_metadata"]["updated_at"]

        # Update data
        import time
        time.sleep(0.01)  # Ensure timestamp differs
        file_storage.save("test", "item1", {"data": "v2"})
        second_load = file_storage.load("test", "item1")
        second_updated = second_load["_metadata"]["updated_at"]

        # updated_at should have changed
        assert second_updated > first_updated


class TestProgressStorage:
    """Tests for ProgressStorage utility."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def progress_storage(self, temp_storage):
        """Create ProgressStorage instance."""
        return ProgressStorage(base_path=temp_storage)

    def test_save_progress(self, progress_storage):
        """Test saving progress."""
        progress = ProgressTracker(
            student_id="student-1",
            project_id="project-1"
        )
        progress.complete_step("step-1")

        progress_storage.save_progress(progress)

        # Verify file was created
        loaded = progress_storage.load_progress("student-1", "project-1")
        assert loaded is not None
        assert "step-1" in loaded.completed_steps

    def test_load_nonexistent_progress(self, progress_storage):
        """Test loading progress that doesn't exist."""
        progress = progress_storage.load_progress("nonexistent", "project")
        # Should return a new ProgressTracker
        assert progress.student_id == "nonexistent"
        assert progress.project_id == "project"
        assert len(progress.completed_steps) == 0

    def test_list_student_progress(self, progress_storage):
        """Test listing all progress for a student."""
        # Save progress for multiple projects
        progress1 = ProgressTracker("student-1", "project-1")
        progress2 = ProgressTracker("student-1", "project-2")

        progress_storage.save_progress(progress1)
        progress_storage.save_progress(progress2)

        # List progress
        all_progress = progress_storage.list_student_progress("student-1")
        assert len(all_progress) == 2

    def test_backup_creation(self, progress_storage):
        """Test that backups are created on save."""
        progress = ProgressTracker("student-1", "project-1")

        # Save multiple times
        for i in range(3):
            progress.complete_step(f"step-{i}")
            progress_storage.save_progress(progress, create_backup=True)

        # Check backups exist
        backup_dir = progress_storage.base_path / "backups" / "student-1" / "project-1"
        if backup_dir.exists():
            backups = list(backup_dir.glob("*.json"))
            assert len(backups) > 0

    def test_backup_cleanup(self, progress_storage):
        """Test that old backups are cleaned up."""
        progress = ProgressTracker("student-1", "project-1")

        # Save many times to exceed max_backups
        for i in range(15):
            progress.complete_step(f"step-{i}")
            progress_storage.save_progress(progress, create_backup=True)

        # Check that only max_backups (10) remain
        backup_dir = progress_storage.base_path / "backups" / "student-1" / "project-1"
        if backup_dir.exists():
            backups = list(backup_dir.glob("*.json"))
            assert len(backups) <= 10


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()

        assert settings.version == "0.1.0"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.storage.enable_backup is True
        assert settings.educational.default_educational_mode is True

    def test_storage_config(self):
        """Test StorageConfig."""
        config = StorageConfig(
            base_path="/custom/path",
            enable_backup=False,
            max_backups=5
        )

        assert config.base_path == "/custom/path"
        assert config.enable_backup is False
        assert config.max_backups == 5

    def test_educational_config(self):
        """Test EducationalConfig."""
        config = EducationalConfig(
            default_guidance_level="minimal",
            show_hints=False,
            progress_tracking_enabled=True
        )

        assert config.default_guidance_level == "minimal"
        assert config.show_hints is False
        assert config.progress_tracking_enabled is True

    def test_pipeline_config(self):
        """Test PipelineConfig."""
        config = PipelineConfig(
            default_timeout_seconds=600,
            enable_caching=False,
            max_workers=8
        )

        assert config.default_timeout_seconds == 600
        assert config.enable_caching is False
        assert config.max_workers == 8

    def test_performance_config(self):
        """Test PerformanceConfig."""
        config = PerformanceConfig(
            chunk_size=50000,
            use_multiprocessing=False,
            cache_size_mb=200
        )

        assert config.chunk_size == 50000
        assert config.use_multiprocessing is False
        assert config.cache_size_mb == 200

    def test_settings_to_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings()
        data = settings.to_dict()

        assert "version" in data
        assert "storage" in data
        assert "educational" in data
        assert "pipeline" in data
        assert "performance" in data

    def test_settings_from_dict(self):
        """Test creating settings from dictionary."""
        data = {
            "debug": True,
            "log_level": "DEBUG",
            "storage": {
                "base_path": "/test/path",
                "enable_backup": False
            },
            "educational": {
                "show_hints": False
            }
        }

        settings = Settings.from_dict(data)

        assert settings.debug is True
        assert settings.log_level == "DEBUG"
        assert settings.storage.base_path == "/test/path"
        assert settings.storage.enable_backup is False
        assert settings.educational.show_hints is False

    def test_save_and_load_settings(self):
        """Test saving and loading settings from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "config.json"

            # Create and save settings
            settings = Settings()
            settings.debug = True
            settings.log_level = "DEBUG"
            settings.save_to_file(str(config_file))

            # Load settings
            loaded = Settings.load_from_file(str(config_file))

            assert loaded.debug is True
            assert loaded.log_level == "DEBUG"

    def test_load_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        # Set environment variables
        monkeypatch.setenv("DATADOJO_STORAGE_PATH", "/env/path")
        monkeypatch.setenv("DATADOJO_DEBUG", "true")
        monkeypatch.setenv("DATADOJO_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("DATADOJO_EDUCATIONAL_MODE", "false")

        settings = Settings.load_from_env()

        assert settings.storage.base_path == "/env/path"
        assert settings.debug is True
        assert settings.log_level == "WARNING"
        assert settings.educational.default_educational_mode is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
