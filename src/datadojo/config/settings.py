"""Configuration management for DataDojo.

Handles loading and managing configuration settings from files and
environment variables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class StorageConfig:
    """Storage configuration settings."""
    base_path: str = field(default_factory=lambda: str(Path.home() / ".datadojo"))
    projects_dir: str = "projects"
    progress_dir: str = "progress"
    datasets_dir: str = "datasets"
    content_dir: str = "content"
    enable_backup: bool = True
    max_backups: int = 10


@dataclass
class EducationalConfig:
    """Educational features configuration."""
    default_guidance_level: str = "detailed"
    default_educational_mode: bool = True
    show_hints: bool = True
    show_analogies: bool = True
    show_examples: bool = True
    progress_tracking_enabled: bool = True


@dataclass
class PipelineConfig:
    """Pipeline execution configuration."""
    default_timeout_seconds: int = 300
    enable_caching: bool = True
    max_workers: int = 4
    log_execution: bool = True
    validate_before_execution: bool = True


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    chunk_size: int = 10000
    use_multiprocessing: bool = True
    cache_size_mb: int = 100
    max_dataset_size_mb: int = 1000


@dataclass
class Settings:
    """Main configuration settings for DataDojo."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    educational: EducationalConfig = field(default_factory=EducationalConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # General settings
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    @classmethod
    def load_from_file(cls, config_path: str) -> "Settings":
        """Load settings from a JSON configuration file.

        Args:
            config_path: Path to configuration file

        Returns:
            Settings instance
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create Settings from dictionary.

        Args:
            data: Dictionary with configuration data

        Returns:
            Settings instance
        """
        settings = cls()

        # Load storage config
        if "storage" in data:
            settings.storage = StorageConfig(**data["storage"])

        # Load educational config
        if "educational" in data:
            settings.educational = EducationalConfig(**data["educational"])

        # Load pipeline config
        if "pipeline" in data:
            settings.pipeline = PipelineConfig(**data["pipeline"])

        # Load performance config
        if "performance" in data:
            settings.performance = PerformanceConfig(**data["performance"])

        # Load general settings
        if "debug" in data:
            settings.debug = data["debug"]
        if "log_level" in data:
            settings.log_level = data["log_level"]

        return settings

    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment variables.

        Returns:
            Settings instance configured from environment
        """
        settings = cls()

        # Storage settings from env
        if os.getenv("DATADOJO_STORAGE_PATH"):
            settings.storage.base_path = os.getenv("DATADOJO_STORAGE_PATH")

        # Educational settings from env
        if os.getenv("DATADOJO_GUIDANCE_LEVEL"):
            settings.educational.default_guidance_level = os.getenv("DATADOJO_GUIDANCE_LEVEL")
        if os.getenv("DATADOJO_EDUCATIONAL_MODE"):
            settings.educational.default_educational_mode = (
                os.getenv("DATADOJO_EDUCATIONAL_MODE").lower() == "true"
            )

        # Debug mode from env
        if os.getenv("DATADOJO_DEBUG"):
            settings.debug = os.getenv("DATADOJO_DEBUG").lower() == "true"
        if os.getenv("DATADOJO_LOG_LEVEL"):
            settings.log_level = os.getenv("DATADOJO_LOG_LEVEL")

        return settings

    def save_to_file(self, config_path: str) -> None:
        """Save settings to a JSON configuration file.

        Args:
            config_path: Path where configuration should be saved
        """
        config_data = self.to_dict()
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dictionary with all configuration data
        """
        return {
            "version": self.version,
            "debug": self.debug,
            "log_level": self.log_level,
            "storage": {
                "base_path": self.storage.base_path,
                "projects_dir": self.storage.projects_dir,
                "progress_dir": self.storage.progress_dir,
                "datasets_dir": self.storage.datasets_dir,
                "content_dir": self.storage.content_dir,
                "enable_backup": self.storage.enable_backup,
                "max_backups": self.storage.max_backups
            },
            "educational": {
                "default_guidance_level": self.educational.default_guidance_level,
                "default_educational_mode": self.educational.default_educational_mode,
                "show_hints": self.educational.show_hints,
                "show_analogies": self.educational.show_analogies,
                "show_examples": self.educational.show_examples,
                "progress_tracking_enabled": self.educational.progress_tracking_enabled
            },
            "pipeline": {
                "default_timeout_seconds": self.pipeline.default_timeout_seconds,
                "enable_caching": self.pipeline.enable_caching,
                "max_workers": self.pipeline.max_workers,
                "log_execution": self.pipeline.log_execution,
                "validate_before_execution": self.pipeline.validate_before_execution
            },
            "performance": {
                "chunk_size": self.performance.chunk_size,
                "use_multiprocessing": self.performance.use_multiprocessing,
                "cache_size_mb": self.performance.cache_size_mb,
                "max_dataset_size_mb": self.performance.max_dataset_size_mb
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance.

    Returns:
        Settings singleton instance
    """
    global _settings
    if _settings is None:
        # Try to load from default config file
        default_config_path = Path.home() / ".datadojo" / "config.json"
        if default_config_path.exists():
            _settings = Settings.load_from_file(str(default_config_path))
        else:
            # Load from environment or use defaults
            _settings = Settings.load_from_env()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance.

    Args:
        settings: Settings instance to use globally
    """
    global _settings
    _settings = settings


def reset_settings() -> None:
    """Reset settings to defaults."""
    global _settings
    _settings = None
