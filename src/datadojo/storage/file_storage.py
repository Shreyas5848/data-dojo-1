"""File-based storage system for DataDojo.

Provides persistent storage for projects, datasets, and configurations
using JSON files on the filesystem.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime


class FileStorage:
    """File-based storage system for DataDojo data.

    Stores data as JSON files in a structured directory hierarchy.
    """

    def __init__(self, base_path: Optional[str] = None):
        """Initialize FileStorage.

        Args:
            base_path: Base directory for storage (defaults to ~/.datadojo)
        """
        if base_path is None:
            base_path = str(Path.home() / ".datadojo")

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Create standard subdirectories
        self.projects_dir = self.base_path / "projects"
        self.progress_dir = self.base_path / "progress"
        self.datasets_dir = self.base_path / "datasets"
        self.content_dir = self.base_path / "content"
        self.config_dir = self.base_path / "config"

        for directory in [
            self.projects_dir,
            self.progress_dir,
            self.datasets_dir,
            self.content_dir,
            self.config_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def save(self, category: str, key: str, data: Dict[str, Any]) -> bool:
        """Save data to storage.

        Args:
            category: Storage category (projects, progress, etc.)
            key: Unique identifier for the data
            data: Data to save (must be JSON serializable)

        Returns:
            True if saved successfully

        Raises:
            ValueError: If category is invalid
        """
        category_path = self._get_category_path(category)
        if not category_path:
            raise ValueError(f"Invalid storage category: {category}")

        file_path = category_path / f"{key}.json"

        # Add metadata
        data_with_meta = {
            "data": data,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(data_with_meta, f, indent=2)
            return True
        except Exception:
            return False

    def load(self, category: str, key: str) -> Optional[Dict[str, Any]]:
        """Load data from storage.

        Args:
            category: Storage category
            key: Unique identifier for the data

        Returns:
            Loaded data or None if not found
        """
        category_path = self._get_category_path(category)
        if not category_path:
            return None

        file_path = category_path / f"{key}.json"
        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                data_with_meta = json.load(f)
                return data_with_meta.get("data")
        except Exception:
            return None

    def delete(self, category: str, key: str) -> bool:
        """Delete data from storage.

        Args:
            category: Storage category
            key: Unique identifier for the data

        Returns:
            True if deleted successfully
        """
        category_path = self._get_category_path(category)
        if not category_path:
            return False

        file_path = category_path / f"{key}.json"
        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except Exception:
            return False

    def list_keys(self, category: str) -> List[str]:
        """List all keys in a category.

        Args:
            category: Storage category

        Returns:
            List of keys (without .json extension)
        """
        category_path = self._get_category_path(category)
        if not category_path:
            return []

        try:
            return [
                f.stem for f in category_path.glob("*.json")
            ]
        except Exception:
            return []

    def exists(self, category: str, key: str) -> bool:
        """Check if data exists in storage.

        Args:
            category: Storage category
            key: Unique identifier for the data

        Returns:
            True if data exists
        """
        category_path = self._get_category_path(category)
        if not category_path:
            return False

        file_path = category_path / f"{key}.json"
        return file_path.exists()

    def update(self, category: str, key: str, data: Dict[str, Any]) -> bool:
        """Update existing data in storage.

        Args:
            category: Storage category
            key: Unique identifier for the data
            data: Updated data

        Returns:
            True if updated successfully
        """
        if not self.exists(category, key):
            return False

        # Load existing metadata
        category_path = self._get_category_path(category)
        file_path = category_path / f"{key}.json"

        try:
            with open(file_path, 'r') as f:
                existing = json.load(f)

            # Preserve created_at, update updated_at
            metadata = existing.get("metadata", {})
            metadata["updated_at"] = datetime.now().isoformat()

            data_with_meta = {
                "data": data,
                "metadata": metadata
            }

            with open(file_path, 'w') as f:
                json.dump(data_with_meta, f, indent=2)
            return True
        except Exception:
            return False

    def clear_category(self, category: str) -> bool:
        """Clear all data in a category.

        Args:
            category: Storage category to clear

        Returns:
            True if cleared successfully
        """
        category_path = self._get_category_path(category)
        if not category_path:
            return False

        try:
            for file_path in category_path.glob("*.json"):
                file_path.unlink()
            return True
        except Exception:
            return False

    def _get_category_path(self, category: str) -> Optional[Path]:
        """Get path for a storage category.

        Args:
            category: Storage category name

        Returns:
            Path to category directory or None if invalid
        """
        category_map = {
            "projects": self.projects_dir,
            "progress": self.progress_dir,
            "datasets": self.datasets_dir,
            "content": self.content_dir,
            "config": self.config_dir
        }
        return category_map.get(category)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get information about storage usage.

        Returns:
            Dictionary with storage statistics
        """
        info = {
            "base_path": str(self.base_path),
            "categories": {}
        }

        for category in ["projects", "progress", "datasets", "content", "config"]:
            category_path = self._get_category_path(category)
            if category_path:
                files = list(category_path.glob("*.json"))
                total_size = sum(f.stat().st_size for f in files)
                info["categories"][category] = {
                    "count": len(files),
                    "size_bytes": total_size
                }

        return info
