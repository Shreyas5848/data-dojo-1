"""Progress tracking persistence for DataDojo.

Manages saving and loading student progress data with backup and recovery.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from .file_storage import FileStorage


class ProgressStorage:
    """Persistent storage for student progress tracking.

    Handles saving, loading, and querying student progress data with
    support for backups and history.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize ProgressStorage.

        Args:
            storage_path: Path for storage (defaults to ~/.datadojo/progress)
        """
        if storage_path is None:
            storage_path = str(Path.home() / ".datadojo" / "progress")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.active_dir = self.storage_path / "active"
        self.history_dir = self.storage_path / "history"
        self.backups_dir = self.storage_path / "backups"

        for directory in [self.active_dir, self.history_dir, self.backups_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.file_storage = FileStorage()

    def save_progress(
        self,
        student_id: str,
        project_id: str,
        progress_data: Dict[str, Any]
    ) -> bool:
        """Save student progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            progress_data: Progress data to save

        Returns:
            True if saved successfully
        """
        key = self._make_key(student_id, project_id)
        file_path = self.active_dir / f"{key}.json"

        # Create backup if file exists
        if file_path.exists():
            self._create_backup(student_id, project_id)

        # Add timestamp
        progress_with_meta = {
            **progress_data,
            "last_saved": datetime.now().isoformat()
        }

        try:
            with open(file_path, 'w') as f:
                json.dump(progress_with_meta, f, indent=2)
            return True
        except Exception:
            return False

    def load_progress(
        self,
        student_id: str,
        project_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load student progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            Progress data or None if not found
        """
        key = self._make_key(student_id, project_id)
        file_path = self.active_dir / f"{key}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    def delete_progress(self, student_id: str, project_id: str) -> bool:
        """Delete student progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            True if deleted successfully
        """
        # Archive before deleting
        self._archive_progress(student_id, project_id)

        key = self._make_key(student_id, project_id)
        file_path = self.active_dir / f"{key}.json"

        if not file_path.exists():
            return False

        try:
            file_path.unlink()
            return True
        except Exception:
            return False

    def list_student_progress(self, student_id: str) -> List[Dict[str, Any]]:
        """List all progress for a student.

        Args:
            student_id: Student identifier

        Returns:
            List of progress records
        """
        progress_list = []

        for file_path in self.active_dir.glob(f"{student_id}_*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract project_id from filename
                    filename = file_path.stem
                    _, project_id = filename.split('_', 1)
                    data['project_id'] = project_id
                    progress_list.append(data)
            except Exception:
                continue

        return progress_list

    def list_project_progress(self, project_id: str) -> List[Dict[str, Any]]:
        """List all progress for a project.

        Args:
            project_id: Project identifier

        Returns:
            List of progress records
        """
        progress_list = []

        for file_path in self.active_dir.glob(f"*_{project_id}.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract student_id from filename
                    filename = file_path.stem
                    student_id, _ = filename.split('_', 1)
                    data['student_id'] = student_id
                    progress_list.append(data)
            except Exception:
                continue

        return progress_list

    def progress_exists(self, student_id: str, project_id: str) -> bool:
        """Check if progress exists.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            True if progress exists
        """
        key = self._make_key(student_id, project_id)
        file_path = self.active_dir / f"{key}.json"
        return file_path.exists()

    def _create_backup(self, student_id: str, project_id: str) -> bool:
        """Create backup of current progress.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            True if backup created successfully
        """
        key = self._make_key(student_id, project_id)
        source_path = self.active_dir / f"{key}.json"

        if not source_path.exists():
            return False

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backups_dir / f"{key}_{timestamp}.json"

        try:
            import shutil
            shutil.copy2(source_path, backup_path)
            # Keep only last 10 backups
            self._cleanup_old_backups(student_id, project_id, keep=10)
            return True
        except Exception:
            return False

    def _archive_progress(self, student_id: str, project_id: str) -> bool:
        """Archive progress to history.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            True if archived successfully
        """
        key = self._make_key(student_id, project_id)
        source_path = self.active_dir / f"{key}.json"

        if not source_path.exists():
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.history_dir / f"{key}_{timestamp}.json"

        try:
            import shutil
            shutil.copy2(source_path, archive_path)
            return True
        except Exception:
            return False

    def _cleanup_old_backups(
        self,
        student_id: str,
        project_id: str,
        keep: int = 10
    ) -> None:
        """Remove old backup files, keeping only the most recent.

        Args:
            student_id: Student identifier
            project_id: Project identifier
            keep: Number of backups to keep
        """
        key = self._make_key(student_id, project_id)
        backups = sorted(
            self.backups_dir.glob(f"{key}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old backups
        for backup in backups[keep:]:
            try:
                backup.unlink()
            except Exception:
                pass

    def _make_key(self, student_id: str, project_id: str) -> str:
        """Create storage key from student and project IDs.

        Args:
            student_id: Student identifier
            project_id: Project identifier

        Returns:
            Combined key string
        """
        return f"{student_id}_{project_id}"

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage stats
        """
        active_count = len(list(self.active_dir.glob("*.json")))
        history_count = len(list(self.history_dir.glob("*.json")))
        backup_count = len(list(self.backups_dir.glob("*.json")))

        return {
            "storage_path": str(self.storage_path),
            "active_progress_count": active_count,
            "archived_count": history_count,
            "backup_count": backup_count
        }
