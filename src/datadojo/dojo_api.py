from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum

class Domain(Enum):
    ECOMMERCE = "ecommerce"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    GENERAL = "general"

class DifficultyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class GuidanceLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    DETAILED = "detailed"

from dataclasses import dataclass

@dataclass
class ProjectInfo:
    id: str
    name: str
    domain: Domain
    difficulty: DifficultyLevel
    description: str
    estimated_time_minutes: int
    learning_objectives: List[str]

@dataclass
class ExecutionResult:
    success: bool
    processed_data: Any
    execution_time_ms: int
    steps_completed: int
    concepts_learned: List[str]
    error_message: Optional[str]

class PipelineInterface(ABC):
    @abstractmethod
    def add_step(self, operation_type: str, interactive: bool = False, **kwargs) -> 'PipelineInterface':
        pass

    @abstractmethod
    def execute(self, data: Any) -> ExecutionResult:
        pass

    @abstractmethod
    def get_available_operations(self) -> List[str]:
        pass

    @abstractmethod
    def preview_next_step(self) -> Dict[str, Any]:
        pass

class ProjectInterface(ABC):
    @property
    @abstractmethod
    def info(self) -> ProjectInfo:
        pass

    @property
    @abstractmethod
    def dataset(self) -> Any:
        pass

    @abstractmethod
    def create_pipeline(self, guidance_level: GuidanceLevel = GuidanceLevel.DETAILED) -> 'PipelineInterface':
        pass

    @abstractmethod
    def get_progress(self, student_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def track_progress(self, student_id: str, completed_step: str, concepts_learned: List[str]) -> None:
        pass

class DojoInterface(ABC):
    @abstractmethod
    def list_projects(self, domain: Optional[Domain] = None, difficulty: Optional[DifficultyLevel] = None) -> List[ProjectInfo]:
        pass

    @abstractmethod
    def load_project(self, project_id: str, difficulty: Optional[DifficultyLevel] = None) -> 'ProjectInterface':
        pass

    @abstractmethod
    def get_educational_interface(self) -> 'EducationalInterface':
        pass

class EducationalInterface(ABC):
    @abstractmethod
    def get_concept_explanation(self, concept_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_step_guidance(self, step_context: Dict[str, Any]) -> Dict[str, Any]:
        pass
