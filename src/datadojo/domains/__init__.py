"""Domain module loader and registry for DataDojo.

Provides centralized access to domain-specific configurations, projects,
and educational content.
"""

from typing import Dict, List, Optional
from ..models.domain_module import DomainModule
from ..models.learning_project import LearningProject, Domain

# Import domain modules
from . import ecommerce
from . import healthcare
from . import finance


class DomainRegistry:
    """Registry for managing domain modules."""

    def __init__(self):
        """Initialize the domain registry."""
        self._modules: Dict[str, DomainModule] = {}
        self._projects: Dict[str, List[LearningProject]] = {}
        self._load_default_domains()

    def _load_default_domains(self) -> None:
        """Load all default domain modules."""
        # Load e-commerce
        ecommerce_module = ecommerce.get_ecommerce_module()
        self.register_module(ecommerce_module)
        self._projects["ecommerce"] = ecommerce.get_sample_projects()

        # Load healthcare
        healthcare_module = healthcare.get_healthcare_module()
        self.register_module(healthcare_module)
        self._projects["healthcare"] = healthcare.get_sample_projects()

        # Load finance
        finance_module = finance.get_finance_module()
        self.register_module(finance_module)
        self._projects["finance"] = finance.get_sample_projects()

    def register_module(self, module: DomainModule) -> None:
        """Register a domain module.

        Args:
            module: DomainModule to register
        """
        self._modules[module.domain_name] = module

    def get_module(self, domain_name: str) -> Optional[DomainModule]:
        """Get a domain module by name.

        Args:
            domain_name: Name of the domain

        Returns:
            DomainModule if found, None otherwise
        """
        return self._modules.get(domain_name)

    def list_modules(self) -> List[DomainModule]:
        """List all registered domain modules.

        Returns:
            List of all DomainModules
        """
        return list(self._modules.values())

    def get_domain_projects(self, domain_name: str) -> List[LearningProject]:
        """Get all projects for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            List of LearningProjects for the domain
        """
        return self._projects.get(domain_name, [])

    def get_all_projects(self) -> List[LearningProject]:
        """Get all projects across all domains.

        Returns:
            List of all LearningProjects
        """
        all_projects = []
        for projects in self._projects.values():
            all_projects.extend(projects)
        return all_projects

    def get_project_by_id(self, project_id: str) -> Optional[LearningProject]:
        """Get a specific project by ID.

        Args:
            project_id: Project identifier

        Returns:
            LearningProject if found, None otherwise
        """
        for projects in self._projects.values():
            for project in projects:
                if project.id == project_id:
                    return project
        return None

    def get_domain_concepts(self, domain_name: str) -> List[Dict]:
        """Get educational concepts for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            List of concept dictionaries
        """
        if domain_name == "ecommerce":
            return ecommerce.get_domain_concepts()
        elif domain_name == "healthcare":
            return healthcare.get_domain_concepts()
        elif domain_name == "finance":
            return finance.get_domain_concepts()
        return []

    def search_projects(
        self,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[LearningProject]:
        """Search for projects with filters.

        Args:
            domain: Filter by domain name
            difficulty: Filter by difficulty level
            keyword: Search in name and description

        Returns:
            List of matching LearningProjects
        """
        projects = self.get_all_projects()

        if domain:
            projects = [p for p in projects if p.domain.value == domain]

        if difficulty:
            projects = [p for p in projects if p.difficulty.value == difficulty]

        if keyword:
            keyword = keyword.lower()
            projects = [
                p for p in projects
                if keyword in p.name.lower() or keyword in p.description.lower()
            ]

        return projects


# Global registry instance
_registry = None


def get_registry() -> DomainRegistry:
    """Get the global domain registry instance.

    Returns:
        DomainRegistry singleton instance
    """
    global _registry
    if _registry is None:
        _registry = DomainRegistry()
    return _registry


def load_domain_projects(project_service) -> None:
    """Load all domain projects into the project service.

    Args:
        project_service: ProjectService instance to load projects into
    """
    registry = get_registry()
    all_projects = registry.get_all_projects()

    for project in all_projects:
        try:
            if not project_service.project_exists(project.id):
                project_service.create_project(project)
        except Exception:
            # Skip if project already exists or has issues
            pass


__all__ = [
    "DomainRegistry",
    "get_registry",
    "load_domain_projects"
]
