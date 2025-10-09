"""DomainService for DataDojo framework.

Loads and manages domain-specific modules and configurations.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import json

from ..models.domain_module import DomainModule, OperationDefinition
from ..models.learning_project import Domain


class DomainService:
    """Service for loading and managing domain-specific modules.

    Provides access to domain configurations, custom operations, and
    validation rules for different subject areas.
    """

    def __init__(self, modules_path: Optional[str] = None):
        """Initialize the DomainService.

        Args:
            modules_path: Path to domain modules storage
        """
        if modules_path is None:
            modules_path = str(Path.home() / ".datadojo" / "domains")

        self.modules_path = Path(modules_path)
        self.modules_path.mkdir(parents=True, exist_ok=True)
        self._modules_cache: Dict[str, DomainModule] = {}
        self._load_modules()

    def _load_modules(self) -> None:
        """Load all domain modules from storage."""
        self._modules_cache.clear()

        if not self.modules_path.exists():
            return

        for module_file in self.modules_path.glob("*.json"):
            try:
                with open(module_file, 'r') as f:
                    data = json.load(f)
                    module = DomainModule.from_dict(data)
                    self._modules_cache[module.domain_name] = module
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Warning: Failed to load module from {module_file}: {e}")

    def _save_module(self, module: DomainModule) -> None:
        """Save a domain module to storage.

        Args:
            module: DomainModule to save
        """
        module_file = self.modules_path / f"{module.domain_name}.json"
        with open(module_file, 'w') as f:
            json.dump(module.to_dict(), f, indent=2)

    def register_domain(self, module: DomainModule) -> DomainModule:
        """Register a new domain module.

        Args:
            module: DomainModule to register

        Returns:
            Registered DomainModule

        Raises:
            ValueError: If domain with same name already exists
        """
        if module.domain_name in self._modules_cache:
            raise ValueError(f"Domain '{module.domain_name}' already registered")

        self._modules_cache[module.domain_name] = module
        self._save_module(module)
        return module

    def get_domain(self, domain_name: str) -> Optional[DomainModule]:
        """Get a domain module by name.

        Args:
            domain_name: Name of the domain to retrieve

        Returns:
            DomainModule if found, None otherwise
        """
        return self._modules_cache.get(domain_name)

    def get_domain_by_enum(self, domain: Domain) -> Optional[DomainModule]:
        """Get a domain module by Domain enum.

        Args:
            domain: Domain enum value

        Returns:
            DomainModule if found, None otherwise
        """
        return self.get_domain(domain.value)

    def list_domains(self) -> List[DomainModule]:
        """List all registered domain modules.

        Returns:
            List of all DomainModules
        """
        return list(self._modules_cache.values())

    def update_domain(self, module: DomainModule) -> DomainModule:
        """Update an existing domain module.

        Args:
            module: DomainModule with updated data

        Returns:
            Updated DomainModule

        Raises:
            ValueError: If domain does not exist
        """
        if module.domain_name not in self._modules_cache:
            raise ValueError(f"Domain '{module.domain_name}' not found")

        self._modules_cache[module.domain_name] = module
        self._save_module(module)
        return module

    def unregister_domain(self, domain_name: str) -> bool:
        """Unregister a domain module.

        Args:
            domain_name: Name of the domain to unregister

        Returns:
            True if domain was removed, False if not found
        """
        if domain_name not in self._modules_cache:
            return False

        del self._modules_cache[domain_name]

        module_file = self.modules_path / f"{domain_name}.json"
        if module_file.exists():
            module_file.unlink()

        return True

    def get_domain_operations(self, domain_name: str) -> List[OperationDefinition]:
        """Get all operations for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            List of OperationDefinitions for the domain
        """
        module = self.get_domain(domain_name)
        if not module:
            return []

        return module.domain_specific_operations

    def get_operation(self, domain_name: str, operation_name: str) -> Optional[OperationDefinition]:
        """Get a specific operation from a domain.

        Args:
            domain_name: Name of the domain
            operation_name: Name of the operation

        Returns:
            OperationDefinition if found, None otherwise
        """
        module = self.get_domain(domain_name)
        if not module:
            return None

        return module.get_operation(operation_name)

    def add_operation_to_domain(
        self,
        domain_name: str,
        operation: OperationDefinition
    ) -> bool:
        """Add an operation to a domain.

        Args:
            domain_name: Name of the domain
            operation: OperationDefinition to add

        Returns:
            True if operation was added, False if domain not found
        """
        module = self.get_domain(domain_name)
        if not module:
            return False

        module.add_operation(operation)
        self._save_module(module)
        return True

    def get_validation_rules(self, domain_name: str) -> List[Dict[str, Any]]:
        """Get validation rules for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            List of validation rules
        """
        module = self.get_domain(domain_name)
        if not module:
            return []

        return module.validation_rules

    def add_validation_rule(
        self,
        domain_name: str,
        rule: Dict[str, Any]
    ) -> bool:
        """Add a validation rule to a domain.

        Args:
            domain_name: Name of the domain
            rule: Validation rule to add

        Returns:
            True if rule was added, False if domain not found
        """
        module = self.get_domain(domain_name)
        if not module:
            return False

        module.add_validation_rule(rule)
        self._save_module(module)
        return True

    def get_projects_for_domain(self, domain_name: str) -> List[str]:
        """Get all project IDs for a domain.

        Args:
            domain_name: Name of the domain

        Returns:
            List of project IDs
        """
        module = self.get_domain(domain_name)
        if not module:
            return []

        return module.available_projects

    def add_project_to_domain(self, domain_name: str, project_id: str) -> bool:
        """Add a project to a domain.

        Args:
            domain_name: Name of the domain
            project_id: ID of the project to add

        Returns:
            True if project was added, False if domain not found
        """
        module = self.get_domain(domain_name)
        if not module:
            return False

        module.add_project(project_id)
        self._save_module(module)
        return True

    def remove_project_from_domain(self, domain_name: str, project_id: str) -> bool:
        """Remove a project from a domain.

        Args:
            domain_name: Name of the domain
            project_id: ID of the project to remove

        Returns:
            True if project was removed, False if not found
        """
        module = self.get_domain(domain_name)
        if not module:
            return False

        removed = module.remove_project(project_id)
        if removed:
            self._save_module(module)
        return removed

    def domain_exists(self, domain_name: str) -> bool:
        """Check if a domain exists.

        Args:
            domain_name: Name of the domain to check

        Returns:
            True if domain exists, False otherwise
        """
        return domain_name in self._modules_cache

    def get_domain_count(self) -> int:
        """Get the total number of registered domains.

        Returns:
            Number of registered domains
        """
        return len(self._modules_cache)

    def initialize_default_domains(self) -> None:
        """Initialize default domain modules if they don't exist.

        Creates basic domain modules for ecommerce, healthcare, and finance
        if they are not already registered.
        """
        default_domains = [
            {
                "domain_name": "ecommerce",
                "display_name": "E-Commerce",
                "description": "Learn data preparation for e-commerce analytics, customer segmentation, and sales forecasting"
            },
            {
                "domain_name": "healthcare",
                "display_name": "Healthcare",
                "description": "Practice data cleaning and feature engineering for healthcare datasets, patient records, and medical analytics"
            },
            {
                "domain_name": "finance",
                "display_name": "Finance",
                "description": "Master data preprocessing for financial analysis, risk assessment, and market predictions"
            }
        ]

        for domain_data in default_domains:
            if not self.domain_exists(domain_data["domain_name"]):
                module = DomainModule(
                    domain_name=domain_data["domain_name"],
                    display_name=domain_data["display_name"],
                    description=domain_data["description"]
                )
                self.register_domain(module)
