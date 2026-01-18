"""
MLflow Model Registry Wrapper

Provides high-level interface for model versioning, registration, and retrieval.
Handles model lifecycle management including staging transitions and metadata.

Features:
- Model registration with semantic versioning
- Version comparison and promotion
- Model metadata and lineage tracking
- Stage transitions (None -> Staging -> Production -> Archived)

"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import mlflow

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Wrapper for MLflow Model Registry operations.
    
    Manages:
    - Model registration and versioning
    - Version metadata and lineage
    - Stage transitions
    - Model comparison and promotion
    """
    
    # Model stage constants
    STAGE_NONE = "None"
    STAGE_STAGING = "Staging"
    STAGE_PRODUCTION = "Production"
    STAGE_ARCHIVED = "Archived"
    
    VALID_STAGES = [STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED]
    
    def __init__(self):
        """Initialize registry wrapper with MLflow client."""
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register a model to the registry.
        
        Args:
            model_uri: MLflow run URI (e.g., 'runs:/run_id/path/to/model')
            model_name: Name for the registered model
            description: Optional model description
            metadata: Optional metadata dict
            
        Returns:
            Model version string (e.g., '1', '2', etc.)
        """
        try:
            result = mlflow.register_model(model_uri, model_name)
            version = result.version
            
            logger.info(f"Registered model: {model_name}, version: {version}")
            
            # Add description if provided
            if description:
                self.client.update_registered_model(
                    name=model_name,
                    description=description,
                )
            
            # Add metadata as tags
            if metadata:
                for key, value in metadata.items():
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=str(value),
                    )
            
            return version
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get registered model metadata.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model dict or None if not found
        """
        try:
            return self.client.get_registered_model(model_name)
        except Exception as e:
            logger.warning(f"Model '{model_name}' not found in registry: {e}")
            return None
    
    def get_version(self, model_name: str, version: str) -> Optional[Dict[str, Any]]:
        """
        Get specific model version.
        
        Args:
            model_name: Name of the model
            version: Version number
            
        Returns:
            ModelVersion object or None
        """
        try:
            return self.client.get_model_version(model_name, version)
        except Exception as e:
            logger.error(f"Failed to get model version {model_name}/{version}: {e}")
            return None
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of ModelVersion objects
        """
        try:
            return self.client.list_model_versions(model_name)
        except Exception as e:
            logger.error(f"Failed to list versions for model {model_name}: {e}")
            return []
    
    def transition_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False,
    ) -> None:
        """
        Transition model version to a new stage.
        
        Stages: None -> Staging -> Production -> Archived
        
        Args:
            model_name: Name of the model
            version: Version to transition
            stage: Target stage (Staging, Production, Archived)
            archive_existing_versions: Whether to archive other versions in target stage
        """
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.VALID_STAGES}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )
            logger.info(f"Transitioned {model_name}@{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model version: {e}")
            raise
    
    def promote_to_staging(self, model_name: str, version: str) -> None:
        """Promote a version to Staging."""
        self.transition_stage(model_name, version, self.STAGE_STAGING)
    
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        archive_existing: bool = True,
    ) -> None:
        """Promote a version to Production, optionally archiving previous production version."""
        self.transition_stage(
            model_name,
            version,
            self.STAGE_PRODUCTION,
            archive_existing_versions=archive_existing,
        )
    
    def archive_version(self, model_name: str, version: str) -> None:
        """Archive a model version."""
        self.transition_stage(model_name, version, self.STAGE_ARCHIVED)
    
    def get_production_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the current production version of a model."""
        versions = self.list_versions(model_name)
        prod_versions = [v for v in versions if v.current_stage == self.STAGE_PRODUCTION]
        
        if prod_versions:
            # Return the most recent production version
            return prod_versions[0]
        return None
    
    def set_model_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        """Set metadata tag on a model version."""
        try:
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=value,
            )
            logger.info(f"Set tag {key}={value} on {model_name}@{version}")
        except Exception as e:
            logger.error(f"Failed to set tag: {e}")
            raise
    
    def add_model_version_metadata(
        self,
        model_name: str,
        version: str,
        metadata: Dict[str, str],
    ) -> None:
        """
        Add multiple metadata tags to a model version.
        
        Args:
            model_name: Name of the model
            version: Version number
            metadata: Dictionary of key-value pairs to set as tags
        """
        for key, value in metadata.items():
            self.set_model_tag(model_name, version, key, str(value))
    
    def download_model(
        self,
        model_name: str,
        version: str,
        dst_path: str,
    ) -> str:
        """
        Download a model version to local path.
        
        Args:
            model_name: Name of the model
            version: Version to download
            dst_path: Destination local path
            
        Returns:
            Path to downloaded model
        """
        try:
            result = mlflow.artifacts.download_artifacts(
                artifact_uri=f"models:/{model_name}/{version}",
                dst_path=dst_path,
            )
            logger.info(f"Downloaded {model_name}@{version} to {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def get_version_metadata(self, model_name: str, version: str) -> Dict:
        """
        Get comprehensive metadata for a model version.
        
        Returns:
            Dictionary with version info, stage, tags, etc.
        """
        model_version = self.get_version(model_name, version)
        if not model_version:
            return {}
        
        return {
            "name": model_version.name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "created_timestamp": datetime.fromtimestamp(
                model_version.creation_timestamp / 1000
            ).isoformat(),
            "updated_timestamp": datetime.fromtimestamp(
                model_version.last_updated_timestamp / 1000
            ).isoformat(),
            "run_id": model_version.run_id,
            "source": model_version.source,
            "status": model_version.status,
            "status_message": model_version.status_message,
        }
    
    def compare_versions(
        self,
        model_name: str,
        version1: str,
        version2: str,
    ) -> Dict:
        """
        Compare two model versions.
        
        Args:
            model_name: Name of the model
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Dictionary with comparison info
        """
        m1 = self.get_version_metadata(model_name, version1)
        m2 = self.get_version_metadata(model_name, version2)
        
        return {
            "version1": m1,
            "version2": m2,
            "created_time_diff_hours": (
                (m2.get("created_timestamp") or 0) -
                (m1.get("created_timestamp") or 0)
            ) / 3600 if m1.get("created_timestamp") and m2.get("created_timestamp") else None,
        }
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete a registered model and all its versions.
        
        WARNING: This is irreversible.
        
        Args:
            model_name: Name of the model to delete
        """
        try:
            self.client.delete_registered_model(model_name)
            logger.warning(f"Deleted registered model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            raise
    
    def list_all_models(self) -> List[Dict[str, Any]]:
        """Get all registered models."""
        try:
            return self.client.search_registered_models()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    registry = ModelRegistry()
    
    # Example usage
    models = registry.list_all_models()
    print(f"\nRegistered models: {len(models)}")
    for model in models:
        print(f"  - {model.name}")
