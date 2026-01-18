"""
MLflow Configuration Module (2026 edition)

Handles setup, initialization, and configuration of MLflow for the Markov chain system.

Modernized approach (recommended for MLflow 6gt;= 2.0):
- Uses SQLite backend for metadata (runs, experiments, params, metrics)
- Keeps artifacts on filesystem in a separate directory
- Avoids deprecated filesystem-only tracking
- No more "malformed experiment" warnings from subfolder confusion

"""

import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# Project structure
# ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent

MLFLOW_ROOT       = PROJECT_ROOT / "model_registry"
MLFLOW_DB_PATH    = MLFLOW_ROOT / "mlflow.db"              # SQLite metadata + tracking
MLFLOW_ARTIFACTS  = MLFLOW_ROOT / "artifacts"               # Pure artifact storage only


class MLflowConfig:
    """
    Centralized MLflow configuration 	6 now using SQLite backend everywhere.
    
    Responsibilities:
    - Single SQLite DB for tracking & metadata
    - Separate filesystem folder for artifacts
    - Per-experiment artifact locations
    - Clean experiment & model registry management
    """

    # Pre-defined experiment names (consistency across the project)
    EXPERIMENT_MARKOV_BASELINE     = "markov_chain_baseline"
    EXPERIMENT_MARKOV_ABSORBING    = "markov_chain_absorbing"
    EXPERIMENT_MARKOV_COMPARISON   = "markov_chain_comparison"
    EXPERIMENT_DATA_SENSITIVITY    = "data_sensitivity_analysis"

    # Registry model names
    MODEL_MARKOV_CHAIN      = "markov_chain_model"
    MODEL_ABSORBING_CHAIN   = "absorbing_markov_model"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_root: Optional[str] = None,
        enable_registry: bool = True,
    ):
        """
        Initialize MLflow config 	6 defaults to local SQLite + filesystem artifacts.
        """
        # ─── Core paths ───────────────────────────────────────
        self.mlflow_db_uri = (
            tracking_uri
            or f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
        )

        self.artifact_root = (
            artifact_root
            or MLFLOW_ARTIFACTS.as_posix()
        )

        self.artifact_root_uri = Path(self.artifact_root).as_uri()
        self.enable_registry = enable_registry

        self.client: Optional[MlflowClient] = None
        self.experiments: Dict[str, str] = {}   # name 1269 experiment_id

    def setup(self) -> None:
        """Initialize directories, SQLite DB, client, and experiments."""
        logger.info("Setting up MLflow configuration (SQLite + filesystem artifacts)...")

        # Ensure directories exist
        MLFLOW_ROOT.mkdir(parents=True, exist_ok=True)
        MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)

        logger.info(f"SQLite tracking DB : {self.mlflow_db_uri}")
        logger.info(f"Artifact root      : {self.artifact_root}")

        # ─── Connect MLflow ───────────────────────────────────
        mlflow.set_tracking_uri(self.mlflow_db_uri)
        self.client = MlflowClient(self.mlflow_db_uri)

        # Create / recover experiments
        self._setup_experiments()

        logger.info(f"MLflow ready. {len(self.experiments)} experiments registered.")

    def _setup_experiments(self) -> None:
        """Create or retrieve all known experiments with explicit artifact locations."""
        experiment_names = [
            self.EXPERIMENT_MARKOV_BASELINE,
            self.EXPERIMENT_MARKOV_ABSORBING,
            self.EXPERIMENT_MARKOV_COMPARISON,
            self.EXPERIMENT_DATA_SENSITIVITY,
        ]

        for name in experiment_names:
            try:
                exp = mlflow.get_experiment_by_name(name)
                if exp:
                    self.experiments[name] = exp.experiment_id
                    logger.info(f"Found experiment: {name} (ID: {exp.experiment_id})")
                else:
                    # Explicit artifact URI per experiment -> very reliable on Windows
                    exp_artifact_path = Path(self.artifact_root) / name
                    exp_artifact_uri = exp_artifact_path.as_uri()

                    exp_id = mlflow.create_experiment(
                        name,
                        artifact_location=exp_artifact_uri
                    )
                    self.experiments[name] = exp_id
                    logger.info(f"Created experiment: {name} (ID: {exp_id}) -> artifacts: {exp_artifact_uri}")
            except Exception as e:
                logger.error(f"Failed to setup experiment '{name}': {e}")

    # ─── Public API ─────────────────────────────────────────────

    def get_experiment_id(self, experiment_name: str) -> str:
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        return self.experiments[experiment_name]

    def list_experiments(self) -> Dict[str, str]:
        return self.experiments.copy()

    def get_registry_model_uri(self, model_name: str, version: str = "latest") -> str:
        return f"models:/{model_name}/{version}"

    def get_run_artifact_uri(self, run_id: str) -> str:
        if not self.client:
            raise RuntimeError("MLflow client not initialized. Call setup() first.")
        run = self.client.get_run(run_id)
        return run.info.artifact_uri


# ─── Global singleton ─────────────────────────────────────────────

_mlflow_config: Optional[MLflowConfig] = None


def initialize_mlflow(
    tracking_uri: Optional[str] = None,
    artifact_root: Optional[str] = None,
) -> MLflowConfig:
    global _mlflow_config
    _mlflow_config = MLflowConfig(
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    _mlflow_config.setup()
    return _mlflow_config


def get_mlflow_config() -> MLflowConfig:
    if _mlflow_config is None:
        raise RuntimeError("MLflow not initialized. Call initialize_mlflow() first.")
    return _mlflow_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = initialize_mlflow()
    print("\nMLflow Configuration Summary:")
    print(f"  Tracking DB URI : {config.mlflow_db_uri}")
    print(f"  Artifact Root   : {config.artifact_root}")
    print(f"  Experiments     : {list(config.list_experiments().keys())}")
"""
MLflow Configuration Module

Handles setup, initialization, and configuration of MLflow tracking server,
experiment management, and model registry for the Markov chain system.

Configuration prioritizes:
1. Local file-based tracking (development)
2. Experiment isolation (separate runs per Markov configuration)
3. Artifact versioning (model states, transition matrices, metrics)
4. Registry organization (semantic versioning for registered models)

"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
from mlflow.tracking import MlflowClient

# Configure logging
logger = logging.getLogger(__name__)

# Project structure paths
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "model_registry" / "mlflow"
# Place artifacts in a sibling directory to avoid confusing MLflow's
# filesystem tracking backend which scans the tracking dir for experiment
# folders (each experiment expects a meta.yaml). Using a separate
# `mlflow_artifacts` directory prevents FileStore from treating the
# artifact folder as an experiment and removes the MissingConfigException
# warnings seen earlier.
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "model_registry" / "artifacts"  # Renamed to avoid "artifacts" keyword
MLFLOW_BACKEND_DIR = PROJECT_ROOT / "model_registry" / "db_backend"   # Renamed


class MLflowConfig:
    """
    Centralized MLflow configuration for the Markov chain system.
    
    Manages:
    - Tracking URI setup (local file backend)
    - Experiment creation and registration
    - Artifact storage paths
    - Model registry configuration
    """
    
    # Experiment names (pre-defined for consistency)
    EXPERIMENT_MARKOV_BASELINE = "markov_chain_baseline"
    EXPERIMENT_MARKOV_ABSORBING = "markov_chain_absorbing"
    EXPERIMENT_MARKOV_COMPARISON = "markov_chain_comparison"
    EXPERIMENT_DATA_SENSITIVITY = "data_sensitivity_analysis"
    
    # Registry model names
    MODEL_MARKOV_CHAIN = "markov_chain_model"
    MODEL_ABSORBING_CHAIN = "absorbing_markov_model"
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        artifact_root: Optional[str] = None,
        backend_store_uri: Optional[str] = None,
        enable_registry: bool = True,
    ):
        """
        Initialize MLflow configuration.
        
        Args:
            tracking_uri: URI for MLflow tracking server (default: local file)
            artifact_root: Root directory for artifact storage
            backend_store_uri: Backend store URI for metadata (default: local SQLite)
            enable_registry: Whether to enable model registry
        """
        # Store Path objects and normalize URIs for MLflow
        tracking_dir = MLFLOW_TRACKING_DIR
        artifacts_dir = MLFLOW_ARTIFACTS_DIR
        backend_db = MLFLOW_BACKEND_DIR / 'mlflow.db'

        # Use file:// URIs for tracking and artifact locations so MLflow
        # recognizes the scheme on Windows and other platforms.
        self.tracking_uri = (
            tracking_uri
            or tracking_dir.as_uri()
        )
        # artifact_root as a filesystem path (string) and also keep a URI helper
        self.artifact_root = artifact_root or artifacts_dir.as_posix()
        self.artifact_root_uri = Path(self.artifact_root).as_uri()
        self.backend_store_uri = backend_store_uri or f"sqlite:///{backend_db.as_posix()}"
        self.enable_registry = enable_registry
        
        self.client: Optional[MlflowClient] = None
        self.experiments: Dict[str, str] = {}  # name -> experiment_id
        
    def setup(self) -> None:
        """
        Initialize MLflow tracking and experiment structure.
        
        Creates:
        - Tracking directory structure
        - Experiment registry
        - MLflow client connection
        """
        logger.info("Setting up MLflow configuration...")
        
        # Create necessary directories (tracking dir, separate artifact root, backend DB dir)
        MLFLOW_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        MLFLOW_BACKEND_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Artifact root: {self.artifact_root}")
        logger.info(f"Tracking URI: {self.tracking_uri}")
        logger.info(f"Backend store: {self.backend_store_uri}")
        
        # Set MLflow configuration
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create MLflow client
        self.client = MlflowClient(self.tracking_uri)
        
        # Initialize experiments
        self._setup_experiments()
        
        logger.info(f"MLflow setup complete. {len(self.experiments)} experiments registered.")
        
    def _setup_experiments(self) -> None:
        """Create or retrieve experiment IDs."""
        experiment_names = [
            self.EXPERIMENT_MARKOV_BASELINE,
            self.EXPERIMENT_MARKOV_ABSORBING,
            self.EXPERIMENT_MARKOV_COMPARISON,
            self.EXPERIMENT_DATA_SENSITIVITY,
        ]
        
        for exp_name in experiment_names:
            try:
                # Try to get existing experiment
                exp = mlflow.get_experiment_by_name(exp_name)
                if exp:
                    self.experiments[exp_name] = exp.experiment_id
                    logger.info(f"Retrieved experiment: {exp_name} (ID: {exp.experiment_id})")
                else:
                    # Create new experiment if not found. Ensure MLflow receives
                    # a file:// artifact location so the artifact repository
                    # resolves correctly on Windows.
                    try:
                        artifact_loc = Path(self.artifact_root).joinpath(exp_name).as_uri()
                    except Exception:
                        artifact_loc = f"file:///{self.artifact_root}/{exp_name}"

                    exp_id = mlflow.create_experiment(
                        exp_name,
                        artifact_location=artifact_loc,
                    )
                    self.experiments[exp_name] = exp_id
                    logger.info(f"Created experiment: {exp_name} (ID: {exp_id})")
            except Exception as e:
                logger.error(f"Failed to setup experiment {exp_name}: {e}")
    
    def get_experiment_id(self, experiment_name: str) -> str:
        """
        Get experiment ID by name.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Experiment ID
            
        Raises:
            ValueError: If experiment not found
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment '{experiment_name}' not registered")
        return self.experiments[experiment_name]
    
    def list_experiments(self) -> Dict[str, str]:
        """Get all registered experiments."""
        return self.experiments.copy()
    
    def get_registry_model_uri(self, model_name: str, version: str = "latest") -> str:
        """
        Get URI for a model in the registry.
        
        Args:
            model_name: Name of the registered model
            version: Model version (default: "latest")
            
        Returns:
            MLflow model URI
        """
        return f"models:/{model_name}/{version}"
    
    def get_run_artifact_uri(self, run_id: str) -> str:
        """Get artifact URI for a run."""
        if not self.client:
            raise RuntimeError("MLflow not initialized. Call setup() first.")
        run = self.client.get_run(run_id)
        return run.info.artifact_uri


# Global configuration instance
_mlflow_config: Optional[MLflowConfig] = None


def initialize_mlflow(
    tracking_uri: Optional[str] = None,
    artifact_root: Optional[str] = None,
) -> MLflowConfig:
    """
    Initialize and configure MLflow globally.
    
    Args:
        tracking_uri: Optional custom tracking URI
        artifact_root: Optional custom artifact root
        
    Returns:
        Configured MLflowConfig instance
    """
    global _mlflow_config
    
    _mlflow_config = MLflowConfig(
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )
    _mlflow_config.setup()
    return _mlflow_config


def get_mlflow_config() -> MLflowConfig:
    """Get global MLflow configuration."""
    if _mlflow_config is None:
        raise RuntimeError("MLflow not initialized. Call initialize_mlflow() first.")
    return _mlflow_config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = initialize_mlflow()
    print("\nMLflow Configuration Summary:")
    print(f"  Tracking URI: {config.tracking_uri}")
    print(f"  Artifact Root: {config.artifact_root}")
    print(f"  Experiments: {list(config.list_experiments().keys())}")
