"""Configuration management utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Load and manage project configurations."""
    
    def __init__(self, config_dir="config"):
        """Initialize config manager."""
        self.config_dir = Path(config_dir)
        self._cache = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration file."""
        if config_name in self._cache:
            return self._cache[config_name]
        
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        self._cache[config_name] = config
        return config
    
    def get(self, config_name: str, key: str, default=None) -> Any:
        """Get a config value."""
        config = self.load_config(config_name)
        keys = key.split(".")
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# Global config manager instance
_config_manager = None


def get_config_manager():
    """Get or create the global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_name: str) -> Dict[str, Any]:
    """Load a configuration file."""
    return get_config_manager().load_config(config_name)
