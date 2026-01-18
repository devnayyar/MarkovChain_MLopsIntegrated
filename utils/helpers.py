"""Helper utilities for the pipeline."""
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional


def ensure_dir(path: str) -> Path:
    """Ensure directory exists."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_data_path(layer: str, filename: str = "") -> Path:
    """Get path for data files."""
    base = Path("data") / layer
    if filename:
        return base / filename
    return base


def load_parquet(filepath: str) -> pd.DataFrame:
    """Load parquet file."""
    return pd.read_parquet(filepath)


def save_parquet(df: pd.DataFrame, filepath: str):
    """Save dataframe to parquet."""
    ensure_dir(str(Path(filepath).parent))
    df.to_parquet(filepath, index=False)


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file."""
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: str, index=False):
    """Save dataframe to CSV."""
    ensure_dir(str(Path(filepath).parent))
    df.to_csv(filepath, index=index)


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_file_size_mb(filepath: str) -> float:
    """Get file size in MB."""
    return Path(filepath).stat().st_size / (1024 * 1024)


def list_files_by_pattern(directory: str, pattern: str = "*") -> List[Path]:
    """List files matching pattern."""
    return list(Path(directory).glob(pattern))


def ensure_columns(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    """Ensure required columns exist."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def rename_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename columns safely."""
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
