"""Statistical anomaly detection for monitoring metrics."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class StatisticalAnomalyDetector:
    """Detect anomalies using multiple statistical methods."""
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """Initialize anomaly detector."""
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
    
    def _load_metrics(self, metrics_path: str) -> pd.DataFrame:
        """Load metrics from JSONL file."""
        records = []
        try:
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {metrics_path}")
            return pd.DataFrame()
        
        if not records:
            return pd.DataFrame()
        
        # Flatten nested metrics
        data = []
        for rec in records:
            row = {"timestamp": rec.get("timestamp", "")}
            if "metrics" in rec:
                row.update(rec["metrics"])
            else:
                row.update(rec)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def detect_zscore_anomalies(self, df: pd.DataFrame, 
                               numeric_cols: Optional[List[str]] = None) -> List[Dict]:
        """Detect anomalies using Z-score method."""
        anomalies = []
        
        if df.empty:
            return anomalies
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in df.columns or len(df[col]) < 2:
                continue
            
            mean = df[col].mean()
            std = df[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df[col] - mean) / std)
            anomaly_indices = np.where(z_scores > self.z_threshold)[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    "method": "z_score",
                    "timestamp": str(df.iloc[idx].get("timestamp", "")),
                    "metric": col,
                    "value": float(df.iloc[idx][col]),
                    "z_score": float(z_scores.iloc[idx]),
                    "mean": float(mean),
                    "std": float(std),
                    "threshold": self.z_threshold
                })
        
        return anomalies
    
    def detect_iqr_anomalies(self, df: pd.DataFrame, 
                            numeric_cols: Optional[List[str]] = None) -> List[Dict]:
        """Detect anomalies using IQR (Interquartile Range) method."""
        anomalies = []
        
        if df.empty:
            return anomalies
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in df.columns or len(df[col]) < 4:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            
            anomaly_indices = np.where((df[col] < lower_bound) | (df[col] > upper_bound))[0]
            
            for idx in anomaly_indices:
                anomalies.append({
                    "method": "iqr",
                    "timestamp": str(df.iloc[idx].get("timestamp", "")),
                    "metric": col,
                    "value": float(df.iloc[idx][col]),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "Q1": float(Q1),
                    "Q3": float(Q3),
                    "IQR": float(IQR)
                })
        
        return anomalies
    
    def detect_trend_anomalies(self, df: pd.DataFrame, 
                              window: int = 5,
                              numeric_cols: Optional[List[str]] = None) -> List[Dict]:
        """Detect anomalies based on trend deviations."""
        anomalies = []
        
        if df.empty or len(df) < window + 2:
            return anomalies
        
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            # Calculate rolling mean and std
            rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = df[col].rolling(window=window, min_periods=1).std()
            
            # Detect deviations
            for i in range(len(df)):
                if rolling_std.iloc[i] > 0:
                    deviation = abs(df.iloc[i][col] - rolling_mean.iloc[i]) / rolling_std.iloc[i]
                    
                    if deviation > self.z_threshold:
                        anomalies.append({
                            "method": "trend",
                            "timestamp": str(df.iloc[i].get("timestamp", "")),
                            "metric": col,
                            "value": float(df.iloc[i][col]),
                            "rolling_mean": float(rolling_mean.iloc[i]),
                            "rolling_std": float(rolling_std.iloc[i]),
                            "deviation": float(deviation),
                            "window": window
                        })
        
        return anomalies
    
    def detect_all_anomalies(self, df: pd.DataFrame,
                            numeric_cols: Optional[List[str]] = None) -> Dict:
        """Run all anomaly detection methods."""
        z_score_anomalies = self.detect_zscore_anomalies(df, numeric_cols)
        iqr_anomalies = self.detect_iqr_anomalies(df, numeric_cols)
        trend_anomalies = self.detect_trend_anomalies(df, window=5, numeric_cols=numeric_cols)
        
        return {
            "z_score": z_score_anomalies,
            "iqr": iqr_anomalies,
            "trend": trend_anomalies,
            "total": len(z_score_anomalies) + len(iqr_anomalies) + len(trend_anomalies)
        }
    
    def run_detection(self, metrics_path: str = "model_registry/performance_metrics.jsonl") -> Dict:
        """Run complete anomaly detection on metrics file."""
        df = self._load_metrics(metrics_path)
        
        if df.empty:
            logger.warning("No metrics available for anomaly detection")
            return {"status": "no_data", "anomalies": {}}
        
        results = self.detect_all_anomalies(df)
        logger.info(f"Found {results['total']} anomalies across all methods")
        
        return {
            "status": "success",
            "records_analyzed": len(df),
            "anomalies": results
        }


class PerformanceDegradationDetector:
    """Detect performance degradation in model metrics."""
    
    def __init__(self, degradation_threshold: float = 0.05):
        """Initialize degradation detector."""
        self.degradation_threshold = degradation_threshold
    
    def detect_degradation(self, metrics_path: str = "model_registry/performance_metrics.jsonl",
                          lookback_periods: int = 10) -> Dict:
        """Detect performance degradation."""
        try:
            records = []
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            
            if len(records) < lookback_periods + 1:
                return {
                    "degradation_detected": False,
                    "reason": "Insufficient historical data"
                }
            
            # Get recent metrics
            recent_records = records[-lookback_periods:]
            historical_records = records[:-lookback_periods]
            
            # Extract numeric metrics
            recent_metrics = {}
            historical_metrics = {}
            
            for rec in recent_records:
                metrics = rec.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        if k not in recent_metrics:
                            recent_metrics[k] = []
                        recent_metrics[k].append(v)
            
            for rec in historical_records:
                metrics = rec.get("metrics", {})
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        if k not in historical_metrics:
                            historical_metrics[k] = []
                        historical_metrics[k].append(v)
            
            # Compare metrics
            degradations = []
            for metric_name in recent_metrics.keys():
                if metric_name in historical_metrics:
                    recent_mean = np.mean(recent_metrics[metric_name])
                    historical_mean = np.mean(historical_metrics[metric_name])
                    
                    # Calculate percentage change
                    if historical_mean != 0:
                        pct_change = (recent_mean - historical_mean) / abs(historical_mean)
                        
                        # Flag if degradation exceeds threshold
                        if pct_change < -self.degradation_threshold:
                            degradations.append({
                                "metric": metric_name,
                                "recent_mean": float(recent_mean),
                                "historical_mean": float(historical_mean),
                                "pct_change": float(pct_change),
                                "threshold": -self.degradation_threshold
                            })
            
            return {
                "degradation_detected": len(degradations) > 0,
                "degradations": degradations,
                "lookback_periods": lookback_periods
            }
        
        except Exception as e:
            logger.error(f"Degradation detection failed: {e}")
            return {
                "degradation_detected": False,
                "error": str(e)
            }
    
    def save_degradation_events(self, degradations: List[Dict],
                               events_path: str = "model_registry/degradation_events.jsonl"):
        """Save degradation events."""
        ensure_dir("model_registry")
        
        with open(events_path, "a") as f:
            for deg in degradations:
                record = {
                    "detected_at": datetime.now().isoformat(),
                    **deg
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Saved {len(degradations)} degradation events")
