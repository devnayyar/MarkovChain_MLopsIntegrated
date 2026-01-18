"""Performance monitoring and model metrics tracking."""
import json
from pathlib import Path
from datetime import datetime
import pandas as pd


class PerformanceMonitor:
    """Track model performance metrics over time."""
    
    def __init__(self, metrics_log_path="model_registry/performance_metrics.jsonl"):
        """Initialize performance monitor with log file."""
        self.metrics_log_path = Path(metrics_log_path)
        self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_metrics(self, metrics_dict, model_version="v1", run_id="default"):
        """Log metrics to JSONL file."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version,
            "run_id": run_id,
            "metrics": metrics_dict
        }
        
        with open(self.metrics_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        print(f"✅ Logged metrics for {model_version}: {metrics_dict}")
    
    def get_performance_trends(self, metric_name, window=10):
        """Get performance trend for a specific metric."""
        if not self.metrics_log_path.exists():
            return None
        
        records = []
        with open(self.metrics_log_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        
        if not records:
            return None
        
        # Extract metric values
        values = []
        timestamps = []
        for rec in records[-window:]:
            if metric_name in rec.get("metrics", {}):
                values.append(rec["metrics"][metric_name])
                timestamps.append(rec["timestamp"])
        
        if not values:
            return None
        
        return {
            "metric": metric_name,
            "values": values,
            "timestamps": timestamps,
            "mean": sum(values) / len(values),
            "current": values[-1],
            "trend": "up" if values[-1] > values[0] else "down"
        }
    
    def get_all_metrics(self):
        """Get all logged metrics."""
        if not self.metrics_log_path.exists():
            return []
        
        records = []
        with open(self.metrics_log_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        
        return records
    
    def generate_report(self):
        """Generate performance summary report."""
        records = self.get_all_metrics()
        
        if not records:
            print("No performance metrics logged yet.")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE MONITORING REPORT")
        print("="*70)
        
        latest = records[-1]
        print(f"\nLatest Run: {latest['timestamp']}")
        print(f"Model Version: {latest['model_version']}")
        print(f"Run ID: {latest['run_id']}\n")
        
        print("Metrics:")
        for metric, value in latest.get("metrics", {}).items():
            trend = self.get_performance_trends(metric, window=5)
            if trend:
                print(f"  {metric:30s}: {value:.6f} ({trend['trend']})")
            else:
                print(f"  {metric:30s}: {value:.6f}")
        
        print("\n" + "="*70)
