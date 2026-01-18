"""Retraining scheduler and trigger logic."""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class RetrainingScheduler:
    """Schedule and trigger model retraining based on multiple criteria."""
    
    def __init__(self, config_path="config/retraining_config.yaml"):
        """Initialize retraining scheduler."""
        self.config = {
            "schedule_interval_days": 7,
            "drift_threshold": 0.1,
            "performance_threshold": 0.05,
            "min_samples": 50
        }
        self.last_retrain = None
    
    def check_retrain_needed(self, gold_path="data/gold/markov_state_sequences.parquet",
                            metrics_path="model_registry/performance_metrics.jsonl"):
        """Check if retraining is needed based on multiple criteria."""
        
        triggers = {
            "schedule": self._check_schedule(),
            "drift": self._check_drift(gold_path),
            "performance": self._check_performance(metrics_path),
            "data_availability": self._check_data_availability(gold_path)
        }
        
        should_retrain = any(triggers.values())
        
        print(f"\n📋 Retraining Check Results:")
        for criterion, triggered in triggers.items():
            status = "⚠️ TRIGGERED" if triggered else "✅ OK"
            print(f"  {criterion:20s}: {status}")
        
        if should_retrain:
            print(f"\n🔄 RETRAINING RECOMMENDED")
        else:
            print(f"\n✅ Model up-to-date. Retraining not needed.")
        
        return should_retrain, triggers
    
    def _check_schedule(self):
        """Check if scheduled interval has passed."""
        if self.last_retrain is None:
            return True  # First time
        
        days_since = (datetime.utcnow() - self.last_retrain).days
        return days_since >= self.config["schedule_interval_days"]
    
    def _check_drift(self, gold_path):
        """Check for concept drift in data."""
        if not Path(gold_path).exists():
            return False
        
        df = pd.read_parquet(gold_path)
        if len(df) < 24:  # Need at least 2 years
            return False
        
        # Compare recent vs historical distribution
        recent = df.tail(12)  # Last 12 months
        historical = df.head(-12)  # Everything before
        
        recent_high_risk = (recent['REGIME_RISK'] == 'HIGH_RISK').sum() / len(recent)
        hist_high_risk = (historical['REGIME_RISK'] == 'HIGH_RISK').sum() / len(historical)
        
        divergence = abs(recent_high_risk - hist_high_risk)
        return divergence > self.config["drift_threshold"]
    
    def _check_performance(self, metrics_path):
        """Check for performance degradation."""
        if not Path(metrics_path).exists():
            return False
        
        import json
        metrics = []
        with open(metrics_path, "r") as f:
            for line in f:
                metrics.append(json.loads(line))
        
        if len(metrics) < 2:
            return False
        
        # Compare last 2 runs
        recent = metrics[-1]["metrics"]
        previous = metrics[-2]["metrics"]
        
        if "log_likelihood" not in recent or "log_likelihood" not in previous:
            return False
        
        recent_ll = recent["log_likelihood"]
        previous_ll = previous["log_likelihood"]
        
        degradation = abs(recent_ll - previous_ll) / abs(previous_ll)
        return degradation > self.config["performance_threshold"]
    
    def _check_data_availability(self, gold_path):
        """Check if sufficient new data is available."""
        if not Path(gold_path).exists():
            return False
        
        df = pd.read_parquet(gold_path)
        return len(df) >= self.config["min_samples"]
    
    def mark_retrain_complete(self):
        """Mark retraining as complete (update last_retrain timestamp)."""
        self.last_retrain = datetime.utcnow()
        print(f"✅ Retraining marked complete at {self.last_retrain}")


if __name__ == "__main__":
    scheduler = RetracingScheduler()
    should_retrain, triggers = scheduler.check_retrain_needed()
    
    if should_retrain:
        print("\n→ Next step: Run retraining pipeline")
