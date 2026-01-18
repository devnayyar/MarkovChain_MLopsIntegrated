"""Scheduled monitoring jobs for automated drift and performance checks."""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

from monitoring.drift_detection import DriftDetector
from monitoring.performance import PerformanceMonitor
from monitoring.alerts import AlertSystem
from utils.helpers import ensure_dir, load_parquet

logger = logging.getLogger(__name__)


class MonitoringJobScheduler:
    """Manages scheduled monitoring jobs."""
    
    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        """Initialize scheduler."""
        self.config = {
            "drift_check_interval_days": 7,
            "performance_check_interval_days": 1,
            "anomaly_z_threshold": 3.0,
            "trend_window_size": 30,
        }
        self.job_history = {}
        self.last_run = {}
    
    def should_run_drift_check(self) -> bool:
        """Check if drift detection should run."""
        if "drift_check" not in self.last_run:
            return True
        
        days_since = (datetime.now() - self.last_run["drift_check"]).days
        return days_since >= self.config["drift_check_interval_days"]
    
    def should_run_performance_check(self) -> bool:
        """Check if performance check should run."""
        if "performance_check" not in self.last_run:
            return True
        
        days_since = (datetime.now() - self.last_run["performance_check"]).days
        return days_since >= self.config["performance_check_interval_days"]
    
    def run_drift_check(self, gold_path: str = "data/gold/markov_state_sequences.parquet",
                       alerts_path: str = "model_registry/alerts.jsonl") -> Dict:
        """Run scheduled drift detection."""
        logger.info("Starting scheduled drift check...")
        try:
            ensure_dir("model_registry")
            
            detector = DriftDetector(
                reference_window=12,
                current_window=3,
                threshold=0.1
            )
            
            result = detector.detect_regime_drift(gold_path)
            logger.info(f"Drift check result: {result}")
            
            # Log alert if drift detected
            if result.get("drift_detected"):
                alert_sys = AlertSystem(alerts_path)
                alert_sys.drift_alert(
                    drift_type="regime_distribution",
                    js_divergence=result.get("js_divergence", 0),
                    threshold=0.1
                )
                logger.warning("Drift detected! Alert logged.")
            
            self.last_run["drift_check"] = datetime.now()
            return {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "result": result
            }
        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_performance_check(self, metrics_path: str = "model_registry/performance_metrics.jsonl") -> Dict:
        """Run scheduled performance monitoring."""
        logger.info("Starting scheduled performance check...")
        try:
            ensure_dir("model_registry")
            
            perf_monitor = PerformanceMonitor(metrics_path)
            trends = perf_monitor.get_performance_trends(window_size=5)
            
            logger.info(f"Performance trends: {trends}")
            
            # Check for performance degradation
            if trends and "current" in trends:
                current_metrics = trends["current"]
                if "accuracy" in current_metrics and current_metrics["accuracy"] < 0.90:
                    logger.warning("Performance degradation detected!")
            
            self.last_run["performance_check"] = datetime.now()
            return {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "trends": trends
            }
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_all_checks(self) -> Dict:
        """Run all scheduled checks."""
        logger.info("Running all scheduled monitoring checks...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        if self.should_run_drift_check():
            results["checks"]["drift"] = self.run_drift_check()
        else:
            logger.info("Drift check not due yet")
        
        if self.should_run_performance_check():
            results["checks"]["performance"] = self.run_performance_check()
        else:
            logger.info("Performance check not due yet")
        
        return results
    
    def save_job_history(self, history_path: str = "model_registry/job_history.jsonl"):
        """Save job execution history."""
        ensure_dir("model_registry")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "last_run": {k: v.isoformat() if isinstance(v, datetime) else v 
                        for k, v in self.last_run.items()}
        }
        
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        logger.info(f"Job history saved to {history_path}")


class AnomalyDetector:
    """Detect anomalies in metrics using statistical methods."""
    
    def __init__(self, z_threshold: float = 3.0):
        """Initialize anomaly detector."""
        self.z_threshold = z_threshold
    
    def detect_anomalies(self, metrics_path: str = "model_registry/performance_metrics.jsonl") -> List[Dict]:
        """Detect anomalies in metrics using Z-score."""
        anomalies = []
        
        try:
            records = []
            with open(metrics_path, "r") as f:
                for line in f:
                    records.append(json.loads(line))
            
            if not records:
                return anomalies
            
            # Convert to DataFrame
            data = []
            for rec in records:
                if "metrics" in rec:
                    row = rec["metrics"].copy()
                    row["timestamp"] = rec.get("timestamp", "")
                    data.append(row)
            
            if not data:
                return anomalies
            
            df = pd.DataFrame(data)
            
            # Calculate Z-scores for numeric columns
            numeric_cols = df.select_dtypes(include=["number"]).columns
            
            for col in numeric_cols:
                if len(df[col]) > 1:
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std > 0:
                        z_scores = abs((df[col] - mean) / std)
                        anomaly_indices = z_scores[z_scores > self.z_threshold].index.tolist()
                        
                        for idx in anomaly_indices:
                            anomalies.append({
                                "timestamp": df.iloc[idx].get("timestamp", ""),
                                "metric": col,
                                "value": df.iloc[idx][col],
                                "z_score": z_scores[idx],
                                "mean": mean,
                                "std": std
                            })
            
            logger.info(f"Found {len(anomalies)} anomalies")
            return anomalies
        
        except FileNotFoundError:
            logger.warning(f"Metrics file not found: {metrics_path}")
            return anomalies
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return anomalies
    
    def save_anomalies(self, anomalies: List[Dict], anomalies_path: str = "model_registry/anomalies.jsonl"):
        """Save detected anomalies."""
        ensure_dir("model_registry")
        
        with open(anomalies_path, "a") as f:
            for anomaly in anomalies:
                record = {
                    "detected_at": datetime.now().isoformat(),
                    **anomaly
                }
                f.write(json.dumps(record) + "\n")
        
        logger.info(f"Saved {len(anomalies)} anomalies to {anomalies_path}")


class AlertEscalation:
    """Manage alert escalation rules."""
    
    def __init__(self, alerts_path: str = "model_registry/alerts.jsonl"):
        """Initialize alert escalation."""
        self.alerts_path = alerts_path
        self.escalation_rules = {
            "INFO": {"threshold": 3, "next_level": "WARNING", "action": "notify_team"},
            "WARNING": {"threshold": 2, "next_level": "CRITICAL", "action": "alert_ops"},
            "CRITICAL": {"threshold": 1, "next_level": "PAGE_ONCALL", "action": "page_oncall"}
        }
    
    def check_escalation(self, time_window_hours: int = 24) -> Dict:
        """Check if alerts should be escalated."""
        try:
            ensure_dir("model_registry")
            
            if not Path(self.alerts_path).exists():
                return {"escalation_needed": False, "reason": "No alerts file"}
            
            # Count recent alerts by level
            alert_counts = {"INFO": 0, "WARNING": 0, "CRITICAL": 0}
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            with open(self.alerts_path, "r") as f:
                for line in f:
                    alert = json.loads(line)
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    
                    if alert_time > cutoff_time:
                        level = alert.get("level", "INFO")
                        alert_counts[level] += 1
            
            # Check escalation rules
            for level, rule in self.escalation_rules.items():
                if alert_counts[level] >= rule["threshold"]:
                    return {
                        "escalation_needed": True,
                        "current_level": level,
                        "next_level": rule["next_level"],
                        "action": rule["action"],
                        "alert_count": alert_counts[level],
                        "time_window_hours": time_window_hours
                    }
            
            return {"escalation_needed": False, "alert_counts": alert_counts}
        
        except Exception as e:
            logger.error(f"Escalation check failed: {e}")
            return {"escalation_needed": False, "error": str(e)}
    
    def execute_escalation_action(self, escalation: Dict):
        """Execute escalation action."""
        action = escalation.get("action")
        logger.info(f"Executing escalation action: {action}")
        
        # Placeholder for actual escalation actions
        if action == "notify_team":
            logger.info("Notifying team of alert escalation")
        elif action == "alert_ops":
            logger.info("Alerting operations team")
        elif action == "page_oncall":
            logger.critical("Paging on-call engineer!")


def run_scheduled_monitoring(config_path: str = "config/monitoring_config.yaml"):
    """Run complete scheduled monitoring cycle."""
    logger.info("Starting scheduled monitoring cycle...")
    
    # Run scheduled checks
    scheduler = MonitoringJobScheduler(config_path)
    check_results = scheduler.run_all_checks()
    scheduler.save_job_history()
    
    # Detect anomalies
    anomaly_detector = AnomalyDetector(z_threshold=3.0)
    anomalies = anomaly_detector.detect_anomalies()
    if anomalies:
        anomaly_detector.save_anomalies(anomalies)
    
    # Check alert escalation
    escalation = AlertEscalation()
    escalation_check = escalation.check_escalation(time_window_hours=24)
    
    if escalation_check.get("escalation_needed"):
        escalation.execute_escalation_action(escalation_check)
    
    return {
        "checks": check_results,
        "anomalies_found": len(anomalies),
        "escalation": escalation_check
    }
