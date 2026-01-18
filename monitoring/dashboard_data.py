"""Dashboard data aggregation for real-time monitoring visualization.

Consolidates alerts, anomalies, performance metrics, and degradation events
into aggregated views for dashboard display.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardDataAggregator:
    """Aggregate monitoring data for dashboard display."""
    
    def __init__(self, data_dir: str = "model_registry"):
        """Initialize dashboard data aggregator."""
        self.data_dir = Path(data_dir)
        self.alerts_path = self.data_dir / "alerts.jsonl"
        self.anomalies_path = self.data_dir / "anomalies.jsonl"
        self.degradation_path = self.data_dir / "degradation_events.jsonl"
        self.metrics_path = self.data_dir / "performance_metrics.jsonl"
        self.job_history_path = self.data_dir / "job_history.jsonl"
    
    def _load_jsonl(self, path: Path, limit: Optional[int] = None) -> List[Dict]:
        """Load records from JSONL file."""
        records = []
        if not path.exists():
            return records
        
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
                        if limit and len(records) >= limit:
                            break
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
        
        return records
    
    def get_recent_alerts(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get recent alerts with severity levels and escalations."""
        alerts = self._load_jsonl(self.alerts_path)
        
        if not alerts:
            return []
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for alert in reversed(alerts):  # Most recent first
            try:
                alert_time = datetime.fromisoformat(alert.get("timestamp", ""))
                if alert_time >= cutoff:
                    recent.append(alert)
                    if len(recent) >= limit:
                        break
            except (ValueError, KeyError):
                continue
        
        return recent
    
    def get_alert_summary(self) -> Dict:
        """Get summary of alerts by severity."""
        alerts = self._load_jsonl(self.alerts_path, limit=100)
        
        summary = {
            "total": len(alerts),
            "by_severity": {
                "INFO": 0,
                "WARNING": 0,
                "CRITICAL": 0
            },
            "by_type": {}
        }
        
        for alert in alerts:
            severity = alert.get("severity", "INFO")
            if severity in summary["by_severity"]:
                summary["by_severity"][severity] += 1
            
            alert_type = alert.get("alert_type", "unknown")
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1
        
        return summary
    
    def get_recent_anomalies(self, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent anomalies detected."""
        anomalies = self._load_jsonl(self.anomalies_path)
        
        if not anomalies:
            return []
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for anomaly in reversed(anomalies):
            try:
                anom_time = datetime.fromisoformat(anomaly.get("timestamp", ""))
                if anom_time >= cutoff:
                    recent.append(anomaly)
                    if len(recent) >= limit:
                        break
            except (ValueError, KeyError):
                continue
        
        return recent
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomalies by detection method."""
        anomalies = self._load_jsonl(self.anomalies_path, limit=100)
        
        summary = {
            "total": len(anomalies),
            "by_method": {
                "z_score": 0,
                "iqr": 0,
                "trend": 0
            },
            "by_metric": {}
        }
        
        for anomaly in anomalies:
            method = anomaly.get("method", "unknown")
            if method in summary["by_method"]:
                summary["by_method"][method] += 1
            
            metric = anomaly.get("metric", "unknown")
            summary["by_metric"][metric] = summary["by_metric"].get(metric, 0) + 1
        
        return summary
    
    def get_degradation_events(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get performance degradation events."""
        events = self._load_jsonl(self.degradation_path)
        
        if not events:
            return []
        
        # Filter by time
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for event in reversed(events):
            try:
                event_time = datetime.fromisoformat(event.get("timestamp", ""))
                if event_time >= cutoff:
                    recent.append(event)
                    if len(recent) >= limit:
                        break
            except (ValueError, KeyError):
                continue
        
        return recent
    
    def get_performance_metrics_latest(self) -> Dict:
        """Get latest performance metrics."""
        metrics = self._load_jsonl(self.metrics_path, limit=1)
        
        if not metrics:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "regime_count": 0,
                "data_quality": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        
        latest = metrics[0]
        return {
            "accuracy": latest.get("metrics", {}).get("accuracy", 0.0),
            "precision": latest.get("metrics", {}).get("precision", 0.0),
            "recall": latest.get("metrics", {}).get("recall", 0.0),
            "regime_count": latest.get("metrics", {}).get("regime_count", 0),
            "data_quality": latest.get("metrics", {}).get("data_quality", 0.0),
            "timestamp": latest.get("timestamp", datetime.now().isoformat())
        }
    
    def get_performance_metrics_history(self, limit: int = 50) -> List[Dict]:
        """Get performance metrics history."""
        metrics_list = self._load_jsonl(self.metrics_path, limit=limit)
        
        result = []
        for m in reversed(metrics_list):
            result.append({
                "timestamp": m.get("timestamp"),
                "accuracy": m.get("metrics", {}).get("accuracy", 0.0),
                "precision": m.get("metrics", {}).get("precision", 0.0),
                "recall": m.get("metrics", {}).get("recall", 0.0),
                "regime_count": m.get("metrics", {}).get("regime_count", 0),
                "data_quality": m.get("metrics", {}).get("data_quality", 0.0)
            })
        
        return result
    
    def get_monitoring_job_status(self) -> Dict:
        """Get status of monitoring jobs (last run times, next scheduled runs)."""
        history = self._load_jsonl(self.job_history_path, limit=10)
        
        if not history:
            return {
                "last_drift_check": None,
                "last_performance_check": None,
                "last_anomaly_detection": None,
                "status": "No monitoring history"
            }
        
        latest = history[0] if history else {}
        
        return {
            "last_drift_check": latest.get("drift_check", {}).get("timestamp"),
            "last_performance_check": latest.get("performance_check", {}).get("timestamp"),
            "last_anomaly_detection": latest.get("anomaly_detection", {}).get("timestamp"),
            "drift_check_status": latest.get("drift_check", {}).get("result", "unknown"),
            "performance_check_status": latest.get("performance_check", {}).get("result", "unknown"),
            "anomaly_detection_status": latest.get("anomaly_detection", {}).get("result", "unknown"),
            "last_update": latest.get("timestamp", datetime.now().isoformat())
        }
    
    def get_dashboard_summary(self) -> Dict:
        """Get complete dashboard summary."""
        return {
            "summary": {
                "alerts": self.get_alert_summary(),
                "anomalies": self.get_anomaly_summary(),
                "recent_degradations": len(self.get_degradation_events(hours=24))
            },
            "latest_metrics": self.get_performance_metrics_latest(),
            "monitoring_status": self.get_monitoring_job_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_alert_timeline(self, hours: int = 24) -> List[Dict]:
        """Get alerts organized by time for timeline visualization."""
        alerts = self.get_recent_alerts(hours=hours, limit=100)
        
        # Group by hour
        timeline = {}
        for alert in alerts:
            try:
                alert_time = datetime.fromisoformat(alert.get("timestamp", ""))
                hour_key = alert_time.strftime("%Y-%m-%d %H:00")
                if hour_key not in timeline:
                    timeline[hour_key] = []
                timeline[hour_key].append(alert)
            except (ValueError, KeyError):
                continue
        
        # Convert to sorted list
        result = []
        for hour in sorted(timeline.keys(), reverse=True):
            result.append({
                "hour": hour,
                "count": len(timeline[hour]),
                "alerts": timeline[hour],
                "severity_distribution": {
                    "INFO": sum(1 for a in timeline[hour] if a.get("severity") == "INFO"),
                    "WARNING": sum(1 for a in timeline[hour] if a.get("severity") == "WARNING"),
                    "CRITICAL": sum(1 for a in timeline[hour] if a.get("severity") == "CRITICAL")
                }
            })
        
        return result
    
    def get_anomaly_details(self, metric: Optional[str] = None, 
                          method: Optional[str] = None, 
                          hours: int = 24) -> List[Dict]:
        """Get detailed anomaly information with optional filtering."""
        anomalies = self.get_recent_anomalies(hours=hours, limit=200)
        
        filtered = []
        for anom in anomalies:
            if metric and anom.get("metric") != metric:
                continue
            if method and anom.get("method") != method:
                continue
            filtered.append(anom)
        
        return filtered
    
    def get_performance_trends(self, metric: str = "accuracy", 
                              hours: int = 168,  # 1 week
                              points: int = 50) -> Dict:
        """Get performance trend data for a specific metric."""
        history = self.get_performance_metrics_history(limit=100)
        
        if not history:
            return {
                "metric": metric,
                "data": [],
                "trend": "stable",
                "current_value": 0.0
            }
        
        # Extract metric values
        timestamps = []
        values = []
        
        cutoff = datetime.now() - timedelta(hours=hours)
        for record in reversed(history):
            try:
                ts = datetime.fromisoformat(record.get("timestamp", ""))
                if ts >= cutoff:
                    timestamps.append(ts)
                    values.append(record.get(metric, 0.0))
            except (ValueError, KeyError):
                continue
        
        if not values:
            return {
                "metric": metric,
                "data": [],
                "trend": "stable",
                "current_value": 0.0
            }
        
        # Determine trend
        if len(values) >= 2:
            trend_slope = (values[-1] - values[0]) / len(values)
            if trend_slope > 0.01:
                trend = "improving"
            elif trend_slope < -0.01:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "metric": metric,
            "data": [
                {"timestamp": str(ts), "value": v}
                for ts, v in zip(timestamps, values)
            ],
            "trend": trend,
            "current_value": values[-1] if values else 0.0,
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": sum(values) / len(values)
        }


def aggregate_dashboard_data(data_dir: str = "model_registry") -> Dict:
    """Convenience function to aggregate all dashboard data."""
    aggregator = DashboardDataAggregator(data_dir=data_dir)
    
    return {
        "summary": aggregator.get_dashboard_summary(),
        "recent_alerts": aggregator.get_recent_alerts(limit=20),
        "alert_timeline": aggregator.get_alert_timeline(),
        "recent_anomalies": aggregator.get_recent_anomalies(limit=30),
        "degradation_events": aggregator.get_degradation_events(limit=20),
        "performance_history": aggregator.get_performance_metrics_history(limit=50),
        "performance_trends": {
            "accuracy": aggregator.get_performance_trends("accuracy"),
            "precision": aggregator.get_performance_trends("precision"),
            "recall": aggregator.get_performance_trends("recall")
        },
        "timestamp": datetime.now().isoformat()
    }
