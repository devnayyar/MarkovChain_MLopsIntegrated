"""Alert system for drift, performance, and anomalies."""
import json
from pathlib import Path
from datetime import datetime
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertSystem:
    """Manage system alerts and notifications."""
    
    def __init__(self, alerts_log_path="model_registry/alerts.jsonl"):
        """Initialize alert system."""
        self.alerts_log_path = Path(alerts_log_path)
        self.alerts_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log_alert(self, message, level=AlertLevel.INFO, alert_type="general", context=None):
        """Log an alert."""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "type": alert_type,
            "message": message,
            "context": context or {}
        }
        
        with open(self.alerts_log_path, "a") as f:
            f.write(json.dumps(alert) + "\n")
        
        # Print to console
        print(f"[{level.value}] {alert_type.upper()}: {message}")
    
    def drift_alert(self, divergence, threshold, context=None):
        """Alert about detected drift."""
        level = AlertLevel.CRITICAL if divergence > threshold * 1.5 else AlertLevel.WARNING
        message = f"Distribution drift detected. Divergence: {divergence:.4f}, Threshold: {threshold:.4f}"
        self.log_alert(message, level=level, alert_type="drift", context=context)
    
    def performance_alert(self, metric_name, current_value, expected_value, context=None):
        """Alert about performance degradation."""
        degradation = abs(current_value - expected_value) / expected_value * 100
        level = AlertLevel.CRITICAL if degradation > 20 else AlertLevel.WARNING
        message = f"Performance issue: {metric_name} = {current_value:.4f}, expected ~{expected_value:.4f} ({degradation:.1f}% deviation)"
        self.log_alert(message, level=level, alert_type="performance", context=context)
    
    def retraining_alert(self, reason, context=None):
        """Alert about retraining trigger."""
        message = f"Retraining triggered: {reason}"
        self.log_alert(message, level=AlertLevel.INFO, alert_type="retraining", context=context)
    
    def get_recent_alerts(self, count=10, level_filter=None):
        """Get recent alerts."""
        if not self.alerts_log_path.exists():
            return []
        
        alerts = []
        with open(self.alerts_log_path, "r") as f:
            for line in f:
                alerts.append(json.loads(line))
        
        if level_filter:
            alerts = [a for a in alerts if a["level"] == level_filter.value]
        
        return alerts[-count:]
    
    def get_critical_alerts(self):
        """Get all critical alerts."""
        return self.get_recent_alerts(count=100, level_filter=AlertLevel.CRITICAL)
    
    def generate_alert_summary(self):
        """Generate alert summary report."""
        all_alerts = []
        if self.alerts_log_path.exists():
            with open(self.alerts_log_path, "r") as f:
                all_alerts = [json.loads(line) for line in f]
        
        if not all_alerts:
            print("No alerts logged.")
            return
        
        print("\n" + "="*70)
        print("ALERT SUMMARY")
        print("="*70)
        
        # Count by level
        level_counts = {}
        type_counts = {}
        for alert in all_alerts:
            level = alert["level"]
            alert_type = alert["type"]
            level_counts[level] = level_counts.get(level, 0) + 1
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        print("\nAlerts by Level:")
        for level in ["CRITICAL", "WARNING", "INFO"]:
            count = level_counts.get(level, 0)
            print(f"  {level:10s}: {count:4d}")
        
        print("\nAlerts by Type:")
        for alert_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {alert_type:15s}: {count:4d}")
        
        print("\nRecent Alerts:")
        for alert in all_alerts[-5:]:
            print(f"  [{alert['timestamp']}] {alert['level']:8s} | {alert['type']:12s} | {alert['message']}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Test alerts
    alert_system.drift_alert(0.15, 0.1)
    alert_system.performance_alert("log_likelihood", -250.0, -205.0)
    alert_system.retraining_alert("Drift threshold exceeded")
    
    alert_system.generate_alert_summary()
