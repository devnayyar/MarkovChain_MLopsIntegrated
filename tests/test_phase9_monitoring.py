"""Tests for Phase 9 Advanced Monitoring system."""
import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from monitoring.scheduled_jobs import MonitoringJobScheduler, AnomalyDetector, AlertEscalation, run_scheduled_monitoring
from monitoring.anomaly_detector import StatisticalAnomalyDetector, PerformanceDegradationDetector
from utils.helpers import ensure_dir


class TestMonitoringJobScheduler:
    """Test scheduled monitoring jobs."""
    
    def test_scheduler_initialization(self):
        """Test scheduler init."""
        scheduler = MonitoringJobScheduler()
        assert scheduler.config is not None
        assert "drift_check_interval_days" in scheduler.config
        assert "performance_check_interval_days" in scheduler.config
    
    def test_drift_check_timing(self):
        """Test drift check scheduling."""
        scheduler = MonitoringJobScheduler()
        assert scheduler.should_run_drift_check()
        
        # After running, shouldn't run again immediately
        scheduler.last_run["drift_check"] = datetime.now()
        # (Would need to mock time to fully test)
    
    def test_performance_check_timing(self):
        """Test performance check scheduling."""
        scheduler = MonitoringJobScheduler()
        assert scheduler.should_run_performance_check()
    
    def test_save_job_history(self):
        """Test job history saving."""
        scheduler = MonitoringJobScheduler()
        scheduler.last_run["drift_check"] = datetime.now()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "job_history.jsonl"
            scheduler.save_job_history(str(history_path))
            
            assert history_path.exists()
            with open(history_path) as f:
                record = json.loads(f.readline())
                assert "timestamp" in record
                assert "last_run" in record


class TestAnomalyDetectorOld:
    """Test anomaly detection (old module)."""
    
    def test_detector_initialization(self):
        """Test anomaly detector init."""
        detector = AnomalyDetector(z_threshold=3.0)
        assert detector.z_threshold == 3.0
    
    def test_save_anomalies(self):
        """Test saving anomalies."""
        detector = AnomalyDetector()
        anomalies = [
            {"metric": "accuracy", "value": 0.85, "z_score": 3.5},
            {"metric": "precision", "value": 0.80, "z_score": 3.2}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            anomalies_path = Path(tmpdir) / "anomalies.jsonl"
            detector.save_anomalies(anomalies, str(anomalies_path))
            
            assert anomalies_path.exists()
            with open(anomalies_path) as f:
                first = json.loads(f.readline())
                assert "metric" in first
                assert "detected_at" in first


class TestStatisticalAnomalyDetector:
    """Test statistical anomaly detection."""
    
    def test_detector_initialization(self):
        """Test detector init."""
        detector = StatisticalAnomalyDetector(z_threshold=3.0, iqr_multiplier=1.5)
        assert detector.z_threshold == 3.0
        assert detector.iqr_multiplier == 1.5
    
    def test_zscore_anomaly_detection(self):
        """Test Z-score anomaly detection."""
        detector = StatisticalAnomalyDetector(z_threshold=1.0)  # Lower threshold
        
        # Create test data with 20 normal values and clear outlier
        normal = [0.95] * 10 + [0.96] * 10
        data = {
            "timestamp": [str(i) for i in range(20)],
            "accuracy": normal
        }
        df = pd.DataFrame(data)
        df["accuracy"] = df["accuracy"].astype(float)
        df.loc[15, "accuracy"] = 0.10  # Add outlier at index 15
        
        anomalies = detector.detect_zscore_anomalies(df, ["accuracy"])
        # Should detect the outlier at index 15
        assert len(anomalies) > 0, f"Should detect outlier. Got {len(anomalies)} anomalies"
        assert any(a["method"] == "z_score" for a in anomalies)
        assert any(a["metric"] == "accuracy" for a in anomalies)
    
    def test_iqr_anomaly_detection(self):
        """Test IQR anomaly detection."""
        detector = StatisticalAnomalyDetector(iqr_multiplier=1.5)
        
        # Create test data with proper length matching
        timestamps = [str(i) for i in range(11)]
        values = [5.0, 6.0, 5.5, 6.5, 5.0, 6.0, 5.5, 6.5, 5.0, 6.0, 100.0]  # Outlier at end
        data = {
            "timestamp": timestamps,
            "metric": values
        }
        df = pd.DataFrame(data)
        df["metric"] = df["metric"].astype(float)
        
        anomalies = detector.detect_iqr_anomalies(df, ["metric"])
        assert len(anomalies) > 0, "Should detect outlier"
        assert any(a["method"] == "iqr" for a in anomalies)
    
    def test_all_anomaly_detection(self):
        """Test combined anomaly detection."""
        detector = StatisticalAnomalyDetector()
        
        data = {
            "timestamp": [str(i) for i in range(20)],
            "value1": list(np.random.normal(100, 10, 20)),
            "value2": list(np.random.normal(50, 5, 20))
        }
        df = pd.DataFrame(data)
        
        results = detector.detect_all_anomalies(df)
        assert "z_score" in results
        assert "iqr" in results
        assert "trend" in results
        assert "total" in results


class TestPerformanceDegradationDetector:
    """Test performance degradation detection."""
    
    def test_detector_initialization(self):
        """Test detector init."""
        detector = PerformanceDegradationDetector(degradation_threshold=0.05)
        assert detector.degradation_threshold == 0.05
    
    def test_degradation_detection_no_data(self):
        """Test degradation detection with no data."""
        detector = PerformanceDegradationDetector()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            result = detector.detect_degradation(str(metrics_path))
            
            assert result["degradation_detected"] == False
            assert "reason" in result or "error" in result
    
    def test_degradation_detection_with_data(self):
        """Test degradation detection with actual data."""
        detector = PerformanceDegradationDetector(degradation_threshold=0.05)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "metrics.jsonl"
            
            # Create sample metrics with degradation
            with open(metrics_path, "w") as f:
                # Historical data (good performance)
                for i in range(15):
                    f.write(json.dumps({
                        "timestamp": f"2026-01-{i+1:02d}",
                        "metrics": {"accuracy": 0.95, "precision": 0.92}
                    }) + "\n")
                
                # Recent data (degraded performance)
                for i in range(10):
                    f.write(json.dumps({
                        "timestamp": f"2026-02-{i+1:02d}",
                        "metrics": {"accuracy": 0.85, "precision": 0.80}
                    }) + "\n")
            
            result = detector.detect_degradation(str(metrics_path), lookback_periods=10)
            assert result["degradation_detected"] == True
            assert len(result["degradations"]) > 0


class TestAlertEscalation:
    """Test alert escalation."""
    
    def test_escalation_initialization(self):
        """Test escalation init."""
        escalation = AlertEscalation()
        assert escalation.escalation_rules is not None
        assert "INFO" in escalation.escalation_rules
        assert "WARNING" in escalation.escalation_rules
        assert "CRITICAL" in escalation.escalation_rules
    
    def test_escalation_check_no_alerts(self):
        """Test escalation check with no alerts."""
        escalation = AlertEscalation()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            alerts_path = Path(tmpdir) / "alerts.jsonl"
            escalation.alerts_path = str(alerts_path)
            
            result = escalation.check_escalation()
            assert result["escalation_needed"] == False
    
    def test_escalation_execution(self):
        """Test escalation action execution."""
        escalation = AlertEscalation()
        
        escalation_event = {
            "escalation_needed": True,
            "action": "notify_team"
        }
        
        # Should not raise any errors
        escalation.execute_escalation_action(escalation_event)


class TestScheduledMonitoringIntegration:
    """Integration tests for scheduled monitoring."""
    
    def test_run_scheduled_monitoring(self):
        """Test complete scheduled monitoring run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy metrics file
            metrics_path = Path(tmpdir) / "performance_metrics.jsonl"
            with open(metrics_path, "w") as f:
                for i in range(10):
                    f.write(json.dumps({
                        "timestamp": f"2026-01-{i+1:02d}",
                        "metrics": {"accuracy": 0.95, "precision": 0.92}
                    }) + "\n")
            
            # Run monitoring (would need to mock paths)
            # For now just test the function exists and is callable
            assert callable(run_scheduled_monitoring)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
