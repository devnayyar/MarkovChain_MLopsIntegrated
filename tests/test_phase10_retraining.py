"""Tests for Phase 10: Advanced retraining and A/B testing."""
import pytest
import json
from datetime import datetime, timedelta
from pathlib import Path
from retraining.scheduler import (
    AdvancedRetrainingScheduler, 
    RetrainingJobExecutor,
    create_retraining_pipeline_config
)
from retraining.ab_testing import (
    ModelComparator,
    ABTestFramework,
    RollbackManager
)


class TestAdvancedRetrainingScheduler:
    """Test advanced retraining scheduler."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = AdvancedRetrainingScheduler(
            schedule_interval_days=7,
            drift_threshold=0.15,
            performance_threshold=0.05
        )
        assert scheduler.schedule_interval_days == 7
        assert scheduler.drift_threshold == 0.15
        assert scheduler.performance_threshold == 0.05
    
    def test_scheduled_retraining_due(self):
        """Test scheduled retraining trigger."""
        scheduler = AdvancedRetrainingScheduler(schedule_interval_days=7)
        
        # No previous retrain = due
        assert scheduler.check_scheduled_retraining(last_retrain_time=None)
        
        # Recent retrain = not due
        recent = (datetime.now() - timedelta(days=1)).isoformat()
        assert not scheduler.check_scheduled_retraining(last_retrain_time=recent)
        
        # Old retrain = due
        old = (datetime.now() - timedelta(days=10)).isoformat()
        assert scheduler.check_scheduled_retraining(last_retrain_time=old)
    
    def test_drift_triggered_retraining(self):
        """Test drift-triggered retraining."""
        scheduler = AdvancedRetrainingScheduler(drift_threshold=0.15)
        
        # High drift = trigger
        drift_metrics = {"regime_drift": 0.20, "transition_drift": 0.05}
        should_retrain, reason = scheduler.check_drift_triggered_retraining(drift_metrics)
        assert should_retrain
        assert "Regime drift" in reason
        
        # Low drift = no trigger
        drift_metrics = {"regime_drift": 0.05, "transition_drift": 0.03}
        should_retrain, reason = scheduler.check_drift_triggered_retraining(drift_metrics)
        assert not should_retrain
    
    def test_performance_triggered_retraining(self):
        """Test performance degradation trigger."""
        scheduler = AdvancedRetrainingScheduler(performance_threshold=0.05)
        
        old_metrics = {"accuracy": 0.90, "precision": 0.85, "recall": 0.88}
        
        # Performance degradation = trigger
        new_metrics = {"accuracy": 0.85, "precision": 0.80, "recall": 0.83}
        should_retrain, reason = scheduler.check_performance_triggered_retraining(
            new_metrics, old_metrics
        )
        assert should_retrain
        
        # No degradation = no trigger
        new_metrics = {"accuracy": 0.91, "precision": 0.86, "recall": 0.89}
        should_retrain, reason = scheduler.check_performance_triggered_retraining(
            new_metrics, old_metrics
        )
        assert not should_retrain
    
    def test_data_quality_triggered_retraining(self):
        """Test data quality trigger."""
        scheduler = AdvancedRetrainingScheduler(data_quality_threshold=0.8)
        
        # Poor quality = trigger
        dq_metrics = {"overall_quality": 0.7, "missing_data_ratio": 0.15, "outlier_ratio": 0.05}
        should_retrain, reason = scheduler.check_data_quality_triggered_retraining(dq_metrics)
        assert should_retrain
        
        # Good quality = no trigger
        dq_metrics = {"overall_quality": 0.95, "missing_data_ratio": 0.05, "outlier_ratio": 0.02}
        should_retrain, reason = scheduler.check_data_quality_triggered_retraining(dq_metrics)
        assert not should_retrain
    
    def test_evaluate_all_criteria(self):
        """Test comprehensive evaluation of all criteria."""
        scheduler = AdvancedRetrainingScheduler()
        
        result = scheduler.evaluate_all_retraining_criteria(
            last_retrain_time=None,
            drift_metrics={"regime_drift": 0.20, "transition_drift": 0.05},
            current_metrics={"accuracy": 0.85},
            baseline_metrics={"accuracy": 0.90},
            data_quality_metrics={"overall_quality": 0.95},
            data_metrics={"new_data_points": 500, "total_data_points": 1000}
        )
        
        assert "should_retrain" in result
        assert "triggers" in result
        assert "reasons" in result


class TestRetrainingJobExecutor:
    """Test retraining job execution."""
    
    def test_create_job(self):
        """Test job creation."""
        executor = RetrainingJobExecutor()
        
        job = executor.create_retraining_job(
            job_type="drift",
            triggered_by=["high_regime_drift"],
            priority="high"
        )
        
        assert job["type"] == "drift"
        assert job["priority"] == "high"
        assert job["status"] == "created"
    
    def test_job_lifecycle(self):
        """Test job lifecycle (create -> start -> complete)."""
        executor = RetrainingJobExecutor()
        
        # Create
        job = executor.create_retraining_job()
        assert job["status"] == "created"
        
        # Start
        job = executor.start_retraining_job(job)
        assert job["status"] == "in_progress"
        
        # Complete
        job = executor.complete_retraining_job(
            job,
            success=True,
            model_version="v2.1",
            metrics={"accuracy": 0.92}
        )
        assert job["status"] == "completed"
        assert job["model_version"] == "v2.1"
        assert job["duration_seconds"] > 0


class TestModelComparator:
    """Test model comparison."""
    
    def test_model_comparison(self):
        """Test model comparison."""
        comparator = ModelComparator(min_improvement_threshold=0.01)
        
        old_metrics = {"accuracy": 0.90, "precision": 0.85, "recall": 0.88}
        new_metrics = {"accuracy": 0.92, "precision": 0.87, "recall": 0.90}
        
        comparison = comparator.compare_models(old_metrics, new_metrics)
        
        assert comparison["overall_winner"] == "new"
        assert comparison["improvements"]["accuracy"] > 0
        assert "Deploy new model" in comparison["recommendation"]
    
    def test_metric_variance_detection(self):
        """Test metric variance detection."""
        comparator = ModelComparator(stability_threshold=0.05)
        
        # Stable metrics
        stable_history = [0.90, 0.91, 0.89, 0.90, 0.91]
        variance = comparator.detect_metric_variance(stable_history)
        assert variance["stable"]
        
        # Unstable metrics
        unstable_history = [0.85, 0.95, 0.80, 0.98, 0.70]
        variance = comparator.detect_metric_variance(unstable_history)
        assert not variance["stable"]


class TestABTestFramework:
    """Test A/B testing framework."""
    
    def test_create_test(self):
        """Test A/B test creation."""
        framework = ABTestFramework(test_split=0.2)
        
        test = framework.create_test(
            old_model_id="model_v1",
            new_model_id="model_v2",
            duration_hours=24
        )
        
        assert test["old_model_id"] == "model_v1"
        assert test["new_model_id"] == "model_v2"
        assert test["test_split"] == 0.2
        assert test["status"] == "active"
    
    def test_request_routing(self):
        """Test request routing."""
        framework = ABTestFramework(test_split=0.2)
        test = framework.create_test("model_v1", "model_v2")
        
        # Route multiple requests
        new_model_count = 0
        for _ in range(1000):
            model_id = framework.route_request(test)
            if model_id == "model_v2":
                new_model_count += 1
        
        # Should be roughly 20% (200) routed to new model
        assert 150 < new_model_count < 250
    
    def test_test_completion(self):
        """Test determining test completion."""
        framework = ABTestFramework(min_test_size=100)
        test = framework.create_test("model_v1", "model_v2")
        
        # Not enough data
        test["old_model_requests"] = 50
        test["new_model_requests"] = 30
        assert not framework.is_test_complete(test)
        
        # Enough data
        test["old_model_requests"] = 80
        test["new_model_requests"] = 20
        assert framework.is_test_complete(test)
    
    def test_analyze_test_results(self):
        """Test A/B test result analysis."""
        framework = ABTestFramework()
        
        test = framework.create_test("model_v1", "model_v2")
        test["old_model_requests"] = 500
        test["new_model_requests"] = 500
        test["old_model_errors"] = 25
        test["new_model_errors"] = 10
        
        old_metrics = {"accuracy": 0.95, "precision": 0.90, "recall": 0.88}
        new_metrics = {"accuracy": 0.98, "precision": 0.96, "recall": 0.94}
        
        analysis = framework.analyze_test_results(test, old_metrics, new_metrics)
        
        assert analysis["winner"] is not None
        assert "recommendation" in analysis


class TestRollbackManager:
    """Test rollback management."""
    
    def test_rollback_execution(self):
        """Test rollback execution."""
        manager = RollbackManager()
        
        rollback = manager.execute_rollback(
            current_model_id="model_v2",
            previous_model_id="model_v1",
            reason="High error rate detected"
        )
        
        assert rollback["type"] == "rollback"
        assert rollback["from_model"] == "model_v2"
        assert rollback["to_model"] == "model_v1"
    
    def test_rollback_trigger(self):
        """Test rollback trigger detection."""
        manager = RollbackManager()
        
        # High error rate = trigger
        assert manager.should_trigger_rollback(
            error_rate=0.15,
            baseline_error_rate=0.05,
            threshold=0.1
        )
        
        # Low error rate = no trigger
        assert not manager.should_trigger_rollback(
            error_rate=0.05,
            baseline_error_rate=0.05,
            threshold=0.1
        )


class TestRetrainingConfiguration:
    """Test retraining configuration."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = create_retraining_pipeline_config()
        
        assert "scheduler" in config
        assert "execution" in config
        assert "validation" in config
        assert "deployment" in config
        
        scheduler_config = config["scheduler"]
        assert scheduler_config["schedule_interval_days"] == 7
        assert scheduler_config["drift_threshold"] == 0.15
        assert scheduler_config["performance_threshold"] == 0.05
