"""Advanced retraining scheduler with multiple trigger criteria and A/B testing integration."""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class AdvancedRetrainingScheduler:
    """Advanced scheduler for model retraining with multiple criteria."""
    
    def __init__(self, 
                 schedule_interval_days: int = 7,
                 drift_threshold: float = 0.15,
                 performance_threshold: float = 0.05,
                 data_quality_threshold: float = 0.8,
                 min_data_points: int = 100):
        """Initialize advanced retraining scheduler.
        
        Args:
            schedule_interval_days: Days between scheduled retraining
            drift_threshold: Drift threshold for drift-triggered retraining
            performance_threshold: Performance degradation threshold (as fraction)
            data_quality_threshold: Minimum acceptable data quality score
            min_data_points: Minimum data points needed before retraining
        """
        self.schedule_interval_days = schedule_interval_days
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.data_quality_threshold = data_quality_threshold
        self.min_data_points = min_data_points
    
    def check_scheduled_retraining(self, last_retrain_time: Optional[str] = None) -> bool:
        """Check if scheduled retraining is due.
        
        Args:
            last_retrain_time: ISO formatted timestamp of last retraining
        
        Returns:
            True if scheduled retraining is due
        """
        if not last_retrain_time:
            return True
        
        try:
            last_time = datetime.fromisoformat(last_retrain_time)
            days_since = (datetime.now() - last_time).days
            return days_since >= self.schedule_interval_days
        except (ValueError, TypeError):
            return True
    
    def check_drift_triggered_retraining(self, drift_metrics: Dict) -> Tuple[bool, Optional[str]]:
        """Check if drift-triggered retraining is needed.
        
        Args:
            drift_metrics: Dictionary with 'regime_drift' and 'transition_drift'
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        regime_drift = drift_metrics.get("regime_drift", 0.0)
        transition_drift = drift_metrics.get("transition_drift", 0.0)
        
        if regime_drift > self.drift_threshold:
            return True, f"Regime drift {regime_drift:.4f} exceeds threshold {self.drift_threshold}"
        
        if transition_drift > self.drift_threshold:
            return True, f"Transition drift {transition_drift:.4f} exceeds threshold {self.drift_threshold}"
        
        return False, None
    
    def check_performance_triggered_retraining(self, 
                                             current_metrics: Dict,
                                             baseline_metrics: Dict) -> Tuple[bool, Optional[str]]:
        """Check if performance degradation triggers retraining.
        
        Args:
            current_metrics: Current model performance metrics
            baseline_metrics: Baseline model performance metrics
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        current_accuracy = current_metrics.get("accuracy", 1.0)
        baseline_accuracy = baseline_metrics.get("accuracy", 1.0)
        
        if baseline_accuracy == 0:
            return False, None
        
        degradation = (baseline_accuracy - current_accuracy) / baseline_accuracy
        
        if degradation > self.performance_threshold:
            return True, f"Performance degradation {degradation:.4f} exceeds threshold {self.performance_threshold}"
        
        # Also check precision and recall
        current_precision = current_metrics.get("precision", 1.0)
        baseline_precision = baseline_metrics.get("precision", 1.0)
        precision_degradation = (baseline_precision - current_precision) / baseline_precision if baseline_precision > 0 else 0
        
        if precision_degradation > self.performance_threshold:
            return True, f"Precision degradation {precision_degradation:.4f} exceeds threshold"
        
        return False, None
    
    def check_data_quality_triggered_retraining(self, data_quality_metrics: Dict) -> Tuple[bool, Optional[str]]:
        """Check if data quality issues trigger retraining.
        
        Args:
            data_quality_metrics: Dictionary with data quality scores
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        overall_quality = data_quality_metrics.get("overall_quality", 1.0)
        missing_data_ratio = data_quality_metrics.get("missing_data_ratio", 0.0)
        outlier_ratio = data_quality_metrics.get("outlier_ratio", 0.0)
        
        if overall_quality < self.data_quality_threshold:
            return True, f"Data quality {overall_quality:.4f} below threshold {self.data_quality_threshold}"
        
        if missing_data_ratio > 0.2:
            return True, f"Missing data ratio {missing_data_ratio:.4f} exceeds 20%"
        
        if outlier_ratio > 0.1:
            return True, f"Outlier ratio {outlier_ratio:.4f} exceeds 10%"
        
        return False, None
    
    def check_new_data_available(self, data_metrics: Dict) -> Tuple[bool, Optional[str]]:
        """Check if sufficient new data is available for retraining.
        
        Args:
            data_metrics: Dictionary with data availability metrics
        
        Returns:
            Tuple of (enough_data, reason)
        """
        new_data_points = data_metrics.get("new_data_points", 0)
        total_data_points = data_metrics.get("total_data_points", 0)
        
        if new_data_points < self.min_data_points:
            return False, f"Only {new_data_points} new data points; need {self.min_data_points}"
        
        return True, None
    
    def evaluate_all_retraining_criteria(self,
                                        last_retrain_time: Optional[str] = None,
                                        drift_metrics: Optional[Dict] = None,
                                        current_metrics: Optional[Dict] = None,
                                        baseline_metrics: Optional[Dict] = None,
                                        data_quality_metrics: Optional[Dict] = None,
                                        data_metrics: Optional[Dict] = None) -> Dict:
        """Evaluate all retraining criteria.
        
        Args:
            last_retrain_time: ISO timestamp of last retraining
            drift_metrics: Drift detection metrics
            current_metrics: Current model performance
            baseline_metrics: Baseline model performance
            data_quality_metrics: Data quality scores
            data_metrics: Data availability metrics
        
        Returns:
            Dictionary with detailed retraining evaluation
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": False,
            "triggers": {
                "scheduled": False,
                "drift": False,
                "performance": False,
                "data_quality": False,
                "data_available": True
            },
            "reasons": [],
            "scores": {}
        }
        
        # Check scheduled retraining
        if self.check_scheduled_retraining(last_retrain_time):
            results["triggers"]["scheduled"] = True
            results["reasons"].append("Scheduled retraining interval reached")
        
        # Check drift-triggered
        if drift_metrics:
            drift_retrain, reason = self.check_drift_triggered_retraining(drift_metrics)
            if drift_retrain:
                results["triggers"]["drift"] = True
                results["reasons"].append(reason)
            results["scores"]["drift"] = {
                "regime_drift": drift_metrics.get("regime_drift", 0.0),
                "transition_drift": drift_metrics.get("transition_drift", 0.0),
                "threshold": self.drift_threshold
            }
        
        # Check performance-triggered
        if current_metrics and baseline_metrics:
            perf_retrain, reason = self.check_performance_triggered_retraining(
                current_metrics, baseline_metrics
            )
            if perf_retrain:
                results["triggers"]["performance"] = True
                results["reasons"].append(reason)
            results["scores"]["performance"] = {
                "current_accuracy": current_metrics.get("accuracy", 0.0),
                "baseline_accuracy": baseline_metrics.get("accuracy", 0.0),
                "threshold": self.performance_threshold
            }
        
        # Check data quality-triggered
        if data_quality_metrics:
            dq_retrain, reason = self.check_data_quality_triggered_retraining(data_quality_metrics)
            if dq_retrain:
                results["triggers"]["data_quality"] = True
                results["reasons"].append(reason)
            results["scores"]["data_quality"] = {
                "overall_quality": data_quality_metrics.get("overall_quality", 1.0),
                "threshold": self.data_quality_threshold
            }
        
        # Check data availability
        if data_metrics:
            has_data, reason = self.check_new_data_available(data_metrics)
            if not has_data:
                results["triggers"]["data_available"] = False
                results["reasons"].append(f"Insufficient data: {reason}")
        
        # Determine final decision
        # Retrain if: (scheduled OR drift OR performance OR data_quality) AND data_available
        should_retrain = (
            (results["triggers"]["scheduled"] or 
             results["triggers"]["drift"] or 
             results["triggers"]["performance"] or 
             results["triggers"]["data_quality"])
            and results["triggers"]["data_available"]
        )
        
        results["should_retrain"] = should_retrain
        
        return results
    
    def plan_retraining_strategy(self, evaluation_result: Dict) -> Dict:
        """Plan retraining strategy based on evaluation.
        
        Args:
            evaluation_result: Result from evaluate_all_retraining_criteria
        
        Returns:
            Dictionary with retraining strategy
        """
        if not evaluation_result.get("should_retrain", False):
            return {
                "action": "skip",
                "reason": "No retraining triggers active",
                "priority": "none"
            }
        
        # Determine priority and strategy
        triggers = evaluation_result.get("triggers", {})
        
        priority = "low"
        strategy = []
        
        if triggers.get("drift"):
            priority = "high"
            strategy.append("Focus on new regime patterns")
            strategy.append("Consider expanding training window")
        
        if triggers.get("performance"):
            priority = "high"
            strategy.append("Investigate feature importance changes")
            strategy.append("Consider model architecture adjustments")
        
        if triggers.get("data_quality"):
            priority = "medium"
            strategy.append("Clean and validate new data")
            strategy.append("Address data quality issues before training")
        
        if triggers.get("scheduled"):
            if priority == "low":
                priority = "medium"
            strategy.append("Perform regular model maintenance")
        
        return {
            "action": "retrain",
            "priority": priority,
            "strategy": strategy,
            "triggers_active": [k for k, v in triggers.items() if v],
            "timestamp": datetime.now().isoformat()
        }
    
    def save_retraining_decision(self, decision: Dict, output_path: str = "model_registry/retraining_decisions.jsonl"):
        """Save retraining decision for audit trail.
        
        Args:
            decision: Retraining decision dictionary
            output_path: Path to save decision
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "a") as f:
                f.write(json.dumps(decision) + "\n")
            
            logger.info(f"Saved retraining decision to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save retraining decision: {e}")


class RetrainingJobExecutor:
    """Execute retraining jobs with progress tracking."""
    
    def __init__(self, model_registry_path: str = "model_registry"):
        """Initialize job executor."""
        self.model_registry_path = Path(model_registry_path)
        self.job_log_path = self.model_registry_path / "retraining_jobs.jsonl"
    
    def create_retraining_job(self, 
                            job_type: str = "scheduled",
                            triggered_by: Optional[List[str]] = None,
                            priority: str = "medium") -> Dict:
        """Create a new retraining job.
        
        Args:
            job_type: Type of retraining (scheduled, drift, performance, data_quality)
            triggered_by: List of trigger reasons
            priority: Job priority (low, medium, high)
        
        Returns:
            Job dictionary with ID and metadata
        """
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = {
            "job_id": job_id,
            "type": job_type,
            "priority": priority,
            "triggered_by": triggered_by or [],
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "metrics": {}
        }
        
        return job
    
    def start_retraining_job(self, job: Dict) -> Dict:
        """Mark job as started.
        
        Args:
            job: Job dictionary
        
        Returns:
            Updated job dictionary
        """
        job["status"] = "in_progress"
        job["started_at"] = datetime.now().isoformat()
        self._log_job(job)
        return job
    
    def complete_retraining_job(self, 
                               job: Dict,
                               success: bool = True,
                               model_version: Optional[str] = None,
                               metrics: Optional[Dict] = None) -> Dict:
        """Mark job as completed.
        
        Args:
            job: Job dictionary
            success: Whether job completed successfully
            model_version: Version of trained model
            metrics: Training metrics
        
        Returns:
            Updated job dictionary
        """
        completed_at = datetime.now()
        started_at = datetime.fromisoformat(job.get("started_at", completed_at.isoformat()))
        duration = (completed_at - started_at).total_seconds()
        
        job["status"] = "completed" if success else "failed"
        job["completed_at"] = completed_at.isoformat()
        job["duration_seconds"] = duration
        job["model_version"] = model_version
        job["metrics"] = metrics or {}
        
        self._log_job(job)
        return job
    
    def _log_job(self, job: Dict):
        """Log job to audit trail."""
        try:
            self.model_registry_path.mkdir(parents=True, exist_ok=True)
            with open(self.job_log_path, "a") as f:
                f.write(json.dumps(job) + "\n")
        except Exception as e:
            logger.error(f"Failed to log retraining job: {e}")
    
    def get_job_history(self, limit: int = 50) -> List[Dict]:
        """Get retraining job history."""
        jobs = []
        if not self.job_log_path.exists():
            return jobs
        
        try:
            with open(self.job_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        jobs.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load job history: {e}")
        
        return jobs[-limit:] if jobs else []


def create_retraining_pipeline_config() -> Dict:
    """Create default retraining pipeline configuration."""
    return {
        "scheduler": {
            "schedule_interval_days": 7,
            "drift_threshold": 0.15,
            "performance_threshold": 0.05,
            "data_quality_threshold": 0.8,
            "min_data_points": 100
        },
        "execution": {
            "max_retrain_time_hours": 2,
            "parallel_jobs": 1,
            "gpu_required": False
        },
        "validation": {
            "min_improvement": 0.01,
            "validation_split": 0.2,
            "cross_validation_folds": 5
        },
        "deployment": {
            "auto_deploy": False,
            "require_approval": True,
            "rollback_on_failure": True
        }
    }
