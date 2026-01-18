"""A/B testing framework for comparing old vs new models."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare performance of two models."""
    
    def __init__(self, 
                 min_improvement_threshold: float = 0.01,
                 stability_threshold: float = 0.05):
        """Initialize model comparator.
        
        Args:
            min_improvement_threshold: Minimum accuracy improvement (as fraction)
            stability_threshold: Maximum acceptable variance in metrics
        """
        self.min_improvement_threshold = min_improvement_threshold
        self.stability_threshold = stability_threshold
    
    def compare_models(self,
                       old_model_metrics: Dict,
                       new_model_metrics: Dict) -> Dict:
        """Compare two models on multiple metrics.
        
        Args:
            old_model_metrics: Metrics dict with accuracy, precision, recall
            new_model_metrics: Metrics dict with accuracy, precision, recall
        
        Returns:
            Comprehensive comparison report
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "old_model": old_model_metrics.copy(),
            "new_model": new_model_metrics.copy(),
            "improvements": {},
            "degradations": {},
            "overall_winner": None,
            "recommendation": None
        }
        
        # Compare accuracy
        old_accuracy = old_model_metrics.get("accuracy", 0.0)
        new_accuracy = new_model_metrics.get("accuracy", 0.0)
        accuracy_improvement = new_accuracy - old_accuracy
        
        comparison["improvements"]["accuracy"] = accuracy_improvement
        if accuracy_improvement > self.min_improvement_threshold:
            comparison["improvements"]["accuracy_significant"] = True
        else:
            comparison["degradations"]["accuracy_significant"] = accuracy_improvement < -self.min_improvement_threshold
        
        # Compare precision
        old_precision = old_model_metrics.get("precision", 0.0)
        new_precision = new_model_metrics.get("precision", 0.0)
        precision_improvement = new_precision - old_precision
        comparison["improvements"]["precision"] = precision_improvement
        
        # Compare recall
        old_recall = old_model_metrics.get("recall", 0.0)
        new_recall = new_model_metrics.get("recall", 0.0)
        recall_improvement = new_recall - old_recall
        comparison["improvements"]["recall"] = recall_improvement
        
        # Compare F1 score
        old_f1 = self._calculate_f1(old_precision, old_recall)
        new_f1 = self._calculate_f1(new_precision, new_recall)
        f1_improvement = new_f1 - old_f1
        comparison["improvements"]["f1"] = f1_improvement
        
        # Determine overall winner
        improvements_count = sum(1 for v in comparison["improvements"].values() if isinstance(v, (int, float)) and v > 0)
        degradations_count = sum(1 for v in comparison["degradations"].values() if isinstance(v, (int, float)) and v > 0)
        
        if improvements_count > degradations_count and accuracy_improvement > 0:
            comparison["overall_winner"] = "new"
            comparison["recommendation"] = "Deploy new model"
        elif degradations_count > improvements_count or accuracy_improvement < -self.min_improvement_threshold:
            comparison["overall_winner"] = "old"
            comparison["recommendation"] = "Keep existing model"
        else:
            comparison["overall_winner"] = "tie"
            comparison["recommendation"] = "Run extended testing or manual review"
        
        return comparison
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_confidence_interval(self, 
                                     metrics: Dict,
                                     n_samples: int = 100,
                                     confidence: float = 0.95) -> Dict:
        """Calculate confidence intervals for metrics.
        
        Args:
            metrics: Model metrics dictionary
            n_samples: Number of samples used
            confidence: Confidence level (0.95 = 95%)
        
        Returns:
            Confidence intervals for each metric
        """
        from scipy import stats
        
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        intervals = {}
        for metric_name in ["accuracy", "precision", "recall"]:
            value = metrics.get(metric_name, 0.0)
            # Approximate standard error
            se = np.sqrt(value * (1 - value) / n_samples)
            margin = z_score * se
            
            intervals[metric_name] = {
                "point_estimate": value,
                "lower_bound": max(0.0, value - margin),
                "upper_bound": min(1.0, value + margin),
                "margin_of_error": margin
            }
        
        return intervals
    
    def detect_metric_variance(self, metric_history: List[float]) -> Dict:
        """Detect if metric variance is too high.
        
        Args:
            metric_history: List of metric values over time
        
        Returns:
            Variance analysis dictionary
        """
        if len(metric_history) < 2:
            return {
                "stable": True,
                "variance": 0.0,
                "std_dev": 0.0,
                "cv": 0.0  # Coefficient of variation
            }
        
        variance = np.var(metric_history)
        std_dev = np.std(metric_history)
        mean = np.mean(metric_history)
        cv = std_dev / mean if mean > 0 else 0.0
        
        is_stable = cv < self.stability_threshold
        
        return {
            "stable": is_stable,
            "variance": float(variance),
            "std_dev": float(std_dev),
            "cv": float(cv),
            "threshold": self.stability_threshold,
            "num_samples": len(metric_history)
        }


class ABTestFramework:
    """A/B testing framework for model comparison and routing."""
    
    def __init__(self, 
                 test_split: float = 0.2,
                 min_test_size: int = 100,
                 confidence_level: float = 0.95):
        """Initialize A/B test framework.
        
        Args:
            test_split: Fraction of traffic to route to new model
            min_test_size: Minimum requests needed for statistical validity
            confidence_level: Statistical confidence level
        """
        self.test_split = test_split
        self.min_test_size = min_test_size
        self.confidence_level = confidence_level
        self.comparator = ModelComparator()
    
    def create_test(self,
                   old_model_id: str,
                   new_model_id: str,
                   duration_hours: int = 24) -> Dict:
        """Create a new A/B test.
        
        Args:
            old_model_id: ID of control (old) model
            new_model_id: ID of treatment (new) model
            duration_hours: Expected test duration
        
        Returns:
            Test configuration dictionary
        """
        test_id = f"abtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "test_id": test_id,
            "old_model_id": old_model_id,
            "new_model_id": new_model_id,
            "test_split": self.test_split,
            "created_at": datetime.now().isoformat(),
            "duration_hours": duration_hours,
            "status": "active",
            "old_model_requests": 0,
            "new_model_requests": 0,
            "old_model_errors": 0,
            "new_model_errors": 0,
            "metrics": {
                "old_model": {},
                "new_model": {}
            }
        }
    
    def route_request(self, test: Dict) -> str:
        """Route request to old or new model based on test split.
        
        Args:
            test: Test configuration dictionary
        
        Returns:
            Model ID to use for this request
        """
        if np.random.random() < self.test_split:
            test["new_model_requests"] += 1
            return test["new_model_id"]
        else:
            test["old_model_requests"] += 1
            return test["old_model_id"]
    
    def record_result(self, test: Dict, 
                     model_id: str,
                     prediction_correct: bool,
                     latency_ms: float):
        """Record test result for a prediction.
        
        Args:
            test: Test configuration dictionary
            model_id: Model that made prediction
            prediction_correct: Whether prediction was correct
            latency_ms: Latency in milliseconds
        """
        if model_id == test["old_model_id"]:
            model_type = "old_model"
        else:
            model_type = "new_model"
        
        if not prediction_correct:
            test[f"{model_type}_errors"] += 1
    
    def is_test_complete(self, test: Dict) -> bool:
        """Check if test has enough data for statistical significance.
        
        Args:
            test: Test configuration dictionary
        
        Returns:
            True if test has sufficient data
        """
        total_requests = test["old_model_requests"] + test["new_model_requests"]
        return total_requests >= self.min_test_size
    
    def analyze_test_results(self, test: Dict,
                            old_model_metrics: Dict,
                            new_model_metrics: Dict) -> Dict:
        """Analyze test results and determine winner.
        
        Args:
            test: Test configuration dictionary
            old_model_metrics: Performance metrics for old model
            new_model_metrics: Performance metrics for new model
        
        Returns:
            Test results analysis
        """
        total_requests = test["old_model_requests"] + test["new_model_requests"]
        old_error_rate = test["old_model_errors"] / max(1, test["old_model_requests"])
        new_error_rate = test["new_model_errors"] / max(1, test["new_model_requests"])
        
        analysis = {
            "test_id": test["test_id"],
            "total_requests": total_requests,
            "has_sufficient_data": total_requests >= self.min_test_size,
            "test_statistics": {
                "old_model_error_rate": old_error_rate,
                "new_model_error_rate": new_error_rate,
                "old_model_requests": test["old_model_requests"],
                "new_model_requests": test["new_model_requests"]
            },
            "winner": None,
            "recommendation": None,
            "is_significant": False
        }
        
        # Compare metrics
        comparison = self.comparator.compare_models(old_model_metrics, new_model_metrics)
        analysis["metrics_comparison"] = comparison
        
        # Perform statistical test
        if total_requests >= self.min_test_size:
            # Chi-square test for error rates
            from scipy.stats import chi2_contingency
            
            contingency_table = [
                [test["old_model_requests"] - test["old_model_errors"], test["old_model_errors"]],
                [test["new_model_requests"] - test["new_model_errors"], test["new_model_errors"]]
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                analysis["statistical_test"] = {
                    "chi2": chi2,
                    "p_value": p_value,
                    "significant": p_value < (1 - self.confidence_level)
                }
                analysis["is_significant"] = p_value < (1 - self.confidence_level)
            except Exception as e:
                logger.warning(f"Statistical test failed: {e}")
        
        # Determine winner
        if comparison["overall_winner"] == "new":
            analysis["winner"] = test["new_model_id"]
            analysis["recommendation"] = "Deploy new model"
        elif comparison["overall_winner"] == "old":
            analysis["winner"] = test["old_model_id"]
            analysis["recommendation"] = "Keep old model"
        else:
            analysis["winner"] = None
            analysis["recommendation"] = "No clear winner; need more data or manual review"
        
        return analysis
    
    def save_test_results(self, analysis: Dict, 
                         output_path: str = "model_registry/ab_tests.jsonl"):
        """Save A/B test results for audit trail.
        
        Args:
            analysis: Test analysis results
            output_path: Path to save results
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "a") as f:
                f.write(json.dumps(analysis) + "\n")
            
            logger.info(f"Saved A/B test results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save A/B test results: {e}")


class RollbackManager:
    """Manage model rollback on failures or performance degradation."""
    
    def __init__(self, model_registry_path: str = "model_registry"):
        """Initialize rollback manager.
        
        Args:
            model_registry_path: Path to model registry
        """
        self.model_registry_path = Path(model_registry_path)
        self.rollback_log_path = self.model_registry_path / "rollback_events.jsonl"
    
    def can_rollback(self, current_model_id: str) -> bool:
        """Check if rollback is possible (previous model exists).
        
        Args:
            current_model_id: Current model ID
        
        Returns:
            True if previous model exists
        """
        # In production, check model registry for previous version
        # For now, assume we can always rollback to a backup
        return True
    
    def execute_rollback(self, 
                        current_model_id: str,
                        previous_model_id: str,
                        reason: str) -> Dict:
        """Execute rollback to previous model version.
        
        Args:
            current_model_id: Current (failing) model ID
            previous_model_id: Previous (backup) model ID
            reason: Reason for rollback
        
        Returns:
            Rollback event dictionary
        """
        rollback_event = {
            "timestamp": datetime.now().isoformat(),
            "type": "rollback",
            "from_model": current_model_id,
            "to_model": previous_model_id,
            "reason": reason,
            "status": "completed"
        }
        
        # Log rollback event
        try:
            self.model_registry_path.mkdir(parents=True, exist_ok=True)
            with open(self.rollback_log_path, "a") as f:
                f.write(json.dumps(rollback_event) + "\n")
            
            logger.warning(f"Rollback executed: {reason}")
        except Exception as e:
            logger.error(f"Failed to log rollback event: {e}")
        
        return rollback_event
    
    def should_trigger_rollback(self, error_rate: float,
                               baseline_error_rate: float,
                               threshold: float = 0.1) -> bool:
        """Determine if rollback should be triggered.
        
        Args:
            error_rate: Current error rate
            baseline_error_rate: Baseline error rate
            threshold: Degradation threshold
        
        Returns:
            True if rollback should be triggered
        """
        if baseline_error_rate == 0:
            return error_rate > 0.1
        
        degradation = (error_rate - baseline_error_rate) / baseline_error_rate
        return degradation > threshold
    
    def get_rollback_history(self, limit: int = 50) -> List[Dict]:
        """Get rollback history."""
        events = []
        if not self.rollback_log_path.exists():
            return events
        
        try:
            with open(self.rollback_log_path, "r") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load rollback history: {e}")
        
        return events[-limit:] if events else []
