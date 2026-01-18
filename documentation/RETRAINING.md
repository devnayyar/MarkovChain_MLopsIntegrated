# RETRAINING: Automated Model Updates & A/B Testing

Complete guide to the automated retraining system and A/B testing framework for continuous model improvement.

---

## Table of Contents

1. [Overview](#overview)
2. [Retraining Triggers](#retraining-triggers)
3. [Scheduler Architecture](#scheduler-architecture)
4. [A/B Testing Framework](#ab-testing-framework)
5. [Retraining Workflow](#retraining-workflow)
6. [Decision Logic](#decision-logic)
7. [Monitoring & Metrics](#monitoring--metrics)
8. [Best Practices](#best-practices)
9. [Configuration](#configuration)

---

## Overview

The FINML retraining system automates model updates through:

1. **Multiple Trigger Criteria**: Scheduled, drift-based, performance-based
2. **A/B Testing**: Compare candidate vs. baseline before deployment
3. **Automatic Promotion**: Winner automatically moves to production
4. **Rollback Support**: Easy reversion to previous model if issues arise

### Why Retraining Matters

Machine learning models degrade over time due to:
- **Data Drift**: Input distribution changes
- **Concept Drift**: Relationship between features and target changes
- **Model Drift**: Prediction quality degrades

FINML automatically detects and addresses these issues.

---

## Retraining Triggers

### Four Independent Trigger Mechanisms

#### 1. Scheduled Retraining (Time-Based)

**What**: Automatically retrain on a regular schedule

**Configuration**:
```yaml
retraining:
  schedule_interval_days: 7  # Weekly
```

**Code**:
```python
def check_scheduled_retraining(last_retrain_time: str) -> bool:
    """
    Check if scheduled retraining is due.
    
    Args:
        last_retrain_time: ISO formatted timestamp
    
    Returns:
        True if days_since_retrain >= interval
    """
    last_time = datetime.fromisoformat(last_retrain_time)
    days_since = (datetime.now() - last_time).days
    return days_since >= self.schedule_interval_days
```

**When to Use**: Regular model refresh, production deployment

#### 2. Drift-Triggered Retraining (Statistical)

**What**: Trigger when input data distribution changes

**Metrics Used**:
- Kolmogorov-Smirnov (KS) statistic: Compares distributions
- Regime transition probability drift: Changes in state transition patterns

**Configuration**:
```yaml
drift_detection:
  ks_statistic_threshold: 0.15      # KS stat > 0.15 triggers retrain
  transition_drift_threshold: 0.10  # Transition prob change > 0.10
```

**Code**:
```python
def check_drift_triggered_retraining(drift_metrics: Dict) -> Tuple[bool, str]:
    """
    Detect data drift and trigger retraining if needed.
    
    Args:
        drift_metrics: {
            'regime_drift': 0.18,
            'transition_drift': 0.08
        }
    
    Returns:
        (should_retrain, reason)
    """
    regime_drift = drift_metrics.get("regime_drift", 0.0)
    
    if regime_drift > self.drift_threshold:
        return True, f"Regime drift {regime_drift:.4f} exceeds threshold"
    
    return False, None
```

**Example Scenario**:
```
Normal times: KS stat = 0.06 (no action)
Market stress: KS stat = 0.18 (TRIGGER RETRAINING)
Crisis: KS stat = 0.35 (URGENT RETRAIN)
```

#### 3. Performance-Triggered Retraining (Quality-Based)

**What**: Trigger when model accuracy degrades

**Configuration**:
```yaml
retraining:
  performance_threshold: 0.05  # 5% accuracy drop triggers retrain
```

**Code**:
```python
def check_performance_triggered_retraining(
    current_metrics: Dict,
    baseline_metrics: Dict
) -> Tuple[bool, str]:
    """
    Detect performance degradation.
    
    Args:
        current_metrics: {'accuracy': 0.82, ...}
        baseline_metrics: {'accuracy': 0.87, ...}
    
    Returns:
        (should_retrain, reason)
    """
    accuracy_drop = baseline_metrics['accuracy'] - current_metrics['accuracy']
    
    if accuracy_drop > self.performance_threshold:
        return True, f"Accuracy dropped {accuracy_drop:.1%}"
    
    return False, None
```

**Example Scenario**:
```
Week 1: Accuracy = 87% (baseline)
Week 2: Accuracy = 85% (2% drop, no action)
Week 3: Accuracy = 81% (6% drop, TRIGGER RETRAIN)
```

#### 4. Data Quality Triggered Retraining

**What**: Trigger when input data quality falls below threshold

**Configuration**:
```yaml
retraining:
  data_quality_threshold: 0.80  # 80% quality minimum
```

**Code**:
```python
def check_data_quality_triggered_retraining(
    quality_score: float
) -> Tuple[bool, str]:
    """
    Detect data quality issues.
    
    Args:
        quality_score: 0.75 (75% quality)
    
    Returns:
        (should_retrain, reason)
    """
    if quality_score < self.data_quality_threshold:
        return True, f"Data quality {quality_score:.1%} below threshold"
    
    return False, None
```

---

## Scheduler Architecture

**File**: `retraining/scheduler.py`

### Core Class: AdvancedRetrainingScheduler

```python
class AdvancedRetrainingScheduler:
    """Manages model retraining with multiple trigger mechanisms."""
    
    def __init__(self,
                 schedule_interval_days: int = 7,
                 drift_threshold: float = 0.15,
                 performance_threshold: float = 0.05,
                 data_quality_threshold: float = 0.80):
        """Initialize with configurable thresholds."""
        self.schedule_interval_days = schedule_interval_days
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.data_quality_threshold = data_quality_threshold
```

### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `check_retraining_needed()` | Determine if any trigger fires | bool |
| `trigger_retraining()` | Start training new model | Model |
| `validate_new_model()` | Check new model quality | bool |
| `schedule_ab_test()` | Setup A/B test comparison | None |
| `record_retraining_event()` | Log to history | None |

### Decision Logic

```python
def should_retrain(self) -> Tuple[bool, List[str]]:
    """
    Aggregate all trigger criteria.
    
    Returns:
        (should_retrain, list_of_triggered_conditions)
    """
    triggers = []
    
    # Check all four criteria
    if self.check_scheduled_retraining():
        triggers.append("scheduled")
    
    if self.check_drift_triggered_retraining(drift_metrics)[0]:
        triggers.append("drift")
    
    if self.check_performance_triggered_retraining(metrics)[0]:
        triggers.append("performance")
    
    if self.check_data_quality_triggered_retraining(quality)[0]:
        triggers.append("quality")
    
    return len(triggers) > 0, triggers
```

---

## A/B Testing Framework

**File**: `retraining/ab_testing.py`

### Purpose

Compare candidate (new) model against baseline (current production) before deployment.

### A/B Test Workflow

```
┌──────────────────────────────┐
│  New Model Trained           │
│  (Candidate)                 │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  Setup A/B Test              │
│  - Traffic split (95/5)      │
│  - Baseline model (95%)      │
│  - Candidate model (5%)      │
│  - Duration: 1 week          │
└───────────┬──────────────────┘
            │
            ▼
┌──────────────────────────────┐
│  Route Predictions           │
│  - Collect metrics from both │
│  - Log results               │
│  - Monitor performance       │
└───────────┬──────────────────┘
            │
            ▼
┌──────────┬──────────────────┐
│          │                  │
Candidate  Candidate    Baseline
Better  Worse/Equal   Better
│          │              │
▼          ▼              ▼
Deploy     Archive     Continue
to Prod    Model       with Baseline
└──────────┴──────────────────┘
```

### A/B Test Parameters

```python
class ABTest:
    """Manages A/B test execution and evaluation."""
    
    def __init__(self,
                 baseline_model_id: str,
                 candidate_model_id: str,
                 traffic_split: float = 0.05,  # 5% to candidate
                 test_duration_days: int = 7,
                 min_samples: int = 100):
        """
        Initialize A/B test.
        
        Args:
            baseline_model_id: Production model
            candidate_model_id: New model to test
            traffic_split: Fraction of traffic to candidate (0.05 = 5%)
            test_duration_days: How long to run test
            min_samples: Minimum predictions before deciding
        """
```

### Test Metrics

Compared metrics between baseline and candidate:

| Metric | Baseline | Candidate | Winner |
|--------|----------|-----------|--------|
| Accuracy | 87.2% | 88.5% | Candidate ✓ |
| Latency | 145ms | 152ms | Baseline |
| Coverage | 99.8% | 99.7% | Baseline |
| Spectral Gap | 0.345 | 0.382 | Candidate ✓ |

### Decision Criteria

```python
def decide_winner(self, baseline_metrics: Dict, candidate_metrics: Dict) -> str:
    """
    Decide which model wins the A/B test.
    
    Candidate wins if:
    1. Accuracy improvement > 1.0%
    2. AND no statistically significant degradation in other metrics
    3. AND latency increase < 10%
    """
    accuracy_improvement = (
        candidate_metrics['accuracy'] - baseline_metrics['accuracy']
    )
    latency_increase = (
        (candidate_metrics['latency'] - baseline_metrics['latency']) / 
        baseline_metrics['latency']
    )
    
    # Candidate must improve accuracy and not hurt latency
    if accuracy_improvement > 0.01 and latency_increase < 0.10:
        return "candidate"
    else:
        return "baseline"
```

### Example A/B Test

```python
from retraining.ab_testing import ABTest

# Create A/B test
ab_test = ABTest(
    baseline_model_id="markov_v1.2.3",
    candidate_model_id="markov_v1.2.4",
    traffic_split=0.05,  # 5% to candidate
    test_duration_days=7
)

# Run test
results = ab_test.run_test(production_data)

# Evaluate
print(f"Baseline Accuracy: {results['baseline']['accuracy']:.4f}")
print(f"Candidate Accuracy: {results['candidate']['accuracy']:.4f}")

# Decide winner
winner = ab_test.decide_winner(
    results['baseline'],
    results['candidate']
)

if winner == "candidate":
    print("✓ Promoting candidate to production")
    ab_test.promote_to_production(candidate_model_id)
else:
    print("✗ Keeping baseline in production")
    ab_test.archive_candidate(candidate_model_id)
```

---

## Retraining Workflow

### Complete End-to-End Process

```
MONITORING LAYER
├─ Detects drift/performance degradation
├─ Calculates trigger metrics
└─ Calls scheduler.check_retraining_needed()
        │
        ▼
SCHEDULER
├─ Evaluates all four trigger criteria
├─ If triggered: calls trigger_retraining()
└─ Creates retraining job
        │
        ▼
DATA PREPARATION
├─ Load latest gold layer data
├─ Verify data quality
└─ Split into train/test
        │
        ▼
MODEL TRAINING
├─ Create new MarkovChain model
├─ Estimate transition matrix
├─ Calculate evaluation metrics
└─ Save model to MLflow
        │
        ▼
A/B TEST SETUP
├─ Baseline = current production model
├─ Candidate = newly trained model
├─ Configure traffic split (e.g., 95/5)
└─ Duration = 1 week
        │
        ▼
A/B TEST EXECUTION
├─ Route inference to both models
├─ Collect metrics
├─ Monitor performance
└─ Log predictions
        │
        ▼
A/B TEST EVALUATION
├─ Compare accuracy, latency, etc.
├─ Run statistical tests
└─ Determine winner
        │
        ▼
PROMOTION DECISION
├─ If candidate wins:
│   ├─ Promote to production
│   ├─ Archive previous version
│   └─ Update serving layer
├─ If baseline wins:
│   ├─ Archive candidate
│   └─ Continue with baseline
└─ Log decision & metrics
        │
        ▼
MONITORING
└─ Continue monitoring new model
   for next retraining trigger
```

### Code Implementation

```python
from retraining.scheduler import AdvancedRetrainingScheduler
from retraining.ab_testing import ABTest
from modeling.models.base_markov import MarkovChain
from serving.experiment_tracker import log_experiment_run

def full_retraining_workflow():
    """Execute complete retraining pipeline."""
    
    # 1. Check if retraining needed
    scheduler = AdvancedRetrainingScheduler()
    should_retrain, triggers = scheduler.should_retrain()
    
    if not should_retrain:
        logger.info("No retraining needed")
        return
    
    logger.info(f"Retraining triggered by: {triggers}")
    
    # 2. Prepare data
    gold_data = pd.read_csv('data/gold/features_final.csv')
    train_data = gold_data[:-100]  # Reserve last 100 for testing
    test_data = gold_data[-100:]
    
    # 3. Train candidate model
    candidate = MarkovChain(train_data['regime'].values)
    candidate.estimate_transition_matrix()
    candidate_metrics = evaluate_model(candidate, test_data)
    
    logger.info(f"Candidate accuracy: {candidate_metrics['accuracy']:.4f}")
    
    # 4. Log to MLflow
    run_id = log_experiment_run(
        experiment_name='markov_chain_baseline',
        metrics=candidate_metrics,
        model=candidate,
        tags={'stage': 'candidate', 'trigger': triggers[0]}
    )
    
    # 5. Setup A/B test
    ab_test = ABTest(
        baseline_model_id=get_production_model_id(),
        candidate_model_id=run_id,
        traffic_split=0.05,
        test_duration_days=7
    )
    
    logger.info("A/B test started")
    
    # 6. After 7 days, evaluate
    ab_results = ab_test.evaluate_results()
    winner = ab_test.decide_winner(
        ab_results['baseline'],
        ab_results['candidate']
    )
    
    # 7. Promote winner
    if winner == "candidate":
        logger.info("✓ Promoting candidate to production")
        ab_test.promote_to_production(candidate_model_id=run_id)
    else:
        logger.info("✗ Archiving candidate, keeping baseline")
        ab_test.archive_candidate(candidate_model_id=run_id)
    
    # 8. Record event
    scheduler.record_retraining_event({
        'timestamp': datetime.now().isoformat(),
        'trigger': triggers[0],
        'baseline_accuracy': ab_results['baseline']['accuracy'],
        'candidate_accuracy': ab_results['candidate']['accuracy'],
        'winner': winner,
        'model_id': run_id
    })

# Schedule as weekly job
if __name__ == "__main__":
    full_retraining_workflow()
```

---

## Decision Logic

### Retraining Decision Tree

```
Is any retraining trigger active?
│
├─ NO → Continue monitoring, check again next hour
│
└─ YES → Proceed with retraining
    │
    ├─ Train candidate model
    │
    ├─ Candidate accuracy > baseline accuracy?
    │   │
    │   ├─ YES → Setup A/B test
    │   │   │
    │   │   ├─ Run for 7 days
    │   │   │
    │   │   ├─ Candidate significantly better?
    │   │   │   ├─ YES → Promote to production ✓
    │   │   │   └─ NO → Keep baseline
    │   │   │
    │   │   └─ Record metrics & decision
    │   │
    │   └─ NO → Archive candidate, keep baseline
    │
    └─ Update retraining history
```

### Promotion Criteria

Candidate gets promoted if:

1. **Accuracy**: `candidate_acc > baseline_acc + 0.01` (>1% improvement)
2. **Stability**: `candidate_spectral_gap` is reasonable (not too persistent)
3. **Latency**: `candidate_latency < baseline_latency * 1.1` (≤10% slower)
4. **Robustness**: No significant degradation on minority classes
5. **Holdout Test**: Passes validation on unseen data

---

## Monitoring & Metrics

### Retraining Job Tracking

```json
{
  "job_id": "retrain_20260117_001",
  "timestamp": "2026-01-17T10:30:00",
  "trigger_type": "performance",
  "trigger_value": 0.062,
  "baseline_model": "markov_v1.2.3",
  "candidate_model": "markov_v1.2.4",
  "training_time_seconds": 245.3,
  "baseline_metrics": {
    "accuracy": 0.8724,
    "spectral_gap": 0.3456,
    "latency_ms": 145
  },
  "candidate_metrics": {
    "accuracy": 0.8891,
    "spectral_gap": 0.3521,
    "latency_ms": 151
  },
  "ab_test_duration_days": 7,
  "ab_test_traffic_split": 0.05,
  "decision": "promote",
  "decision_timestamp": "2026-01-24T10:30:00"
}
```

**Stored in**: `model_registry/retraining_jobs.jsonl`

### Monitoring Metrics

Track during retraining:

| Metric | Monitored | Threshold |
|--------|-----------|-----------|
| Training Time | Duration of model training | < 5 min |
| Data Quality | Input data quality score | > 0.80 |
| Candidate Accuracy | New model accuracy | > baseline |
| Convergence | Model fitting convergence | < 100 iterations |
| Spectral Gap | Eigenvalue gap | 0.1 - 0.5 |

### Alerts

```python
# Alert configurations
ALERTS = {
    "training_timeout": {
        "condition": "training_time > 300s",
        "severity": "HIGH",
        "action": "Abort retraining, investigate"
    },
    "candidate_worse": {
        "condition": "candidate_acc < baseline_acc - 0.02",
        "severity": "MEDIUM",
        "action": "Archive candidate, investigate"
    },
    "data_quality_low": {
        "condition": "data_quality < 0.70",
        "severity": "CRITICAL",
        "action": "Block retraining, fix data pipeline"
    }
}
```

---

## Best Practices

### 1. Conservative Traffic Split

Start A/B tests with small traffic share to minimize risk:

```python
# Good progression
Day 1-2: 1% to candidate (verify no crashes)
Day 3-4: 5% to candidate (monitor metrics)
Day 5-7: 10% to candidate (final evaluation)
```

### 2. Minimum Sample Size

Ensure statistical significance:

```python
# Wait for minimum samples before deciding
min_samples = 1000  # predictions

if len(predictions) < min_samples:
    logger.info(f"Waiting for more data: {len(predictions)}/{min_samples}")
    continue_ab_test()
else:
    decide_winner()  # Safe to decide
```

### 3. Holdout Validation

Always validate on truly unseen data:

```python
# Split chronologically (no data leakage)
train_data = gold_data[:-14 days]
validation_data = gold_data[-14 days:-7 days]
test_data = gold_data[-7 days:]

# Train on train_data
# Tune on validation_data  
# Final eval on test_data
```

### 4. Automated Rollback

Detect and revert bad models:

```python
def monitor_production_model():
    """Monitor deployed model, rollback if needed."""
    while True:
        metrics = get_production_metrics()
        
        if metrics['accuracy'] < baseline_accuracy - 0.05:
            logger.critical("Model degradation detected!")
            rollback_to_previous_model()
            trigger_urgent_retraining()
```

### 5. Regular Retraining Schedule

Even without triggers, retrain regularly:

```python
# Weekly retraining ensures model freshness
schedule_interval_days = 7
```

---

## Configuration

### Example Configuration File

**config/retraining_config.yaml**:

```yaml
retraining:
  # Schedule-based
  schedule_enabled: true
  schedule_interval_days: 7  # Weekly
  
  # Drift-based
  drift_enabled: true
  drift_threshold: 0.15
  transition_drift_threshold: 0.10
  
  # Performance-based
  performance_enabled: true
  performance_threshold: 0.05  # 5% accuracy drop
  
  # Quality-based
  quality_enabled: true
  data_quality_threshold: 0.80
  
  # Constraints
  min_data_points: 100  # Need >100 new samples before retraining
  max_retrain_frequency_hours: 6  # Don't retrain more than every 6 hours
  
ab_testing:
  enabled: true
  traffic_split: 0.05  # 5% to candidate
  test_duration_days: 7
  min_samples_for_decision: 500
  
  # Decision criteria
  accuracy_improvement_threshold: 0.01  # 1% improvement needed
  latency_increase_tolerance: 0.10  # 10% slower is OK
  
logging:
  log_to_file: model_registry/retraining_jobs.jsonl
  log_to_mlflow: true
```

---

## Summary

The FINML retraining system provides:

1. **Automated Triggers**: Multiple criteria for detecting when to retrain
2. **A/B Testing**: Safe evaluation of new models before production
3. **Risk Management**: Conservative rollouts with automatic fallback
4. **Comprehensive Logging**: Full audit trail of all retraining decisions
5. **Production Ready**: Robust error handling and monitoring

This ensures models stay fresh and accurate in production without manual intervention.
