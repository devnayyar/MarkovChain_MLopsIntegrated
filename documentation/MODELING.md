# MODELING: Markov Chain Models & ML Pipeline

Comprehensive guide to the Markov model implementation, model evaluation, experiments, and feature analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Markov Chain Theory](#markov-chain-theory)
3. [Models Structure](#models-structure)
4. [Base Markov Chain Model](#base-markov-chain-model)
5. [Absorbing Markov Chain Model](#absorbing-markov-chain-model)
6. [Model Evaluation](#model-evaluation)
7. [Experiments Framework](#experiments-framework)
8. [Feature Analysis](#feature-analysis)
9. [Training Pipeline](#training-pipeline)
10. [Performance Metrics](#performance-metrics)
11. [Integration with MLflow](#integration-with-mlflow)

---

## Overview

The FINML modeling system implements sophisticated **first-order Markov Chain models** for financial regime detection and transition prediction. The system supports multiple model variants with comprehensive evaluation and experimentation frameworks.

### Key Components:

- **Base Markov Chain**: Standard first-order Markov model
- **Absorbing Markov Chain**: Extended model handling absorbing states (crises, defaults)
- **Evaluation Suite**: Comprehensive metrics and cross-validation
- **Experiments Framework**: Baseline, sensitivity analysis, ablation studies
- **Feature Analysis**: Regime impact, transition patterns, stability metrics

### Models Are Used For:

1. **Regime Detection**: Classify current market state (Low/Medium/High Risk)
2. **Transition Prediction**: Predict next regime with probabilities
3. **Risk Assessment**: Quantify market stress and crisis probability
4. **Portfolio Monitoring**: Alert on regime changes
5. **Strategy Selection**: Choose trading strategy based on current regime

---

## Markov Chain Theory

### What is a Markov Chain?

A Markov chain is a **stochastic process** where the next state depends only on the current state, not on the history.

**Key Property (Markov Property):**
$$P(X_{t+1} = s' | X_t = s, X_{t-1}, X_{t-2}, ...) = P(X_{t+1} = s' | X_t = s)$$

### Transition Probability Matrix

The core of a Markov chain is the **transition probability matrix P**:

$$P = \begin{bmatrix} 
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{bmatrix}$$

Where:
- $p_{ij}$ = probability of transitioning from state $i$ to state $j$
- Each row sums to 1 (rows are probability distributions)
- Non-negative entries (probabilities ≥ 0)

### Example: 3-State Financial Regime

```
States:
- State 0: LOW_RISK (normal market)
- State 1: MEDIUM_RISK (elevated stress)
- State 2: HIGH_RISK (crisis)

Transition Matrix:
             Low    Medium  High
Low         [0.70   0.25   0.05]
Medium      [0.20   0.60   0.20]
High        [0.10   0.40   0.50]

Interpretation:
- From Low Risk: 70% stay low, 25% become medium, 5% become high
- From Medium: 20% improve, 60% stay medium, 20% worsen
- From High: 10% improve to low, 40% to medium, 50% stay high
```

### Stationary Distribution

The **stationary distribution** π is the long-run probability of each state:

$$\pi = \pi \cdot P$$
$$\pi \cdot \mathbf{1} = 1$$

This answers: "In the very long run, what's the probability of being in each state?"

**Financial Interpretation**: Despite fluctuations, markets tend toward an average stress level.

### Spectral Gap

The **spectral gap** is the difference between the largest and second-largest eigenvalues:

$$\text{Gap} = \lambda_1 - \lambda_2$$

**Interpretation**:
- Smaller gap → Faster convergence to stationary distribution
- Larger gap → More persistence in current state
- Gap close to 1 → Strong mean reversion
- Gap close to 0 → High persistence/momentum

### Sojourn Time (Expected Duration)

Average time spent in a state before leaving:

$$\text{Sojourn Time}_i = \frac{1}{1 - p_{ii}}$$

Where $p_{ii}$ is the self-transition probability.

**Example**: If $p_{ii} = 0.7$, sojourn time = 1/(1-0.7) = 3.33 periods

---

## Models Structure

```
modeling/
├── models/                          # Core model implementations
│   ├── base_markov.py              # Base first-order Markov chain
│   ├── absorbing_markov.py         # Absorbing state analysis
│   └── __init__.py                 # Package exports
│
├── evaluation/                      # Model evaluation metrics
│   ├── metrics.py                  # Core evaluation metrics
│   ├── comparison.py               # Model comparison framework
│   ├── cross_validation.py         # Time-series CV
│   └── __init__.py
│
├── experiments/                     # Experimental configurations
│   ├── baseline_experiment.py      # Standard model training
│   ├── sensitivity_analysis.py     # Hyperparameter sensitivity
│   ├── ablation_study.py           # Component importance
│   └── __init__.py
│
├── feature_analysis/                # Feature importance & stability
│   ├── regime_impact.py            # How regimes affect predictions
│   ├── transition_analysis.py      # Transition pattern analysis
│   ├── stability_metrics.py        # Model stability over time
│   └── __init__.py
│
└── __init__.py                     # Top-level package exports
```

---

## Base Markov Chain Model

**File**: `modeling/models/base_markov.py`

### Purpose

Implements a standard first-order Markov chain for regime modeling.

### Key Class: MarkovChain

#### Initialization

```python
from modeling.models.base_markov import MarkovChain

# Create from state sequence
state_sequence = np.array([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, ...])
states = ['LOW', 'MEDIUM', 'HIGH']

model = MarkovChain(state_sequence, states)
```

#### Core Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `estimate_transition_matrix()` | Compute P from historical data | np.ndarray (n_states × n_states) |
| `get_stationary_distribution()` | Long-run state probabilities | np.ndarray (n_states,) |
| `get_spectral_gap()` | Convergence rate metric | float |
| `get_sojourn_times()` | Expected duration in each state | dict |
| `predict_next_state(current_state)` | Single-step prediction | str (state name) |
| `predict_next_state_probs(current_state)` | Prediction probabilities | np.ndarray |
| `simulate_path(steps, start_state)` | Generate synthetic paths | np.ndarray |
| `save_model(filepath)` | Persist model | None |
| `load_model(filepath)` | Load saved model | MarkovChain |

### Implementation Details

#### Transition Matrix Estimation

```python
def estimate_transition_matrix(self) -> np.ndarray:
    """
    Count transitions from historical data:
    1. For each time t, count: current_state → next_state
    2. Build contingency table (transition counts)
    3. Normalize rows to get probabilities
    """
    # Count transitions
    for t in range(len(self.state_sequence) - 1):
        current = self.state_sequence[t]
        next_state = self.state_sequence[t + 1]
        transition_counts[current, next_state] += 1
    
    # Normalize to probabilities
    P = transition_counts / transition_counts.sum(axis=1)
    return P
```

**Key Properties**:
- Estimated from historical data (data-driven)
- Row-stochastic (each row sums to 1)
- Non-negative entries
- Handles unseen transitions gracefully

#### Spectral Analysis

```python
def get_spectral_gap(self) -> float:
    """
    Compute eigenvalues of transition matrix P.
    Gap = λ₁ - λ₂ where λ₁=1 (stationary)
    """
    eigenvalues = np.linalg.eigvals(self.transition_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    return spectral_gap
```

**Interpretation**:
- Gap = 0.1 → Fast mixing (rapid state changes)
- Gap = 0.5 → Moderate mixing
- Gap = 0.9 → Slow mixing (persistent states)

#### Sojourn Time Calculation

```python
def get_sojourn_times(self) -> dict:
    """
    For each state i:
    Sojourn_time_i = 1 / (1 - p_ii)
    
    p_ii = probability of staying in state i
    """
    sojourn_times = {}
    for i, state in enumerate(self.states):
        p_ii = self.transition_matrix[i, i]
        # Avoid division by zero
        if p_ii >= 1.0:
            sojourn_times[state] = np.inf  # Absorbing state
        else:
            sojourn_times[state] = 1.0 / (1.0 - p_ii)
    return sojourn_times
```

### Usage Example

```python
import numpy as np
from modeling.models.base_markov import MarkovChain

# Historical regime sequence
regimes = ['LOW', 'LOW', 'MEDIUM', 'MEDIUM', 'HIGH', 'MEDIUM', 'LOW', ...]

# Create model
mc = MarkovChain(np.array(regimes))

# Estimate transitions
P = mc.estimate_transition_matrix()
print("Transition Matrix:\n", P)

# Get long-run probabilities
pi = mc.get_stationary_distribution()
print("Stationary Distribution:", pi)

# Get convergence speed
gap = mc.get_spectral_gap()
print(f"Spectral Gap: {gap:.4f}")

# Get expected durations
sojourns = mc.get_sojourn_times()
print("Sojourn Times:", sojourns)

# Predict next state from current state
current = 'MEDIUM'
next_probs = mc.predict_next_state_probs(current)
print(f"From {current}: {next_probs}")

# Generate synthetic paths
path = mc.simulate_path(steps=100, start_state='LOW')
print(f"Simulated path (100 steps): {path}")
```

---

## Absorbing Markov Chain Model

**File**: `modeling/models/absorbing_markov.py`

### Purpose

Extended model for handling **absorbing states** (states you can't leave from).

**Examples of Absorbing States**:
- Default (in credit models)
- Crisis/Collapse (in financial stress)
- Recession (can't escape on its own)

### Key Class: AbsorbingMarkovChain

#### Initialization

```python
from modeling.models.absorbing_markov import AbsorbingMarkovChain

# With explicit absorbing states
absorbing_mc = AbsorbingMarkovChain(
    transition_matrix=P,
    state_names=['LOW', 'MEDIUM', 'HIGH', 'DEFAULT'],
    absorbing_states=['DEFAULT']  # Can't escape default
)
```

#### Core Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `fundamental_matrix()` | Expected visits before absorption | np.ndarray |
| `absorption_probability_matrix()` | Probability of reaching each absorbing state | pd.DataFrame |
| `expected_time_to_absorption()` | Steps until absorption from each state | pd.Series |
| `recovery_probability()` | Probability of recovering from transient state | float |

### Mathematical Framework

#### Canonical Form

Reorder matrix to:
$$P = \begin{bmatrix} Q & R \\ 0 & I \end{bmatrix}$$

Where:
- **Q**: Transient → Transient transitions
- **R**: Transient → Absorbing transitions
- **0**: No transitions from absorbing to transient
- **I**: Absorbing states stay in themselves

#### Fundamental Matrix

$$N = (I - Q)^{-1}$$

Where:
- $N_{ij}$ = expected number of visits to transient state $j$ starting from state $i$
- Used to compute absorption metrics

#### Absorption Probabilities

$$B = N \cdot R$$

Where:
- $B_{ij}$ = probability of eventual absorption into state $j$ starting from state $i$

### Usage Example

```python
from modeling.models.absorbing_markov import AbsorbingMarkovChain

# Create absorbing chain (with DEFAULT as absorbing state)
abs_mc = AbsorbingMarkovChain(
    transition_matrix=P,
    state_names=['LOW', 'MEDIUM', 'HIGH', 'DEFAULT'],
    absorbing_states=['DEFAULT']
)

# Get fundamental matrix (expected visits)
N = abs_mc.fundamental_matrix()
print("Fundamental Matrix:\n", N)

# Get absorption probabilities
B = abs_mc.absorption_probability_matrix()
print("Absorption Probabilities:\n", B)
# Shows: from each transient state, probability of hitting DEFAULT

# Expected steps to absorption
t = abs_mc.expected_time_to_absorption()
print("Expected Time to Default:\n", t)

# Recovery probability (escape absorbing state)
# Note: In true absorbing chains, this is 0
# But useful for quasi-absorbing analysis
recovery = abs_mc.recovery_probability()
print(f"Recovery Probability: {recovery:.4f}")
```

---

## Model Evaluation

**File**: `modeling/evaluation/metrics.py`, `comparison.py`, `cross_validation.py`

### Purpose

Comprehensive evaluation framework to validate model quality.

### Core Evaluation Metrics

#### 1. Accuracy Metrics

```python
# Prediction Accuracy
accuracy = (predictions == actual).mean()

# Per-state accuracy
per_state_accuracy = {
    state: accuracy[actual == state]
    for state in states
}
```

#### 2. Markov-Specific Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Spectral Gap** | $\lambda_1 - \lambda_2$ | Convergence speed (larger = slower) |
| **Sojourn Time** | $1/(1-p_{ii})$ | Expected duration in each state |
| **Stationary Dist.** | $\pi = \pi P$ | Long-run regime probabilities |
| **KL Divergence** | $\sum_i p_i \log(p_i/q_i)$ | Distance between distributions |
| **Wasserstein Distance** | Earth Mover's Distance | Distance between state distributions |

#### 3. Cross-Validation

Time-series cross-validation (avoids data leakage):

```python
def time_series_cross_validation(data, n_splits=5):
    """
    Implement time-series safe cross-validation:
    
    Split 1: Train [0:n/5], Test [n/5:2n/5]
    Split 2: Train [0:2n/5], Test [2n/5:3n/5]
    ...
    
    Training set always precedes test set (no future data leaks).
    """
```

### Evaluation Methods

```python
from modeling.evaluation.metrics import ModelEvaluator

evaluator = ModelEvaluator(model)

# Get comprehensive metrics
metrics = evaluator.evaluate(
    predictions=pred_regimes,
    actual=true_regimes,
    return_dict=True
)

# Example output:
{
    'accuracy': 0.847,
    'per_state_accuracy': {'LOW': 0.92, 'MEDIUM': 0.81, 'HIGH': 0.76},
    'spectral_gap': 0.345,
    'sojourn_times': {'LOW': 3.2, 'MEDIUM': 2.8, 'HIGH': 1.9},
    'stationary_distribution': [0.40, 0.35, 0.25],
    'kl_divergence': 0.052,
    'wasserstein_distance': 0.18
}

# Compare models
comparison = evaluator.compare_models(model1, model2)
print(f"Model 1 Accuracy: {comparison['model1']['accuracy']}")
print(f"Model 2 Accuracy: {comparison['model2']['accuracy']}")
print(f"Better: {comparison['winner']}")
```

---

## Experiments Framework

**File**: `modeling/experiments/`

### Purpose

Organize and manage multiple model training experiments.

### Experiment Types

#### 1. Baseline Experiment

```python
# Standard model training
# location: baseline_experiment.py

def run_baseline_experiment(gold_data):
    """
    Train standard Markov chain:
    - Fit on full dataset
    - Evaluate on test split
    - Log metrics to MLflow
    """
    model = MarkovChain(gold_data['regime'])
    P = model.estimate_transition_matrix()
    metrics = evaluate_model(model)
    
    # Log to MLflow
    mlflow.log_metrics(metrics)
    mlflow.log_params({'model_type': 'baseline'})
    
    return model, metrics
```

#### 2. Sensitivity Analysis

```python
# Test how hyperparameters affect performance
# location: sensitivity_analysis.py

def sensitivity_analysis():
    """
    Test parameter sensitivity:
    - Varying train/test split
    - Different smoothing parameters
    - Alternative discretization thresholds
    """
    results = {}
    for train_ratio in [0.6, 0.7, 0.8, 0.9]:
        for smoothing in [0.0, 0.01, 0.05, 0.1]:
            model = train_model(
                train_ratio=train_ratio,
                smoothing=smoothing
            )
            results[(train_ratio, smoothing)] = evaluate_model(model)
    
    return results
```

#### 3. Ablation Study

```python
# Test importance of each component
# location: ablation_study.py

def ablation_study():
    """
    Remove components one at a time:
    - Without smoothing
    - Without stationary constraint
    - Without spectral analysis
    """
    baseline_score = evaluate_full_model()
    
    ablations = {
        'no_smoothing': evaluate_model(model_no_smoothing),
        'no_stationarity': evaluate_model(model_no_stationarity),
        'no_spectral': evaluate_model(model_no_spectral),
    }
    
    return ablations, baseline_score
```

### MLflow Integration

All experiments automatically log to MLflow:

```python
# Experiments registered:
- "markov_chain_baseline" - Standard training
- "markov_chain_absorbing" - Absorbing state variant
- "markov_chain_comparison" - Model comparison
- "data_sensitivity_analysis" - Hyperparameter sensitivity

# Each experiment has runs with:
- Metrics (accuracy, spectral_gap, etc.)
- Parameters (train_ratio, smoothing, etc.)
- Artifacts (model.pkl, transition_matrix.csv)
- Tags (version, stage, status)
```

---

## Feature Analysis

**File**: `modeling/feature_analysis/`

### Purpose

Analyze which features matter and how model behaves.

#### 1. Regime Impact Analysis

```python
# location: regime_impact.py

def analyze_regime_impact():
    """
    For each feature, measure its impact on regime transitions:
    - Correlation with regime changes
    - Feature importance via permutation
    - Interaction effects
    """
    impacts = {}
    for feature in features:
        # Permute feature and measure accuracy drop
        accuracy_before = model.evaluate()
        model.data[feature] = np.random.permutation(model.data[feature])
        accuracy_after = model.evaluate()
        
        impact = accuracy_before - accuracy_after
        impacts[feature] = impact
    
    return impacts
```

**Output Example:**
```
Feature Impact on Regime Detection:
- volatility: 0.145 (high impact)
- momentum: 0.082 (medium)
- correlation: 0.031 (low)
```

#### 2. Transition Pattern Analysis

```python
# location: transition_analysis.py

def analyze_transitions():
    """
    Which transitions are most common/rare/dangerous?
    - Transition frequency (Low→High vs Low→Medium)
    - Transition duration (how long does Low→High take?)
    - Transition triggers (what causes transitions?)
    """
    transitions = {
        'LOW→MEDIUM': 0.25,
        'LOW→HIGH': 0.05,
        'MEDIUM→HIGH': 0.20,
        'HIGH→MEDIUM': 0.40,
        ...
    }
    
    return transitions
```

#### 3. Stability Metrics

```python
# location: stability_metrics.py

def analyze_stability():
    """
    How stable is the model over time?
    - Train on first half, test on second half
    - Measure metric drift
    - Detect concept drift (model degradation)
    """
    metrics_first_half = model.evaluate(data[:n//2])
    metrics_second_half = model.evaluate(data[n//2:])
    
    drift = metrics_first_half - metrics_second_half
    
    return drift
```

---

## Training Pipeline

### Complete Training Workflow

```
┌─────────────────────────────────────────┐
│      Gold Layer Data (Preprocessed)     │
│     - Normalized features              │
│     - Regime labels (0,1,2)            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│   Split into Train/Test                 │
│   (Time-series safe split)              │
│   Train: [0:n*0.8], Test: [n*0.8:n]    │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Initialize MarkovChain Model           │
│  Specify number of states (usually 3)   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Estimate Transition Matrix             │
│  - Count state transitions              │
│  - Normalize to probabilities           │
│  - Handle smoothing if needed           │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Compute Spectral Properties            │
│  - Eigenvalues (spectral gap)           │
│  - Stationary distribution              │
│  - Sojourn times                        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Evaluate on Test Set                   │
│  - Prediction accuracy                  │
│  - Per-state metrics                    │
│  - Cross-validation scores              │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Log to MLflow                          │
│  - Metrics                              │
│  - Parameters                           │
│  - Model artifacts                      │
│  - Evaluation results                   │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  Save Model                             │
│  - Pickle serialization                 │
│  - Transition matrix CSV                │
│  - Metadata JSON                        │
└─────────────────────────────────────────┘
```

### Example Training Code

```python
from modeling.models.base_markov import MarkovChain
from modeling.evaluation.metrics import ModelEvaluator
from serving.experiment_tracker import log_experiment_run

def train_markov_model():
    """Complete training pipeline."""
    
    # 1. Load gold data
    gold_data = pd.read_csv('data/gold/features_final.csv')
    regimes = gold_data['regime'].values
    
    # 2. Split data (time-series safe)
    split_idx = int(0.8 * len(regimes))
    train_regimes = regimes[:split_idx]
    test_regimes = regimes[split_idx:]
    
    # 3. Create and train model
    model = MarkovChain(train_regimes)
    P = model.estimate_transition_matrix()
    
    # 4. Evaluate
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(
        predictions=model.predict_batch(test_regimes),
        actual=test_regimes
    )
    
    # 5. Log to MLflow
    log_experiment_run(
        experiment_name='markov_chain_baseline',
        metrics=metrics,
        params={'n_states': 3, 'smoothing': 0.0},
        model=model,
        tags={'version': 'v1.0', 'status': 'production'}
    )
    
    return model, metrics

# Run
model, metrics = train_markov_model()
print(f"Model Accuracy: {metrics['accuracy']:.4f}")
print(f"Spectral Gap: {metrics['spectral_gap']:.4f}")
```

---

## Performance Metrics

### Key Metrics Tracked

| Metric | Formula | Good Value | Explanation |
|--------|---------|-----------|--------------|
| **Accuracy** | Correct / Total | > 0.80 | % of correct predictions |
| **Spectral Gap** | $\lambda_1 - \lambda_2$ | 0.1 - 0.5 | Convergence speed |
| **Sojourn (Low)** | $1/(1-p_{00})$ | 3-5 | Duration in normal market |
| **Sojourn (High)** | $1/(1-p_{22})$ | 1-2 | Duration in crisis (shorter is better) |
| **KL Divergence** | $\sum p_i \log p_i/q_i$ | < 0.05 | Distance to stationary |
| **Abs. Deviation** | $\|p - q\|$ | < 0.03 | Mean absolute error |

### Interpretation Examples

```
Scenario 1: Healthy Model
- Accuracy: 0.87
- Spectral Gap: 0.32
- Interpretation: Good predictions, moderate persistence

Scenario 2: Too Persistent
- Accuracy: 0.92
- Spectral Gap: 0.92
- Problem: Model predicts same state too often (overfitting)

Scenario 3: Too Noisy
- Accuracy: 0.64
- Spectral Gap: 0.05
- Problem: Model too sensitive, jumps between states
```

---

## Integration with MLflow

### Automatic Logging

```python
import mlflow
from serving.experiment_tracker import initialize_mlflow

# Initialize MLflow
initialize_mlflow()

# In training code:
with mlflow.start_run(experiment_id=1):
    # Log parameters
    mlflow.log_param("n_states", 3)
    mlflow.log_param("smoothing", 0.01)
    
    # Log metrics
    mlflow.log_metric("accuracy", 0.847)
    mlflow.log_metric("spectral_gap", 0.345)
    mlflow.log_metric("sojourn_low", 3.2)
    
    # Log artifacts
    mlflow.log_artifact("transition_matrix.csv")
    mlflow.log_artifact("model.pkl")
    
    # Log model
    mlflow.sklearn.log_model(model, "markov_model")
```

### Query Experiments

```python
import mlflow

client = mlflow.tracking.MlflowClient()

# List all experiments
experiments = client.search_experiments()

# Search runs in experiment
runs = client.search_runs(
    experiment_ids=[1],
    filter_string="metrics.accuracy > 0.85"
)

# Get best model
best_run = runs[0]
print(f"Best accuracy: {best_run.data.metrics['accuracy']:.4f}")

# Load model
model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/markov_model")
```

---

## Summary

The FINML modeling system provides:

1. **Flexible Models**: Base and Absorbing Markov chains for different scenarios
2. **Comprehensive Evaluation**: Multiple metrics tailored for Markov models
3. **Experimentation**: Organized framework for testing variations
4. **Feature Analysis**: Understanding what drives model decisions
5. **MLflow Integration**: Automatic tracking and model management
6. **Production Ready**: Validated, tested, deployment-ready models

All models are designed for **financial regime detection** with built-in safeguards for stability, interpretability, and monitoring.
