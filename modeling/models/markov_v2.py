"""
Enhanced Markov Chain V2 - Phase 1

Improvements over V1:
1. Bayesian parameter estimation (vs. MLE)
   - Dirichlet priors for smoothing
   - Uncertainty quantification
   - More stable estimates with limited data

2. Confidence calibration
   - Isotonic regression
   - Platt scaling
   - Prediction intervals

3. Adaptive smoothing
   - Dynamic alpha tuning
   - Cross-validation for hyperparameters

4. Model metrics
   - Perplexity
   - Log-likelihood
   - AIC/BIC for model selection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
from scipy.special import digamma, polygamma
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class MarkovChainV2:
    """
    Enhanced Markov Chain with Bayesian estimation and calibration.
    
    Improvements over V1:
    - Bayesian transition matrix estimation
    - Confidence calibration
    - Uncertainty quantification
    - Better numerical stability
    """
    
    def __init__(self, 
                 state_names: List[str] = None,
                 alpha: float = 1.0,
                 min_observations: int = 50):
        """
        Initialize Markov Chain V2.
        
        Parameters:
        ───────────
        state_names : List[str]
            Names of states (e.g., ['LOW_RISK', 'MODERATE_RISK', 'HIGH_RISK'])
        alpha : float
            Dirichlet prior concentration parameter
            - Higher alpha: smoother transitions (more prior influence)
            - Lower alpha: closer to data (less smoothing)
            Default: 1.0 (uniform prior)
        min_observations : int
            Minimum transitions per state for stable estimation
        """
        self.state_names = state_names or ['LOW_RISK', 'MODERATE_RISK', 'HIGH_RISK']
        self.n_states = len(self.state_names)
        self.alpha = alpha
        self.min_observations = min_observations
        
        # Model parameters
        self.transition_matrix = None
        self.transition_counts = None
        self.stationary_dist = None
        self.log_likelihood = None
        
        # Calibration
        self.calibrator = None
        self.is_calibrated = False
        
        # Uncertainty
        self.transition_uncertainty = None
        self.prediction_intervals = {}
        
        logger.info(
            f"Initialized MarkovChainV2 with states: {self.state_names}, "
            f"alpha={alpha}"
        )
    
    def fit(self, 
            state_sequence: np.ndarray,
            use_bayesian: bool = True,
            auto_tune_alpha: bool = False) -> 'MarkovChainV2':
        """
        Fit Markov chain to state sequence.
        
        Parameters:
        ───────────
        state_sequence : np.ndarray
            Sequence of state indices (0, 1, 2, ...)
        use_bayesian : bool
            Use Bayesian estimation (default) vs MLE
        auto_tune_alpha : bool
            Automatically tune alpha via cross-validation
        
        Returns:
        ────────
        self : MarkovChainV2
            Fitted model
        """
        logger.info(f"Fitting MarkovChainV2 on {len(state_sequence)} observations")
        
        # Convert state names to indices if needed
        if isinstance(state_sequence[0], str):
            state_to_idx = {s: i for i, s in enumerate(self.state_names)}
            state_sequence = np.array([state_to_idx[s] for s in state_sequence])
        
        # Step 1: Count transitions
        self._count_transitions(state_sequence)
        
        # Step 2: Validate minimum observations
        transitions_per_state = self.transition_counts.sum(axis=1)
        if np.any(transitions_per_state < self.min_observations):
            logger.warning(
                f"Some states have few transitions: {transitions_per_state}"
            )
        
        # Step 3: Estimate transition matrix
        if auto_tune_alpha:
            logger.info("Auto-tuning alpha via cross-validation...")
            self.alpha = self._tune_alpha(state_sequence)
        
        if use_bayesian:
            self._estimate_transition_matrix_bayesian()
        else:
            self._estimate_transition_matrix_mle()
        
        # Step 4: Calculate stationary distribution
        self._calculate_stationary_distribution()
        
        # Step 5: Calculate log-likelihood
        self._calculate_log_likelihood(state_sequence)
        
        logger.info(
            f"Model fitted. Log-likelihood: {self.log_likelihood:.2f}, "
            f"Perplexity: {self.get_perplexity():.4f}"
        )
        
        return self
    
    def _count_transitions(self, state_sequence: np.ndarray):
        """
        Count state transitions.
        
        Builds transition count matrix where element [i,j]
        is the number of times we went from state i to state j.
        """
        self.transition_counts = np.zeros((self.n_states, self.n_states))
        
        for t in range(len(state_sequence) - 1):
            current = int(state_sequence[t])
            next_state = int(state_sequence[t + 1])
            
            # Validate indices
            if 0 <= current < self.n_states and 0 <= next_state < self.n_states:
                self.transition_counts[current, next_state] += 1
        
        logger.debug(f"Transition counts:\n{self.transition_counts}")
    
    def _estimate_transition_matrix_mle(self):
        """
        Maximum Likelihood Estimation (V1 method).
        
        P[i,j] = count(i→j) / count(i→any)
        
        Problem: Can give zero probabilities if no transitions observed
        """
        # Add small epsilon to avoid division by zero
        row_sums = self.transition_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        
        self.transition_matrix = self.transition_counts / row_sums
        
        logger.debug("Using MLE (V1) estimation")
    
    def _estimate_transition_matrix_bayesian(self):
        """
        Bayesian estimation using Dirichlet-Multinomial conjugacy.
        
        Prior: Dirichlet(alpha) for each row
        Likelihood: Multinomial from observed transitions
        Posterior: Dirichlet(alpha + counts)
        
        Advantages:
        - Never gives zero probabilities
        - Natural uncertainty quantification
        - More stable with few observations
        
        Formula: P[i,j] = (alpha + count[i,j]) / (n_states*alpha + count[i,:])
        """
        # Posterior parameters
        posterior_alpha = self.alpha + self.transition_counts
        
        # Compute probabilities
        row_sums = posterior_alpha.sum(axis=1, keepdims=True)
        self.transition_matrix = posterior_alpha / row_sums
        
        # Compute uncertainty (Dirichlet variance)
        # Var[P[i,j]] = P[i,j] * (1 - P[i,j]) / (sum_alpha + 1)
        self.transition_uncertainty = np.zeros_like(self.transition_matrix)
        for i in range(self.n_states):
            sum_alpha = row_sums[i, 0]
            for j in range(self.n_states):
                p_ij = self.transition_matrix[i, j]
                self.transition_uncertainty[i, j] = (
                    p_ij * (1 - p_ij) / (sum_alpha + 1)
                )
        
        logger.debug(
            f"Using Bayesian estimation with alpha={self.alpha}\n"
            f"Transition matrix:\n{self.transition_matrix}"
        )
    
    def _calculate_stationary_distribution(self):
        """
        Calculate stationary distribution (long-run probabilities).
        
        Solves: π = π * P (where P is transition matrix)
        
        This is the left eigenvector corresponding to eigenvalue 1.
        """
        # Get left eigenvector (transpose, then right eigenvector)
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize to probabilities
        self.stationary_dist = stationary / stationary.sum()
        
        logger.debug(
            f"Stationary distribution: "
            f"{dict(zip(self.state_names, self.stationary_dist))}"
        )
    
    def _calculate_log_likelihood(self, state_sequence: np.ndarray):
        """
        Calculate log-likelihood of observed sequence.
        
        L = Σ log P(state_t | state_t-1)
        """
        log_lik = 0
        
        for t in range(len(state_sequence) - 1):
            current = int(state_sequence[t])
            next_state = int(state_sequence[t + 1])
            
            if 0 <= current < self.n_states and 0 <= next_state < self.n_states:
                prob = self.transition_matrix[current, next_state]
                # Add small epsilon to avoid log(0)
                log_lik += np.log(prob + 1e-10)
        
        self.log_likelihood = log_lik
        
        return log_lik
    
    def _tune_alpha(self, 
                   state_sequence: np.ndarray,
                   alpha_range: Tuple[float, float] = (0.1, 10.0),
                   n_folds: int = 5) -> float:
        """
        Automatically tune alpha via cross-validation.
        
        Splits data into folds, trains on fold i with different alphas,
        evaluates on held-out fold. Returns alpha with best average
        performance.
        """
        fold_size = len(state_sequence) // n_folds
        alphas_to_test = np.logspace(
            np.log10(alpha_range[0]),
            np.log10(alpha_range[1]),
            5
        )
        
        cv_scores = {alpha: [] for alpha in alphas_to_test}
        
        for fold in range(n_folds):
            # Split into train/test
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(state_sequence)
            
            train_data = np.concatenate([
                state_sequence[:test_start],
                state_sequence[test_end:]
            ])
            test_data = state_sequence[test_start:test_end]
            
            for alpha in alphas_to_test:
                # Train model
                model = MarkovChainV2(state_names=self.state_names, alpha=alpha)
                model.fit(train_data, use_bayesian=True)
                
                # Evaluate on test
                log_lik = model._calculate_log_likelihood(test_data)
                cv_scores[alpha].append(log_lik)
        
        # Choose best alpha
        best_alpha = max(
            cv_scores.keys(),
            key=lambda a: np.mean(cv_scores[a])
        )
        
        logger.info(f"Best alpha from CV: {best_alpha:.4f}")
        
        return best_alpha
    
    def predict_next_state(self, 
                          current_state: str,
                          return_confidence: bool = True) -> Dict:
        """
        Predict probability distribution of next state.
        
        Parameters:
        ───────────
        current_state : str
            Current state name
        return_confidence : bool
            Include confidence/uncertainty estimates
        
        Returns:
        ────────
        {
            'regime': 'HIGH_RISK',              # Most likely next state
            'confidence': 0.82,                 # Probability of most likely
            'probabilities': {...},             # Full distribution
            'confidence_interval': (0.75, 0.89) # 95% CI if calibrated
        }
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Get state index
        state_idx = self.state_names.index(current_state)
        
        # Get transition probabilities
        probs = self.transition_matrix[state_idx].copy()
        
        # Most likely next state
        next_state_idx = np.argmax(probs)
        next_state = self.state_names[next_state_idx]
        confidence = probs[next_state_idx]
        
        # Uncertainty (standard deviation)
        if self.transition_uncertainty is not None:
            uncertainty = self.transition_uncertainty[state_idx, next_state_idx]
            ci_lower = max(0, confidence - 1.96 * np.sqrt(uncertainty))
            ci_upper = min(1, confidence + 1.96 * np.sqrt(uncertainty))
            confidence_interval = (ci_lower, ci_upper)
        else:
            confidence_interval = None
        
        return {
            'regime': next_state,
            'confidence': float(confidence),
            'probabilities': dict(zip(self.state_names, probs.astype(float))),
            'confidence_interval': confidence_interval,
            'uncertainty': self.transition_uncertainty[state_idx, next_state_idx] 
                          if self.transition_uncertainty is not None else None
        }
    
    def get_mean_sojourn_time(self, state: str) -> float:
        """
        Get average time spent in a state before transitioning.
        
        Formula: E[duration] = 1 / (1 - p_ii)
        
        where p_ii is probability of staying in state.
        """
        state_idx = self.state_names.index(state)
        stay_prob = self.transition_matrix[state_idx, state_idx]
        
        if stay_prob >= 1.0:
            return np.inf
        
        mean_duration = 1 / (1 - stay_prob)
        return mean_duration
    
    def get_absorption_probability(self, from_state: str, to_state: str) -> float:
        """
        Get probability of eventually reaching to_state from from_state.
        
        Useful for understanding regime transitions.
        """
        from_idx = self.state_names.index(from_state)
        to_idx = self.state_names.index(to_state)
        
        # Iterative computation (simulate many steps)
        probs = np.zeros(self.n_states)
        probs[from_idx] = 1.0
        
        for _ in range(1000):  # Simulate 1000 steps
            probs = probs @ self.transition_matrix
        
        return float(probs[to_idx])
    
    def get_perplexity(self) -> float:
        """
        Calculate perplexity of the model.
        
        Lower perplexity = better model
        Perplexity = exp(-log-likelihood / n_observations)
        
        Intuition: Average branching factor
        """
        if self.log_likelihood is None:
            return np.inf
        
        # Note: This is approximate without access to full sequence length
        n_transitions = self.transition_counts.sum()
        
        if n_transitions == 0:
            return np.inf
        
        perplexity = np.exp(-self.log_likelihood / n_transitions)
        return float(perplexity)
    
    def get_aic(self, n_observations: int) -> float:
        """
        Akaike Information Criterion.
        
        AIC = 2k - 2*log(L)
        where k = number of parameters, L = likelihood
        
        Lower AIC = better model (balances fit and complexity)
        """
        k = self.n_states * (self.n_states - 1)  # Free parameters
        aic = 2 * k - 2 * self.log_likelihood
        return float(aic)
    
    def get_bic(self, n_observations: int) -> float:
        """
        Bayesian Information Criterion.
        
        BIC = k*log(n) - 2*log(L)
        
        Penalizes complexity more than AIC
        """
        k = self.n_states * (self.n_states - 1)
        bic = k * np.log(n_observations) - 2 * self.log_likelihood
        return float(bic)
    
    def calibrate(self, 
                 predictions: np.ndarray,
                 labels: np.ndarray,
                 method: str = 'sigmoid'):
        """
        Calibrate confidence scores using held-out validation data.
        
        This ensures that when model predicts 85% confident,
        it's actually correct 85% of the time.
        
        Parameters:
        ───────────
        predictions : np.ndarray
            Predicted class indices
        labels : np.ndarray
            True class indices
        method : str
            'sigmoid' (Platt scaling) or 'isotonic'
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        # Create dummy base estimator (we just need calibration)
        class DummyClassifier:
            def predict_proba(self, X):
                return np.eye(3)[predictions]
        
        self.calibrator = CalibratedClassifierCV(
            DummyClassifier(),
            method=method,
            cv=5
        )
        
        self.calibrator.fit(predictions.reshape(-1, 1), labels)
        self.is_calibrated = True
        
        logger.info(f"Model calibrated using {method} method")
    
    def get_model_summary(self) -> Dict:
        """
        Get summary statistics of fitted model.
        """
        return {
            'states': self.state_names,
            'transition_matrix': self.transition_matrix,
            'stationary_distribution': dict(zip(
                self.state_names,
                self.stationary_dist
            )),
            'log_likelihood': self.log_likelihood,
            'perplexity': self.get_perplexity(),
            'mean_sojourn_times': {
                state: self.get_mean_sojourn_time(state)
                for state in self.state_names
            },
            'is_calibrated': self.is_calibrated
        }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample state sequence
    np.random.seed(42)
    
    # Create transitions based on real transition matrix
    P_true = np.array([
        [0.85, 0.12, 0.03],  # LOW -> mostly stay
        [0.15, 0.75, 0.10],  # MODERATE
        [0.10, 0.20, 0.70]   # HIGH -> mostly stay
    ])
    
    # Generate sequence
    states = [0]
    for _ in range(858):
        next_state = np.random.choice(3, p=P_true[states[-1]])
        states.append(next_state)
    
    states = np.array(states)
    
    # Fit model
    state_names = ['LOW_RISK', 'MODERATE_RISK', 'HIGH_RISK']
    model_v2 = MarkovChainV2(state_names=state_names, alpha=1.0)
    model_v2.fit(states, use_bayesian=True)
    
    print("\n" + "="*60)
    print("MARKOV CHAIN V2 RESULTS")
    print("="*60)
    
    summary = model_v2.get_model_summary()
    print(f"\nTransition Matrix:\n{summary['transition_matrix']}")
    print(f"\nStationary Distribution: {summary['stationary_distribution']}")
    print(f"\nLog-Likelihood: {summary['log_likelihood']:.2f}")
    print(f"Perplexity: {summary['perplexity']:.4f}")
    
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    
    for state in state_names:
        pred = model_v2.predict_next_state(state)
        print(f"\nFrom {state}:")
        print(f"  Most likely: {pred['regime']} ({pred['confidence']:.2%})")
        print(f"  Distribution: {pred['probabilities']}")
