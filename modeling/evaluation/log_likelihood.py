"""
Log-likelihood and model fitness evaluation for Markov chains.

Measures how well the estimated transition matrix explains observed data.
Higher likelihood = better fit.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class LogLikelihoodEvaluator:
    """Computes log-likelihood of regime sequences under Markov model."""
    
    def __init__(self, transition_matrix: np.ndarray, state_names: List[str],
                 stationary_dist: np.ndarray = None):
        """
        Initialize evaluator.
        
        Args:
            transition_matrix: Estimated transition matrix T[i,j] = P(j|i)
            state_names: List of state names
            stationary_dist: Stationary distribution (estimated if None)
        """
        self.P = np.asarray(transition_matrix)
        self.state_names = state_names
        self.state_to_idx = {s: i for i, s in enumerate(state_names)}
        self.n_states = len(state_names)
        
        # Estimate stationary distribution if not provided
        if stationary_dist is None:
            self.stationary_dist = self._estimate_stationary()
        else:
            self.stationary_dist = np.asarray(stationary_dist)
        
        logger.info(f"Initialized log-likelihood evaluator with {self.n_states} states")
    
    def _estimate_stationary(self) -> np.ndarray:
        """
        Estimate stationary distribution via power iteration.
        
        Returns:
            Stationary distribution (normalized)
        """
        # Power iteration: pi_{n+1} = pi_n @ P
        pi = np.ones(self.n_states) / self.n_states
        
        for _ in range(1000):
            pi_new = pi @ self.P
            if np.allclose(pi, pi_new, atol=1e-8):
                break
            pi = pi_new
        
        return pi / pi.sum()  # Normalize
    
    def log_likelihood_sequence(self, regime_sequence: List[str]) -> float:
        """
        Compute log-likelihood of observed regime sequence.
        
        LL = log(pi_0) + sum_t log(P[regime_t, regime_{t+1}])
        
        Args:
            regime_sequence: List of regime strings
            
        Returns:
            Log-likelihood (higher = better fit)
        """
        if len(regime_sequence) < 2:
            logger.warning("Sequence too short for evaluation")
            return 0.0
        
        ll = 0.0
        
        # Initial state log-probability
        initial_idx = self.state_to_idx.get(regime_sequence[0])
        if initial_idx is not None and self.stationary_dist[initial_idx] > 0:
            ll += np.log(self.stationary_dist[initial_idx])
        else:
            ll += np.log(1e-10)  # Small value to avoid -inf
        
        # Transition log-probabilities
        for t in range(len(regime_sequence) - 1):
            from_state = regime_sequence[t]
            to_state = regime_sequence[t + 1]
            
            from_idx = self.state_to_idx.get(from_state)
            to_idx = self.state_to_idx.get(to_state)
            
            if from_idx is None or to_idx is None:
                logger.warning(f"Unknown state in sequence: {from_state} or {to_state}")
                continue
            
            trans_prob = self.P[from_idx, to_idx]
            
            if trans_prob > 0:
                ll += np.log(trans_prob)
            else:
                ll += np.log(1e-10)  # Small value for zero transitions
        
        return ll
    
    def per_capita_likelihood(self, regime_sequence: List[str]) -> float:
        """
        Compute average log-likelihood per observation.
        
        More comparable across sequences of different lengths.
        
        Args:
            regime_sequence: List of regime strings
            
        Returns:
            Average log-likelihood per observation
        """
        ll = self.log_likelihood_sequence(regime_sequence)
        
        if len(regime_sequence) > 0:
            return ll / len(regime_sequence)
        else:
            return 0.0
    
    def perplexity(self, regime_sequence: List[str]) -> float:
        """
        Compute perplexity = exp(-LL/n).
        
        Measures branching factor: how many equally likely sequences expected.
        Lower is better. Perplexity = n_states is random baseline.
        
        Args:
            regime_sequence: List of regime strings
            
        Returns:
            Perplexity (lower = better)
        """
        avg_ll = self.per_capita_likelihood(regime_sequence)
        
        if avg_ll < 0:
            return np.exp(-avg_ll)
        else:
            return np.exp(-avg_ll)
    
    def aic(self, regime_sequence: List[str], n_parameters: int = None) -> float:
        """
        Akaike Information Criterion.
        
        AIC = 2*k - 2*LL (k = number of parameters)
        Lower is better. Penalizes model complexity.
        
        Args:
            regime_sequence: List of regime strings
            n_parameters: Number of free parameters (estimated if None)
            
        Returns:
            AIC value
        """
        ll = self.log_likelihood_sequence(regime_sequence)
        
        # Estimate number of parameters if not provided
        if n_parameters is None:
            # For transition matrix: (n-1)*n parameters (per-row probabilities sum to 1)
            n_parameters = (self.n_states - 1) * self.n_states
        
        aic = 2 * n_parameters - 2 * ll
        
        return aic
    
    def bic(self, regime_sequence: List[str], n_parameters: int = None) -> float:
        """
        Bayesian Information Criterion.
        
        BIC = k*log(n) - 2*LL (k = parameters, n = observations)
        Lower is better. More penalizes complexity than AIC.
        
        Args:
            regime_sequence: List of regime strings
            n_parameters: Number of free parameters (estimated if None)
            
        Returns:
            BIC value
        """
        ll = self.log_likelihood_sequence(regime_sequence)
        n = len(regime_sequence)
        
        if n_parameters is None:
            n_parameters = (self.n_states - 1) * self.n_states
        
        bic = n_parameters * np.log(n) - 2 * ll
        
        return bic
    
    def likelihood_ratio_test(self, observed_seq: List[str], 
                             null_model_p: np.ndarray = None) -> Dict:
        """
        Likelihood ratio test against null model.
        
        LR = -2 * (LL_null - LL_alt)
        Under null: LR ~ Chi2(df)
        
        Args:
            observed_seq: Observed regime sequence
            null_model_p: Null model transition matrix (random if None)
            
        Returns:
            Dict with test statistic, p-value, significance
        """
        # Observed log-likelihood
        ll_alt = self.log_likelihood_sequence(observed_seq)
        
        # Null log-likelihood (random walk if not specified)
        if null_model_p is None:
            null_model_p = np.ones((self.n_states, self.n_states)) / self.n_states
        
        null_eval = LogLikelihoodEvaluator(null_model_p, self.state_names)
        ll_null = null_eval.log_likelihood_sequence(observed_seq)
        
        # Test statistic
        lr = -2 * (ll_null - ll_alt)
        
        # Degrees of freedom (parameters difference)
        df = (self.n_states - 1) * self.n_states  # Full model vs random
        
        # Approximation: Chi2 critical value at alpha=0.05
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr, df)
        
        return {
            'test_statistic': lr,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'df': df
        }
    
    def bootstrap_ci(self, regime_sequence: List[str], n_bootstrap: int = 100,
                    ci: float = 0.95) -> Dict:
        """
        Bootstrap confidence interval for log-likelihood.
        
        Args:
            regime_sequence: Observed sequence
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level (0.95 = 95%)
            
        Returns:
            Dict with point estimate and CI bounds
        """
        regime_sequence = np.array(regime_sequence)
        ll_values = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            idx = np.random.choice(len(regime_sequence), size=len(regime_sequence), replace=True)
            boot_seq = list(regime_sequence[idx])
            
            ll = self.log_likelihood_sequence(boot_seq)
            ll_values.append(ll)
        
        ll_values = np.array(ll_values)
        point_est = np.mean(ll_values)
        
        alpha = 1 - ci
        lower = np.percentile(ll_values, 100 * alpha / 2)
        upper = np.percentile(ll_values, 100 * (1 - alpha / 2))
        
        return {
            'point_estimate': point_est,
            'lower_ci': lower,
            'upper_ci': upper,
            'std_error': np.std(ll_values)
        }
    
    def print_evaluation_report(self, regime_sequence: List[str]) -> None:
        """Print comprehensive evaluation report."""
        print("\n" + "="*80)
        print("LOG-LIKELIHOOD EVALUATION REPORT")
        print("="*80)
        
        ll = self.log_likelihood_sequence(regime_sequence)
        ll_per_cap = self.per_capita_likelihood(regime_sequence)
        perplexity = self.perplexity(regime_sequence)
        aic = self.aic(regime_sequence)
        bic = self.bic(regime_sequence)
        
        print(f"\nSequence Length: {len(regime_sequence)}")
        print(f"\nLog-Likelihood: {ll:.4f}")
        print(f"Per-Capita LL:  {ll_per_cap:.4f}")
        print(f"Perplexity:     {perplexity:.4f} (baseline={self.n_states:.1f})")
        print(f"\nAIC:  {aic:.4f}")
        print(f"BIC:  {bic:.4f}")
        
        # Likelihood ratio test
        lr_result = self.likelihood_ratio_test(regime_sequence)
        print(f"\nLikelihood Ratio Test (vs Random):")
        print(f"  Test Statistic: {lr_result['test_statistic']:.4f}")
        print(f"  P-value:        {lr_result['p_value']:.4f}")
        print(f"  Significant:    {lr_result['significant']}")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example regime sequence
    states = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.2, 0.75]
    ])
    
    evaluator = LogLikelihoodEvaluator(P, states)
    
    # Create sample sequence
    regime_seq = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "MODERATE_RISK", 
                  "LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    
    evaluator.print_evaluation_report(regime_seq)
