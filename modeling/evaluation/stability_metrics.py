"""
Spectral and stability metrics for Markov chains.

Measures convergence speed, mixing properties, and asymptotic behavior.
- Eigenvalues: spectrum tells us mixing time
- Spectral gap: convergence to stationary distribution
- Mixing time: steps until P(X_t|X_0) ≈ stationary
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """Analyzes spectral properties of Markov chain."""
    
    def __init__(self, transition_matrix: np.ndarray, state_names: List[str]):
        """
        Initialize analyzer.
        
        Args:
            transition_matrix: Transition matrix P[i,j] = P(j|i)
            state_names: List of state names
        """
        self.P = np.asarray(transition_matrix)
        self.state_names = state_names
        self.n_states = len(state_names)
        
        # Compute eigenvalues and eigenvectors
        self._compute_spectral_decomposition()
        
        logger.info(f"Initialized spectral analyzer with {self.n_states} states")
    
    def _compute_spectral_decomposition(self) -> None:
        """Compute eigenvalues and eigenvectors of transition matrix."""
        try:
            # Eigenvalues in descending order (numpy returns ascending)
            evals, evecs = np.linalg.eig(self.P.T)  # Use transpose for row-stochastic
            
            # Sort by magnitude (descending)
            idx = np.argsort(np.abs(evals))[::-1]
            self.eigenvalues = evals[idx]
            self.eigenvectors = evecs[:, idx]
            
            logger.info(f"Eigenvalues: {self.eigenvalues}")
        except np.linalg.LinAlgError:
            logger.error("Cannot compute eigenvalues")
            self.eigenvalues = None
            self.eigenvectors = None
    
    @property
    def dominant_eigenvalue(self) -> float:
        """First eigenvalue (should be 1 for irreducible chains)."""
        if self.eigenvalues is not None and len(self.eigenvalues) > 0:
            return np.real(self.eigenvalues[0])
        return None
    
    @property
    def spectral_gap(self) -> float:
        """
        Spectral gap = 1 - |lambda_2| (second largest eigenvalue magnitude).
        
        Larger gap = faster convergence to stationary distribution.
        Range: [0, 1] (0 = no convergence, 1 = immediate convergence)
        """
        if self.eigenvalues is not None and len(self.eigenvalues) > 1:
            lambda_2 = np.abs(self.eigenvalues[1])
            gap = 1.0 - lambda_2
            return max(0, min(1, gap))  # Clamp to [0, 1]
        return None
    
    @property
    def mixing_time_approx(self) -> float:
        """
        Approximate mixing time using spectral gap.
        
        t_mix ≈ -1 / log(lambda_2)
        
        Number of steps until distance to stationary < 1/e
        """
        gap = self.spectral_gap
        
        if gap is not None and gap > 0:
            lambda_2 = 1.0 - gap
            if lambda_2 > 0:
                return -1.0 / np.log(lambda_2)
        
        return None
    
    def stationary_distribution(self) -> pd.Series:
        """
        Extract stationary distribution from dominant eigenvector.
        
        Returns:
            Series mapping state → stationary probability
        """
        if self.eigenvectors is None:
            logger.warning("Cannot extract eigenvectors")
            return None
        
        # First eigenvector (corresponding to lambda=1)
        v1 = np.real(self.eigenvectors[:, 0])
        
        # Normalize
        pi = np.abs(v1) / np.sum(np.abs(v1))
        
        return pd.Series(pi, index=self.state_names)
    
    def condition_number(self) -> float:
        """
        Condition number = |lambda_max| / |lambda_min|.
        
        Measures numerical stability and mixing speed.
        Close to 1 = well-conditioned, large = poorly-conditioned.
        """
        if self.eigenvalues is None:
            return None
        
        abs_evals = np.abs(self.eigenvalues)
        
        # Filter out near-zero eigenvalues
        abs_evals = abs_evals[abs_evals > 1e-10]
        
        if len(abs_evals) > 0:
            return np.max(abs_evals) / np.min(abs_evals)
        else:
            return None
    
    def is_aperiodic(self, tol: float = 1e-6) -> bool:
        """
        Check if chain is aperiodic (irreducible and no period > 1).
        
        Aperiodic if gcd of cycle lengths = 1.
        Condition: second eigenvalue is real and < 1.
        
        Args:
            tol: Tolerance for eigenvalue magnitude
            
        Returns:
            True if aperiodic
        """
        if self.eigenvalues is None or len(self.eigenvalues) < 2:
            return False
        
        # Check if second eigenvalue is real and strictly less than 1
        lambda_2_real = np.abs(np.imag(self.eigenvalues[1])) < tol
        lambda_2_small = np.abs(self.eigenvalues[1]) < 1 - tol
        
        return lambda_2_real and lambda_2_small
    
    def is_irreducible(self, tol: float = 1e-6) -> bool:
        """
        Check if chain is irreducible (all states communicate).
        
        Condition: only one eigenvalue = 1, all others < 1.
        
        Args:
            tol: Tolerance for eigenvalue comparison
            
        Returns:
            True if irreducible
        """
        if self.eigenvalues is None:
            return False
        
        # Count eigenvalues near 1
        near_one = np.sum(np.abs(self.eigenvalues - 1.0) < tol)
        
        # Should have exactly one
        return near_one == 1
    
    def print_spectral_report(self) -> None:
        """Print comprehensive spectral analysis report."""
        print("\n" + "="*80)
        print("SPECTRAL ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nTransition Matrix Shape: {self.P.shape}")
        print(f"States: {', '.join(self.state_names)}")
        
        print("\n" + "-"*80)
        print("EIGENVALUES")
        print("-"*80)
        if self.eigenvalues is not None:
            for i, ev in enumerate(self.eigenvalues[:min(5, len(self.eigenvalues))]):
                mag = np.abs(ev)
                angle = np.angle(ev)
                print(f"  λ_{i+1} = {mag:.6f} ∠ {np.degrees(angle):.2f}°")
        
        print("\n" + "-"*80)
        print("MIXING PROPERTIES")
        print("-"*80)
        print(f"Dominant Eigenvalue:    {self.dominant_eigenvalue:.6f}")
        print(f"Spectral Gap:           {self.spectral_gap:.6f}")
        print(f"Mixing Time (approx):   {self.mixing_time_approx:.2f} steps")
        print(f"Condition Number:       {self.condition_number():.2f}")
        print(f"Irreducible:            {self.is_irreducible()}")
        print(f"Aperiodic:              {self.is_aperiodic()}")
        
        print("\n" + "-"*80)
        print("STATIONARY DISTRIBUTION")
        print("-"*80)
        pi = self.stationary_distribution()
        if pi is not None:
            for state, prob in pi.items():
                print(f"  {state:<20}: {prob:.6f}")
        
        print("\n" + "="*80 + "\n")


class StabilityMetrics:
    """Compute stability and robustness metrics for transitions."""
    
    def __init__(self, transition_matrix: np.ndarray, state_names: List[str]):
        """
        Initialize stability metrics.
        
        Args:
            transition_matrix: Transition matrix
            state_names: State names
        """
        self.P = np.asarray(transition_matrix)
        self.state_names = state_names
        self.n_states = len(state_names)
    
    def persistence_matrix(self) -> pd.DataFrame:
        """
        Probability of staying in state for k periods.
        
        M[i,k] = (P[i,i])^k
        
        Returns:
            DataFrame with persistence probabilities
        """
        persistence = {}
        
        for i, state in enumerate(self.state_names):
            persistence[state] = []
            p_self = self.P[i, i]
            
            for k in range(1, 13):  # Up to 12 periods
                persistence[state].append(p_self ** k)
        
        df = pd.DataFrame(persistence, index=[f"{k}mo" for k in range(1, 13)])
        
        return df
    
    def expected_sojourn_time(self) -> pd.Series:
        """
        Expected number of steps in state before leaving.
        
        E[T_i] = 1 / (1 - P[i,i])
        
        Returns:
            Series of expected sojourn times
        """
        sojourn = {}
        
        for i, state in enumerate(self.state_names):
            p_self = self.P[i, i]
            
            if p_self < 1:
                sojourn[state] = 1.0 / (1.0 - p_self)
            else:
                sojourn[state] = np.inf  # Absorbing state
        
        return pd.Series(sojourn)
    
    def mean_first_passage_time(self, target_state: str) -> pd.Series:
        """
        Expected steps from each state to reach target state.
        
        Solves: t_i = 0 if i=target, 1 + sum_j P[i,j]*t_j otherwise
        
        Args:
            target_state: Target state name
            
        Returns:
            Series of mean first passage times
        """
        target_idx = self.state_names.index(target_state)
        
        # Modify transition matrix: absorb target state
        Q = self.P.copy()
        Q[target_idx, :] = 0
        Q[target_idx, target_idx] = 1
        
        # Solve (I - Q)t = 1
        I = np.eye(self.n_states)
        try:
            times = np.linalg.solve(I - Q, np.ones(self.n_states))
        except np.linalg.LinAlgError:
            logger.warning("Cannot solve first passage time")
            times = np.full(self.n_states, np.nan)
        
        return pd.Series(times, index=self.state_names)
    
    def transition_volatility(self) -> pd.DataFrame:
        """
        Measure uncertainty in transitions (entropy of each row).
        
        Entropy[i] = -sum_j P[i,j] * log(P[i,j])
        
        Returns:
            DataFrame with entropy metrics
        """
        entropy_values = []
        concentration = []  # 1 - entropy/log(n) [0=uniform, 1=deterministic]
        
        for i in range(self.n_states):
            row = self.P[i, :]
            
            # Entropy (ignore zero probabilities)
            row_nonzero = row[row > 0]
            entropy = -np.sum(row_nonzero * np.log(row_nonzero))
            entropy_values.append(entropy)
            
            # Concentration (1 = all weight on one state, 0 = uniform)
            max_conc = np.max(row)
            concentration.append(max_conc)
        
        df = pd.DataFrame({
            'Entropy': entropy_values,
            'Concentration': concentration,
            'Predictability': concentration  # Same as concentration for rows
        }, index=self.state_names)
        
        return df
    
    def print_stability_report(self) -> None:
        """Print stability metrics report."""
        print("\n" + "="*80)
        print("STABILITY METRICS REPORT")
        print("="*80)
        
        print("\n" + "-"*80)
        print("EXPECTED SOJOURN TIME (steps in state)")
        print("-"*80)
        sojourn = self.expected_sojourn_time()
        for state, time in sojourn.items():
            if np.isinf(time):
                print(f"  {state:<20}: ∞ (absorbing)")
            else:
                print(f"  {state:<20}: {time:.2f}")
        
        print("\n" + "-"*80)
        print("TRANSITION VOLATILITY (entropy)")
        print("-"*80)
        vol = self.transition_volatility()
        print(vol.round(4))
        
        print("\n" + "-"*80)
        print("PERSISTENCE (12-month horizon)")
        print("-"*80)
        persist = self.persistence_matrix()
        print(persist.iloc[11, :].round(4))  # 12-month
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    states = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.2, 0.75]
    ])
    
    # Spectral analysis
    spectral = SpectralAnalyzer(P, states)
    spectral.print_spectral_report()
    
    # Stability metrics
    stability = StabilityMetrics(P, states)
    stability.print_stability_report()
