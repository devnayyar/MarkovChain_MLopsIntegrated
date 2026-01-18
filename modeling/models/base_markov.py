"""
Base Markov Chain Model.

Core components:
- Transition matrix estimation
- State probability calculations
- Eigenvalue/eigenvector analysis (spectral properties)
- Model persistence and loading
- First-order Markov chains
- Transition matrix estimation
- Absorbing state detection
- Stationary distributions
- Expected absorption times
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MarkovChain:
    """First-order Markov Chain for economic regime modeling."""
    
    def __init__(self, state_sequence: np.ndarray, states: List[str] = None):
        """
        Initialize Markov chain from state sequence.
        
        Args:
            state_sequence: 1D array of states (integers or strings)
            states: List of state names (optional, auto-inferred if None)
        """
        self.state_sequence = np.asarray(state_sequence)
        
        # Convert strings to integers if needed
        if self.state_sequence.dtype == 'object':
            if states is None:
                unique_states = sorted(set(self.state_sequence))
            else:
                unique_states = states
            
            self.state_to_int = {s: i for i, s in enumerate(unique_states)}
            self.int_to_state = {i: s for s, i in self.state_to_int.items()}
            self.state_sequence = np.array([self.state_to_int[s] for s in self.state_sequence])
        else:
            if states is None:
                n_states = int(self.state_sequence.max()) + 1
                self.states = [f"State_{i}" for i in range(n_states)]
            else:
                self.states = states
            
            self.int_to_state = {i: self.states[i] for i in range(len(self.states))}
            self.state_to_int = {s: i for i, s in self.int_to_state.items()}
        
        self.n_states = len(self.int_to_state)
        self.states = [self.int_to_state[i] for i in range(self.n_states)]
        
        logger.info(f"Initialized Markov chain with {self.n_states} states: {self.states}")
        logger.info(f"Sequence length: {len(self.state_sequence)} transitions")
    
    def estimate_transition_matrix(self) -> np.ndarray:
        """
        Estimate transition matrix P(state_t+1 | state_t).
        
        Returns:
            Transition matrix (n_states × n_states)
            P[i,j] = probability of moving from state i to state j
        """
        # Count transitions
        transition_counts = np.zeros((self.n_states, self.n_states))
        
        for t in range(len(self.state_sequence) - 1):
            current_state = self.state_sequence[t]
            next_state = self.state_sequence[t + 1]
            transition_counts[current_state, next_state] += 1
        
        # Convert to probabilities (row normalization)
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        
        self.transition_matrix = transition_counts / row_sums
        
        logger.info("Transition matrix estimated")
        logger.info(f"\nTransition Matrix:\n{self._format_matrix()}")
        
        return self.transition_matrix
    
    def _format_matrix(self) -> str:
        """Format transition matrix for display."""
        df = pd.DataFrame(
            self.transition_matrix,
            index=[f"{s}" for s in self.states],
            columns=[f"{s}" for s in self.states]
        )
        return df.round(3).to_string()
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary (long-run) distribution of states.
        
        Solves: π = π @ P, subject to sum(π) = 1
        
        Returns:
            Stationary distribution (n_states,)
        """
        # Use eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvalue closest to 1 (should be exactly 1 for stochastic matrices)
        idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
        stationary = np.real(eigenvectors[:, idx])
        stationary = np.abs(stationary) / np.sum(np.abs(stationary))
        
        self.stationary_distribution = stationary
        
        logger.info("Stationary distribution computed")
        logger.info(f"\nLong-run probabilities:")
        for state, prob in zip(self.states, stationary):
            logger.info(f"  {state}: {prob:.4f}")
        
        return stationary
    
    def detect_absorbing_states(self, threshold: float = 0.99) -> List[str]:
        """
        Detect absorbing states (state i where P[i,i] ≈ 1).
        
        Args:
            threshold: Minimum self-loop probability to be "absorbing-like"
            
        Returns:
            List of absorbing state names
        """
        absorbing = []
        
        for i in range(self.n_states):
            if self.transition_matrix[i, i] >= threshold:
                absorbing.append(self.states[i])
                logger.warning(f"Absorbing-like state detected: {self.states[i]} (P={self.transition_matrix[i, i]:.4f})")
        
        self.absorbing_states = absorbing
        return absorbing
    
    def expected_time_to_absorption(self, state: str) -> Optional[float]:
        """
        Compute expected time to absorption from a given state.
        
        For transient states in absorbing Markov chains,
        computes E[T | starting at state].
        
        Args:
            state: Starting state name
            
        Returns:
            Expected time to absorption (or None if no absorbing states)
        """
        if not self.absorbing_states:
            logger.info(f"No absorbing states found - expected absorption time undefined")
            return None
        
        # Identify transient and absorbing states
        absorbing_idx = [self.state_to_int[s] for s in self.absorbing_states]
        transient_idx = [i for i in range(self.n_states) if i not in absorbing_idx]
        
        if not transient_idx:
            return 0.0
        
        # Extract transient submatrix
        Q = self.transition_matrix[np.ix_(transient_idx, transient_idx)]
        
        # Fundamental matrix: N = (I - Q)^-1
        try:
            N = np.linalg.inv(np.eye(len(transient_idx)) - Q)
        except np.linalg.LinAlgError:
            logger.warning("Cannot invert (I - Q), singular matrix")
            return None
        
        # Expected time = sum of row in N
        if state in self.states and self.state_to_int[state] in transient_idx:
            state_idx = transient_idx.index(self.state_to_int[state])
            expected_time = np.sum(N[state_idx, :])
            logger.info(f"Expected time to absorption from {state}: {expected_time:.2f} periods")
            return expected_time
        else:
            return None
    
    def log_likelihood(self) -> float:
        """
        Compute log-likelihood of observed transitions.
        
        Higher = better fit. Measures how likely the observed
        sequence is under the estimated model.
        
        Returns:
            Log-likelihood of sequence under model
        """
        ll = 0.0
        count = 0
        
        for t in range(len(self.state_sequence) - 1):
            current_state = self.state_sequence[t]
            next_state = self.state_sequence[t + 1]
            
            prob = self.transition_matrix[current_state, next_state]
            
            if prob > 0:
                ll += np.log(prob)
                count += 1
            else:
                ll += np.log(1e-10)  # Avoid log(0)
                logger.warning(f"Zero probability transition: {self.states[current_state]} → {self.states[next_state]}")
        
        self.log_likelihood_value = ll / count if count > 0 else 0
        logger.info(f"Log-likelihood per transition: {self.log_likelihood_value:.4f}")
        
        return self.log_likelihood_value
    
    def predict_next_state(self, current_state: str) -> Dict[str, float]:
        """
        Predict probability distribution of next state.
        
        Args:
            current_state: Current regime state
            
        Returns:
            Dict mapping state names to probabilities
        """
        if current_state not in self.state_to_int:
            raise ValueError(f"Unknown state: {current_state}")
        
        state_idx = self.state_to_int[current_state]
        next_probs = self.transition_matrix[state_idx, :]
        
        return {self.states[i]: next_probs[i] for i in range(self.n_states)}
    
    def forecast_path(self, current_state: str, steps: int = 12) -> List[Dict]:
        """
        Multi-step forecast from current state.
        
        Args:
            current_state: Starting state
            steps: Number of periods to forecast
            
        Returns:
            List of dicts with probabilities at each step
        """
        state_idx = self.state_to_int[current_state]
        state_prob = np.zeros(self.n_states)
        state_prob[state_idx] = 1.0
        
        forecast = []
        
        for step in range(steps):
            state_prob = state_prob @ self.transition_matrix
            forecast.append({
                "step": step + 1,
                "state_probs": {self.states[i]: state_prob[i] for i in range(self.n_states)}
            })
        
        logger.info(f"Forecast path from {current_state} ({steps} steps)")
        
        return forecast
    
    def summary(self) -> Dict:
        """
        Get comprehensive summary of Markov chain.
        
        Returns:
            Dict with key metrics
        """
        return {
            "n_states": self.n_states,
            "states": self.states,
            "n_transitions": len(self.state_sequence) - 1,
            "n_observations": len(self.state_sequence),
            "absorbing_states": self.absorbing_states if hasattr(self, 'absorbing_states') else [],
            "log_likelihood": self.log_likelihood_value if hasattr(self, 'log_likelihood_value') else None,
            "stationary_dist": {
                self.states[i]: self.stationary_distribution[i]
                for i in range(self.n_states)
            } if hasattr(self, 'stationary_distribution') else None,
        }
    
    def print_summary(self):
        """Print comprehensive summary."""
        summary = self.summary()
        
        print("\n" + "="*80)
        print("MARKOV CHAIN SUMMARY")
        print("="*80)
        print(f"\nStates ({summary['n_states']}): {', '.join(summary['states'])}")
        print(f"Observations: {summary['n_observations']}")
        print(f"Transitions: {summary['n_transitions']}")
        
        if summary['absorbing_states']:
            print(f"\nAbsorbing States: {summary['absorbing_states']}")
        
        if summary['log_likelihood']:
            print(f"Log-Likelihood: {summary['log_likelihood']:.4f}")
        
        if summary['stationary_dist']:
            print(f"\nStationary Distribution:")
            for state, prob in summary['stationary_dist'].items():
                print(f"  {state}: {prob:.4f}")
        
        print("\n" + "="*80 + "\n")


class RollingWindowMarkovChain:
    """Non-stationary Markov chains using rolling windows."""
    
    def __init__(self, state_sequence: np.ndarray, window_size: int = 60):
        """
        Initialize rolling window Markov chain.
        
        Args:
            state_sequence: 1D array of states
            window_size: Window size in periods
        """
        self.state_sequence = np.asarray(state_sequence)
        self.window_size = window_size
        self.chains = []
        
        logger.info(f"Initialized rolling window Markov with window_size={window_size}")
    
    def fit(self, states: List[str] = None):
        """
        Fit rolling window chains.
        
        Args:
            states: List of state names
        """
        n_windows = len(self.state_sequence) - self.window_size + 1
        
        for i in range(n_windows):
            window = self.state_sequence[i:i + self.window_size]
            chain = MarkovChain(window, states)
            chain.estimate_transition_matrix()
            self.chains.append(chain)
        
        logger.info(f"Fit {len(self.chains)} rolling window chains")
    
    def get_transition_drift(self) -> np.ndarray:
        """
        Compute Frobenius norm of transition matrix differences.
        
        Measures how much the transition dynamics are changing.
        
        Returns:
            Array of drift values over time
        """
        drift = []
        
        for i in range(1, len(self.chains)):
            P_old = self.chains[i-1].transition_matrix
            P_new = self.chains[i].transition_matrix
            
            diff_norm = np.linalg.norm(P_new - P_old, 'fro')
            drift.append(diff_norm)
        
        logger.info(f"Transition drift range: [{min(drift):.4f}, {max(drift):.4f}]")
        
        return np.array(drift)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Simple 3-state Markov chain
    states = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    
    # Simulated state sequence (for testing)
    np.random.seed(42)
    state_seq = np.random.choice([0, 1, 2], size=100)
    
    mc = MarkovChain(state_seq, states)
    mc.estimate_transition_matrix()
    mc.get_stationary_distribution()
    mc.detect_absorbing_states()
    mc.log_likelihood()
    
    print("\n" + "="*80)
    print("PREDICTIONS FROM CURRENT STATE")
    print("="*80)
    
    for state in states:
        probs = mc.predict_next_state(state)
        print(f"\nFrom {state}:")
        for next_state, prob in probs.items():
            print(f"  → {next_state}: {prob:.4f}")
    
    mc.print_summary()
