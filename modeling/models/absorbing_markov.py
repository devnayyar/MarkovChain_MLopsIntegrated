"""
Absorbing Markov Chain Analysis.

Extends base Markov chain with absorbing state analysis:
- Absorbing state detection
- Fundamental matrix computation
- Expected time to absorption
- Absorption probability
- Recovery probability
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from scipy import linalg

logger = logging.getLogger(__name__)


class AbsorbingMarkovChain:
    """
    Markov chain with absorbing state analysis.
    
    An absorbing state is one from which the chain cannot leave.
    Examples: default (in credit), crisis (in financial stress), recession end.
    """
    
    def __init__(self, transition_matrix: np.ndarray, state_names: List[str],
                 absorbing_states: List[str] = None):
        """
        Initialize absorbing Markov chain.
        
        Args:
            transition_matrix: Transition probability matrix T[i,j] = P(j|i)
            state_names: List of state labels
            absorbing_states: List of absorbing state labels (detected if None)
        """
        self.transition_matrix = transition_matrix
        self.state_names = state_names
        self.n_states = len(state_names)
        
        # Detect absorbing states (if not provided)
        if absorbing_states is None:
            self.absorbing_states = self._detect_absorbing_states()
        else:
            self.absorbing_states = absorbing_states
        
        self.transient_states = [s for s in state_names if s not in self.absorbing_states]
        self.n_absorbing = len(self.absorbing_states)
        self.n_transient = len(self.transient_states)
        
        # Canonical form (transient states first, then absorbing)
        self._build_canonical_form()
        
        logger.info(f"Absorbing states: {self.absorbing_states}")
        logger.info(f"Transient states: {self.transient_states}")
    
    def _detect_absorbing_states(self) -> List[str]:
        """
        Detect absorbing states (rows with 1 on diagonal, 0s elsewhere).
        
        Returns:
            List of absorbing state names
        """
        absorbing = []
        
        for i, state in enumerate(self.state_names):
            # Check if T[i,i] = 1 (stays in same state with probability 1)
            if np.isclose(self.transition_matrix[i, i], 1.0):
                absorbing.append(state)
        
        return absorbing
    
    def _build_canonical_form(self) -> None:
        """
        Reorder matrix to canonical form:
        [Q | R]
        [0 | I]
        
        Where:
        - Q: transient to transient
        - R: transient to absorbing
        - 0: absorbing to transient (zeros)
        - I: absorbing to absorbing (identity)
        """
        # Map state names to indices in original matrix
        transient_idx = [self.state_names.index(s) for s in self.transient_states]
        absorbing_idx = [self.state_names.index(s) for s in self.absorbing_states]
        
        # Reorder transition matrix
        idx = transient_idx + absorbing_idx
        canonical = self.transition_matrix[np.ix_(idx, idx)]
        
        # Extract blocks
        self.Q = canonical[:self.n_transient, :self.n_transient]  # Transient → transient
        self.R = canonical[:self.n_transient, self.n_transient:]  # Transient → absorbing
        
        logger.debug(f"Q matrix (transient → transient):\n{self.Q}")
        logger.debug(f"R matrix (transient → absorbing):\n{self.R}")
    
    def fundamental_matrix(self) -> np.ndarray:
        """
        Compute fundamental matrix N = (I - Q)^{-1}.
        
        N[i,j] = expected number of times chain visits state j
             starting from transient state i (before absorption).
        
        Returns:
            Fundamental matrix
        """
        I = np.eye(self.n_transient)
        try:
            N = np.linalg.inv(I - self.Q)
            logger.info("Fundamental matrix computed successfully")
            return N
        except np.linalg.LinAlgError:
            logger.error("Fundamental matrix inversion failed (singular matrix)")
            return None
    
    def absorption_probability_matrix(self) -> pd.DataFrame:
        """
        Compute absorption probability matrix B = N @ R.
        
        B[i,j] = probability that starting from transient state i,
             the chain eventually gets absorbed in state j.
        
        Returns:
            DataFrame with absorption probabilities
        """
        N = self.fundamental_matrix()
        if N is None:
            return None
        
        B = N @ self.R
        
        # Create DataFrame
        df = pd.DataFrame(
            B,
            index=self.transient_states,
            columns=self.absorbing_states
        )
        
        logger.info("Absorption probability matrix:")
        logger.info(f"\n{df.round(4)}")
        
        return df
    
    def expected_time_to_absorption(self) -> pd.Series:
        """
        Compute expected time to absorption from each transient state.
        
        t[i] = (N @ 1) = expected number of steps before absorption
               starting from state i.
        
        Returns:
            Series mapping state → expected absorption time
        """
        N = self.fundamental_matrix()
        if N is None:
            return None
        
        ones = np.ones(self.n_transient)
        times = N @ ones
        
        return pd.Series(times, index=self.transient_states)
    
    def expected_visits(self) -> pd.DataFrame:
        """
        Expected number of visits to each state before absorption.
        
        Returns:
            DataFrame: rows=starting transient state, cols=visit target state
        """
        N = self.fundamental_matrix()
        if N is None:
            return None
        
        return pd.DataFrame(
            N,
            index=self.transient_states,
            columns=self.transient_states
        )
    
    def risk_escalation_probability(self, current_state: str, high_risk_states: List[str] = None) -> float:
        """
        Probability of reaching a high-risk state before absorption.
        
        Args:
            current_state: Starting state
            high_risk_states: List of states considered "high risk"
            
        Returns:
            Probability of visiting high-risk state before absorption
        """
        if high_risk_states is None:
            high_risk_states = [s for s in self.transient_states if 'HIGH' in s or 'CRISIS' in s]
        
        if current_state not in self.transient_states:
            return 0.0
        
        N = self.fundamental_matrix()
        if N is None:
            return None
        
        current_idx = self.transient_states.index(current_state)
        
        # Probability = sum of expected visits to high-risk states
        prob = 0.0
        for high_state in high_risk_states:
            if high_state in self.transient_states:
                high_idx = self.transient_states.index(high_state)
                prob += N[current_idx, high_idx]
        
        # Normalize by total expected lifetime
        total_visits = N[current_idx, :].sum()
        if total_visits > 0:
            prob = prob / total_visits
        
        return min(prob, 1.0)  # Cap at 1.0
    
    def recovery_probability(self, current_state: str, recovery_state: str) -> float:
        """
        Probability of recovery (reaching recovery_state before absorption).
        
        Args:
            current_state: Starting state
            recovery_state: Target recovery state
            
        Returns:
            Recovery probability
        """
        if current_state not in self.transient_states:
            return 0.0
        
        if recovery_state not in self.transient_states:
            return 0.0  # Can't recover to non-transient state
        
        N = self.fundamental_matrix()
        if N is None:
            return None
        
        current_idx = self.transient_states.index(current_state)
        recovery_idx = self.transient_states.index(recovery_state)
        
        # Probability of visiting recovery state >= 1 time
        prob = min(N[current_idx, recovery_idx], 1.0)
        
        return prob
    
    def print_analysis_report(self) -> None:
        """Print comprehensive absorption analysis."""
        print("\n" + "="*80)
        print("ABSORBING MARKOV CHAIN ANALYSIS")
        print("="*80)
        
        print(f"\nAbsorbing States: {self.absorbing_states}")
        print(f"Transient States: {self.transient_states}")
        
        print("\n" + "-"*80)
        print("FUNDAMENTAL MATRIX (N)")
        print("-"*80)
        N = self.fundamental_matrix()
        if N is not None:
            print(pd.DataFrame(N, index=self.transient_states, columns=self.transient_states).round(4))
        
        print("\n" + "-"*80)
        print("ABSORPTION PROBABILITY MATRIX (B)")
        print("-"*80)
        B = self.absorption_probability_matrix()
        if B is not None:
            print(B.round(4))
        
        print("\n" + "-"*80)
        print("EXPECTED TIME TO ABSORPTION")
        print("-"*80)
        times = self.expected_time_to_absorption()
        if times is not None:
            for state, time in times.items():
                print(f"  {state}: {time:.2f} steps")
        
        print("\n" + "-"*80)
        print("EXPECTED VISITS (before absorption)")
        print("-"*80)
        visits = self.expected_visits()
        if visits is not None:
            print(visits.round(4))
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example: Simple absorbing chain
    # States: LOW, MODERATE, HIGH (HIGH is absorbing)
    T = np.array([
        [0.5, 0.3, 0.2],  # From LOW
        [0.2, 0.5, 0.3],  # From MODERATE
        [0.0, 0.0, 1.0],  # From HIGH (absorbing)
    ])
    
    states = ['LOW', 'MODERATE', 'HIGH']
    
    amc = AbsorbingMarkovChain(T, states, absorbing_states=['HIGH'])
    amc.print_analysis_report()
    
    print("Risk escalation from LOW:", amc.risk_escalation_probability('LOW'))
    print("Recovery to MODERATE from HIGH: (not possible - absorbing)")
