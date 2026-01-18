"""
Business-oriented risk metrics derived from Markov chain.

Translates mathematical properties to business KPIs:
- Risk escalation probability
- Average time in crisis
- Recovery time distributions
- Value-at-Risk metrics
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class RiskMetricsCalculator:
    """Compute business-oriented risk metrics."""
    
    def __init__(self, transition_matrix: np.ndarray, state_names: List[str],
                 risk_levels: Dict[str, int] = None):
        """
        Initialize calculator.
        
        Args:
            transition_matrix: Transition matrix P[i,j] = P(j|i)
            state_names: State names
            risk_levels: Dict mapping state name to risk level (0=low, 1=med, 2=high)
        """
        self.P = np.asarray(transition_matrix)
        self.state_names = state_names
        self.state_to_idx = {s: i for i, s in enumerate(state_names)}
        self.n_states = len(state_names)
        
        # Define risk levels (override with provided)
        if risk_levels is None:
            self.risk_levels = self._infer_risk_levels()
        else:
            self.risk_levels = risk_levels
        
        logger.info(f"Risk levels: {self.risk_levels}")
    
    def _infer_risk_levels(self) -> Dict[str, int]:
        """Infer risk levels from state names."""
        risk_levels = {}
        
        for state in self.state_names:
            if 'HIGH' in state or 'CRISIS' in state:
                risk_levels[state] = 2
            elif 'MODERATE' in state or 'MED' in state:
                risk_levels[state] = 1
            elif 'LOW' in state or 'CALM' in state:
                risk_levels[state] = 0
            else:
                risk_levels[state] = 1  # Default to medium
        
        return risk_levels
    
    def escalation_probability_matrix(self) -> pd.DataFrame:
        """
        Probability of escalating to each risk level within k steps.
        
        Computed via: (I + P + P^2 + ... + P^{k-1})
        
        Returns:
            DataFrame: rows=start states, cols=target risk levels, values=probabilities
        """
        escalation = {}
        
        for start_state in self.state_names:
            start_risk = self.risk_levels[start_state]
            
            # Probability of reaching each risk level
            escalation[start_state] = {}
            
            # Compute 12-month horizon
            P_power = np.eye(self.n_states)
            cumulative_prob = np.zeros(self.n_states)
            
            for month in range(12):
                P_power = P_power @ self.P
                cumulative_prob += P_power
            
            start_idx = self.state_to_idx[start_state]
            
            # Aggregate by risk level
            for risk_level in [0, 1, 2]:
                target_states = [s for s, r in self.risk_levels.items() if r == risk_level]
                target_idx = [self.state_to_idx[s] for s in target_states]
                
                prob = np.sum(cumulative_prob[start_idx, target_idx])
                escalation[start_state][f"Level_{risk_level}"] = prob
        
        return pd.DataFrame(escalation).T
    
    def crisis_duration_distribution(self, crisis_state: str = 'HIGH_RISK') -> Dict:
        """
        Distribution of duration in crisis state.
        
        P(duration = k) = (P[crisis,crisis])^{k-1} * (1 - P[crisis,crisis])
        
        Args:
            crisis_state: State considered "crisis"
            
        Returns:
            Dict with duration statistics
        """
        if crisis_state not in self.state_to_idx:
            logger.warning(f"Unknown state: {crisis_state}")
            return None
        
        idx = self.state_to_idx[crisis_state]
        p_stay = self.P[idx, idx]
        
        # Geometric distribution with p_stay
        # E[duration] = 1 / (1 - p_stay)
        if p_stay >= 1:
            expected = np.inf
        else:
            expected = 1.0 / (1.0 - p_stay)
        
        # Standard deviation for geometric dist
        if p_stay < 1:
            std = np.sqrt(p_stay) / (1 - p_stay)
        else:
            std = np.inf
        
        # Quantiles
        durations = {}
        for quantile in [0.25, 0.50, 0.75, 0.95]:
            if 0 < p_stay < 1:
                # Inverse CDF of geometric
                duration = np.ceil(np.log(1 - quantile) / np.log(p_stay))
                durations[f"{int(quantile*100)}th_percentile"] = int(duration)
        
        return {
            'state': crisis_state,
            'mean_duration_months': expected,
            'std_duration_months': std,
            'self_loop_probability': p_stay,
            'quantiles': durations
        }
    
    def recovery_time_to_state(self, from_state: str, target_state: str) -> Optional[float]:
        """
        Expected number of steps from state to target state.
        
        Uses first-passage time calculation.
        
        Args:
            from_state: Starting state
            target_state: Target state
            
        Returns:
            Expected time (steps)
        """
        if from_state not in self.state_to_idx or target_state not in self.state_to_idx:
            return None
        
        from_idx = self.state_to_idx[from_state]
        target_idx = self.state_to_idx[target_state]
        
        if from_idx == target_idx:
            return 0.0  # Already there
        
        # Absorb target state, solve first passage time
        Q = self.P.copy()
        Q[target_idx, :] = 0
        Q[target_idx, target_idx] = 1
        
        I = np.eye(self.n_states)
        try:
            times = np.linalg.solve(I - Q, np.ones(self.n_states))
            return times[from_idx]
        except np.linalg.LinAlgError:
            logger.warning("Cannot compute recovery time")
            return None
    
    def value_at_risk(self, current_state: str, horizon_months: int = 12,
                      var_level: float = 0.95) -> Dict:
        """
        Value-at-Risk: worst-case regime transition probability.
        
        VaR_α = min {x : P(Z > x) <= 1-α}
        
        Args:
            current_state: Current regime
            horizon_months: Time horizon
            var_level: Confidence level (0.95 = 95% VaR)
            
        Returns:
            Dict with VaR metrics
        """
        if current_state not in self.state_to_idx:
            return None
        
        from_idx = self.state_to_idx[current_state]
        current_risk = self.risk_levels[current_state]
        
        # Simulate 1-step ahead distribution
        distribution = self.P[from_idx, :]
        
        # Worst case: highest risk state reached with var_level probability
        # Rank states by risk level
        risk_ranking = [(r, s) for s, r in self.risk_levels.items()]
        risk_ranking.sort(reverse=True)
        
        cumulative_prob = 0.0
        var_state = current_state
        var_risk = current_risk
        
        for risk_level, state in risk_ranking:
            state_idx = self.state_to_idx[state]
            cumulative_prob += distribution[state_idx]
            
            if cumulative_prob >= (1 - var_level):
                var_state = state
                var_risk = risk_level
                break
        
        return {
            'current_state': current_state,
            'var_state': var_state,
            'var_risk_level': var_risk,
            'var_probability': cumulative_prob,
            'confidence_level': var_level
        }
    
    def stress_test(self, shock_scenario: Dict[str, float]) -> pd.DataFrame:
        """
        Stress test: modify transition matrix under scenario.
        
        Args:
            shock_scenario: Dict mapping state pairs to shock magnitude
                           E.g. {'LOW_RISK->HIGH_RISK': 0.2} adds 20% escalation
            
        Returns:
            Modified transition matrix
        """
        P_shock = self.P.copy()
        
        for transition, shock_amount in shock_scenario.items():
            parts = transition.split('->')
            if len(parts) != 2:
                continue
            
            from_state = parts[0].strip()
            to_state = parts[1].strip()
            
            if from_state not in self.state_to_idx or to_state not in self.state_to_idx:
                continue
            
            from_idx = self.state_to_idx[from_state]
            to_idx = self.state_to_idx[to_state]
            
            # Add shock to transition
            P_shock[from_idx, to_idx] += shock_amount
            # Reduce self-loop to maintain row sum = 1
            P_shock[from_idx, from_idx] = max(0, 1.0 - P_shock[from_idx, :].sum())
            
            # Normalize row
            P_shock[from_idx, :] = P_shock[from_idx, :] / P_shock[from_idx, :].sum()
        
        return pd.DataFrame(P_shock, index=self.state_names, columns=self.state_names)
    
    def print_risk_report(self) -> None:
        """Print comprehensive risk metrics report."""
        print("\n" + "="*80)
        print("BUSINESS RISK METRICS REPORT")
        print("="*80)
        
        print("\n" + "-"*80)
        print("CRISIS DURATION ANALYSIS")
        print("-"*80)
        
        crisis_states = [s for s, r in self.risk_levels.items() if r == 2]
        for crisis_state in crisis_states:
            crisis_info = self.crisis_duration_distribution(crisis_state)
            if crisis_info:
                print(f"\n{crisis_state}:")
                print(f"  Mean Duration: {crisis_info['mean_duration_months']:.1f} months")
                print(f"  Std Deviation: {crisis_info['std_duration_months']:.1f} months")
                print(f"  Self-Loop Prob: {crisis_info['self_loop_probability']:.4f}")
        
        print("\n" + "-"*80)
        print("RECOVERY TIMES (expected months to return to LOW_RISK)")
        print("-"*80)
        
        target_state = 'LOW_RISK' if 'LOW_RISK' in self.state_names else self.state_names[0]
        for from_state in self.state_names:
            recovery = self.recovery_time_to_state(from_state, target_state)
            if recovery is not None:
                if np.isinf(recovery):
                    print(f"  {from_state:<20}: Cannot recover (or absorbing)")
                else:
                    print(f"  {from_state:<20}: {recovery:.1f} months")
        
        print("\n" + "-"*80)
        print("VALUE AT RISK (95% VaR, 1-month horizon)")
        print("-"*80)
        
        for state in self.state_names:
            var_result = self.value_at_risk(state, horizon_months=1, var_level=0.95)
            if var_result:
                print(f"  From {state:<15} → {var_result['var_state']:<15} (Prob={var_result['var_probability']:.4f})")
        
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    states = ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK"]
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.2, 0.75]
    ])
    
    risk_calc = RiskMetricsCalculator(P, states)
    risk_calc.print_risk_report()
    
    # Stress test example
    print("\nSTRESS TEST: 20% increased escalation LOW→HIGH")
    shock_scenario = {"LOW_RISK->HIGH_RISK": 0.2}
    P_shock = risk_calc.stress_test(shock_scenario)
    print(P_shock.round(4))
