"""Drift detection for Markov chain state distributions."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon


class DriftDetector:
    """Detect distribution drift in regime states over time."""
    
    def __init__(self, reference_window=12, current_window=3, threshold=0.1):
        """
        Args:
            reference_window: months to use as baseline (default 1 year)
            current_window: months to check for drift (default quarterly)
            threshold: KL divergence threshold for drift alert (0.1 = 10% divergence)
        """
        self.reference_window = reference_window
        self.current_window = current_window
        self.threshold = threshold
    
    def detect_regime_drift(self, gold_path="data/gold/markov_state_sequences.parquet"):
        """Detect drift in REGIME_RISK distribution."""
        if not Path(gold_path).exists():
            print("Gold data not found for drift detection.")
            return None
        
        df = pd.read_parquet(gold_path)
        df = df.sort_values("date")
        
        # Split into reference (older) and current (recent)
        split_idx = len(df) - self.current_window
        reference = df.iloc[:split_idx]
        current = df.iloc[split_idx:]
        
        # Get distributions
        ref_dist = reference['REGIME_RISK'].value_counts(normalize=True)
        curr_dist = current['REGIME_RISK'].value_counts(normalize=True)
        
        # Ensure same index
        all_regimes = set(ref_dist.index) | set(curr_dist.index)
        ref_dist = ref_dist.reindex(list(all_regimes), fill_value=0.0001)
        curr_dist = curr_dist.reindex(list(all_regimes), fill_value=0.0001)
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(ref_dist.values, curr_dist.values)
        
        print(f"\n📊 Regime Distribution Drift Detection")
        print(f"  Reference Period: {reference['date'].min()} to {reference['date'].max()}")
        print(f"  Current Period: {current['date'].min()} to {current['date'].max()}")
        print(f"  Reference Distribution:\n{ref_dist}")
        print(f"  Current Distribution:\n{curr_dist}")
        print(f"  Jensen-Shannon Divergence: {js_div:.4f}")
        print(f"  Threshold: {self.threshold:.4f}")
        
        if js_div > self.threshold:
            print(f"  ⚠️  DRIFT DETECTED! Divergence exceeds threshold.")
            return {"drift": True, "divergence": js_div, "alert": True}
        else:
            print(f"  ✅ No significant drift detected.")
            return {"drift": False, "divergence": js_div, "alert": False}
    
    def detect_transition_drift(self, gold_path="data/gold/markov_state_sequences.parquet"):
        """Detect drift in transition matrix probabilities."""
        if not Path(gold_path).exists():
            return None
        
        df = pd.read_parquet(gold_path)
        df = df.sort_values("date")
        
        # Split into reference and current
        split_idx = len(df) - self.current_window
        reference = df.iloc[:split_idx]
        current = df.iloc[split_idx:]
        
        # Get transition matrices
        ref_matrix = self._get_transition_matrix(reference['REGIME_RISK'].values)
        curr_matrix = self._get_transition_matrix(current['REGIME_RISK'].values)
        
        # Calculate Frobenius norm (matrix difference)
        diff = np.linalg.norm(ref_matrix - curr_matrix, 'fro')
        
        print(f"\n📊 Transition Matrix Drift Detection")
        print(f"  Reference Matrix Norm: {np.linalg.norm(ref_matrix, 'fro'):.4f}")
        print(f"  Current Matrix Norm: {np.linalg.norm(curr_matrix, 'fro'):.4f}")
        print(f"  Frobenius Norm Difference: {diff:.4f}")
        
        return {"transition_drift": True if diff > 0.1 else False, "norm_diff": diff}
    
    @staticmethod
    def _get_transition_matrix(sequences):
        """Build transition matrix from state sequence."""
        states = sorted(set(sequences))
        n_states = len(states)
        state_to_idx = {s: i for i, s in enumerate(states)}
        
        matrix = np.zeros((n_states, n_states))
        for i in range(len(sequences) - 1):
            from_idx = state_to_idx[sequences[i]]
            to_idx = state_to_idx[sequences[i+1]]
            matrix[from_idx, to_idx] += 1
        
        # Normalize
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        matrix = matrix / row_sums
        
        return matrix


if __name__ == "__main__":
    detector = DriftDetector()
    detector.detect_regime_drift()
    detector.detect_transition_drift()
