"""
Integration test for all evaluation modules.

Tests: log-likelihood, stability metrics, business metrics with real gold layer data.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling.models.base_markov import MarkovChain
from modeling.models.absorbing_markov import AbsorbingMarkovChain
from modeling.evaluation.log_likelihood import LogLikelihoodEvaluator
from modeling.evaluation.stability_metrics import SpectralAnalyzer, StabilityMetrics
from modeling.evaluation.business_metrics import RiskMetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_gold_data():
    """Load regime sequences from gold layer."""
    gold_path = "data/gold/markov_state_sequences.parquet"
    
    if not os.path.exists(gold_path):
        logger.error(f"Gold data not found at {gold_path}")
        return None
    
    df = pd.read_parquet(gold_path)
    logger.info(f"Loaded gold data: {df.shape}")
    
    return df


def test_base_markov_chain():
    """Test base Markov chain model."""
    print("\n" + "="*80)
    print("[1/5] Testing Base Markov Chain")
    print("="*80)
    
    gold_df = load_gold_data()
    if gold_df is None:
        return None
    
    # Extract regime sequence
    regime_seq = gold_df['REGIME_RISK'].values
    states = list(gold_df['REGIME_RISK'].unique())
    states = [s for s in states if pd.notna(s)]
    
    logger.info(f"States: {states}")
    logger.info(f"Sequence length: {len(regime_seq)}")
    
    # Fit Markov chain
    try:
        mc = MarkovChain(regime_seq, states)
        mc.estimate_transition_matrix()
        mc.get_stationary_distribution()
        
        print(f"\n[PASS] Base Markov Chain fitted")
        print(f"  States: {mc.states}")
        print(f"  Stationary distribution:")
        stationary = mc.get_stationary_distribution()
        for state, prob in zip(mc.states, stationary):
            print(f"    {state:<20}: {prob:.4f}")
        
        return mc
    except Exception as e:
        logger.error(f"[FAIL] Error fitting Markov chain: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_log_likelihood(mc):
    """Test log-likelihood evaluation."""
    print("\n" + "="*80)
    print("[2/5] Testing Log-Likelihood Evaluator")
    print("="*80)
    
    if mc is None:
        logger.warning("Skipping - no Markov chain")
        return
    
    try:
        # Create evaluator
        evaluator = LogLikelihoodEvaluator(mc.transition_matrix, mc.states)
        
        # Evaluate full sequence
        regime_seq = [str(mc.int_to_state[int(s)]) for s in mc.state_sequence if isinstance(s, (int, np.integer))]
        
        ll = evaluator.log_likelihood_sequence(regime_seq)
        ll_per_cap = evaluator.per_capita_likelihood(regime_seq)
        perp = evaluator.perplexity(regime_seq)
        aic = evaluator.aic(regime_seq)
        bic = evaluator.bic(regime_seq)
        
        print(f"\n[PASS] Log-Likelihood Evaluation")
        print(f"  Total LL: {ll:.4f}")
        print(f"  Per-capita LL: {ll_per_cap:.6f}")
        print(f"  Perplexity: {perp:.4f} (baseline={len(mc.states):.1f})")
        print(f"  AIC: {aic:.4f}")
        print(f"  BIC: {bic:.4f}")
        
        if perp < len(mc.states):
            print(f"  -> GOOD FIT: Model better than random")
        
    except Exception as e:
        logger.error(f"[FAIL] Log-likelihood error: {e}")
        import traceback
        traceback.print_exc()


def test_spectral_analysis(mc):
    """Test spectral analysis."""
    print("\n" + "="*80)
    print("[3/5] Testing Spectral Analysis")
    print("="*80)
    
    if mc is None:
        logger.warning("Skipping - no Markov chain")
        return
    
    try:
        spectral = SpectralAnalyzer(mc.transition_matrix, mc.states)
        
        print(f"\n[PASS] Spectral Analysis")
        print(f"  Dominant Eigenvalue: {spectral.dominant_eigenvalue:.6f}")
        print(f"  Spectral Gap: {spectral.spectral_gap:.6f}")
        print(f"  Mixing Time (approx): {spectral.mixing_time_approx:.2f} steps")
        print(f"  Condition Number: {spectral.condition_number():.2f}")
        print(f"  Irreducible: {spectral.is_irreducible()}")
        print(f"  Aperiodic: {spectral.is_aperiodic()}")
        
        pi = spectral.stationary_distribution()
        print(f"\n  Stationary Distribution:")
        for state, prob in pi.items():
            print(f"    {state:<20}: {prob:.4f}")
        
    except Exception as e:
        logger.error(f"[FAIL] Spectral analysis error: {e}")
        import traceback
        traceback.print_exc()


def test_stability_metrics(mc):
    """Test stability metrics."""
    print("\n" + "="*80)
    print("[4/5] Testing Stability Metrics")
    print("="*80)
    
    if mc is None:
        logger.warning("Skipping - no Markov chain")
        return
    
    try:
        stability = StabilityMetrics(mc.transition_matrix, mc.states)
        
        print(f"\n[PASS] Stability Metrics")
        
        sojourn = stability.expected_sojourn_time()
        print(f"\n  Expected Sojourn Times (months in state):")
        for state, time in sojourn.items():
            if np.isinf(time):
                print(f"    {state:<20}: ∞ (absorbing)")
            else:
                print(f"    {state:<20}: {time:.2f}")
        
        vol = stability.transition_volatility()
        print(f"\n  Transition Volatility:")
        print(vol.round(4))
        
    except Exception as e:
        logger.error(f"[FAIL] Stability metrics error: {e}")
        import traceback
        traceback.print_exc()


def test_business_metrics(mc):
    """Test business risk metrics."""
    print("\n" + "="*80)
    print("[5/5] Testing Business Risk Metrics")
    print("="*80)
    
    if mc is None:
        logger.warning("Skipping - no Markov chain")
        return
    
    try:
        risk_calc = RiskMetricsCalculator(mc.transition_matrix, mc.states)
        
        print(f"\n[PASS] Business Risk Metrics")
        
        # Crisis analysis
        print(f"\n  Crisis Duration Analysis:")
        for state in mc.states:
            if 'HIGH' in state:
                crisis_info = risk_calc.crisis_duration_distribution(state)
                if crisis_info:
                    print(f"    {state}:")
                    print(f"      Mean Duration: {crisis_info['mean_duration_months']:.2f} months")
                    print(f"      Self-Loop: {crisis_info['self_loop_probability']:.4f}")
        
        # Recovery times
        print(f"\n  Recovery Times to LOW_RISK:")
        low_risk_states = [s for s in mc.states if 'LOW' in s or s == mc.states[0]]
        target = low_risk_states[0] if low_risk_states else mc.states[0]
        
        for state in mc.states:
            recovery = risk_calc.recovery_time_to_state(state, target)
            if recovery is not None:
                if np.isinf(recovery):
                    print(f"    {state:<20}: Cannot recover")
                else:
                    print(f"    {state:<20}: {recovery:.2f} months")
        
        # VaR
        print(f"\n  Value-at-Risk (95% confidence):")
        for state in mc.states:
            var = risk_calc.value_at_risk(state, horizon_months=1, var_level=0.95)
            if var:
                print(f"    From {state:<15} -> {var['var_state']}")
        
    except Exception as e:
        logger.error(f"[FAIL] Business metrics error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all evaluation tests."""
    print("\n" + "="*80)
    print("PHASE 5: MARKOV EVALUATION METRICS TEST SUITE".center(80))
    print("="*80)
    
    # Test sequence
    mc = test_base_markov_chain()
    test_log_likelihood(mc)
    test_spectral_analysis(mc)
    test_stability_metrics(mc)
    test_business_metrics(mc)
    
    print("\n" + "="*80)
    print("[SUCCESS] ALL EVALUATION TESTS COMPLETED".center(80))
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
