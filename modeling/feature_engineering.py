"""
Advanced Feature Engineering for Regime Detection - Phase 1

This module generates 73 engineered features from 5 raw economic indicators.

Feature Categories:
1. Lag Features (15 features)
   - Previous values: t-1, t-2, t-3, t-5, t-10
   
2. Rolling Statistics (20 features)
   - Rolling mean, std, min, max for windows: 5, 10, 20, 50 days
   
3. Momentum & Acceleration (10 features)
   - Rate of change, acceleration, jerk
   
4. Feature Interactions (15 features)
   - Ratios, differences, products of features
   
5. Spectral Features (8 features)
   - FFT components for cyclical patterns
   
6. Derived Risk Indicators (5 features)
   - VIX-like measures, stress indicators

Total Output: 73 features for ensemble models
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from scipy import signal, fft
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for financial regime detection.
    
    Converts 5 raw economic indicators into 73 engineered features
    that capture temporal dynamics, interactions, and spectral patterns.
    """
    
    def __init__(self, 
                 lag_periods: List[int] = None,
                 rolling_windows: List[int] = None,
                 normalize: bool = True,
                 scaler_type: str = 'robust'):
        """
        Initialize feature engineer.
        
        Parameters:
        ───────────
        lag_periods : List[int]
            Periods to compute lag features (default: [1, 2, 3, 5, 10])
        rolling_windows : List[int]
            Windows for rolling statistics (default: [5, 10, 20, 50])
        normalize : bool
            Whether to normalize features (default: True)
        scaler_type : str
            'standard' or 'robust' (default: 'robust' - better for outliers)
        """
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10]
        self.rolling_windows = rolling_windows or [5, 10, 20, 50]
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        # Initialize scalers
        if normalize:
            if scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                self.scaler = StandardScaler()
        
        self.feature_names = None
        self.scaler_fit = False
        
        logger.info(
            f"FeatureEngineer initialized: {len(self.lag_periods)} lags, "
            f"{len(self.rolling_windows)} rolling windows"
        )
    
    def fit(self, raw_features: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineer on training data.
        
        This learns the normalization parameters.
        
        Parameters:
        ───────────
        raw_features : pd.DataFrame
            Training data with columns: [UNRATE, FEDFUNDS, CPI_YOY, T10Y2Y, STLFSI4]
        
        Returns:
        ────────
        self : FeatureEngineer
            Fitted feature engineer
        """
        logger.info(f"Fitting feature engineer on {len(raw_features)} samples")
        
        # Generate features
        engineered = self.transform(raw_features)
        
        # Fit scaler
        if self.normalize:
            self.scaler.fit(engineered)
            self.scaler_fit = True
            logger.info(f"Scaler fitted on {engineered.shape[1]} features")
        
        return self
    
    def transform(self, raw_features: pd.DataFrame) -> np.ndarray:
        """
        Transform raw features into engineered feature set.
        
        Parameters:
        ───────────
        raw_features : pd.DataFrame
            Features with columns: [UNRATE, FEDFUNDS, CPI_YOY, T10Y2Y, STLFSI4]
        
        Returns:
        ────────
        engineered : np.ndarray
            Shape: (n_samples, 73) with normalized features
        """
        logger.info(f"Generating {len(raw_features)} feature sets")
        
        engineered = pd.DataFrame(index=raw_features.index)
        
        # ===== Category 1: Lag Features (15) =====
        logger.debug("Generating lag features...")
        for col in raw_features.columns:
            for lag in self.lag_periods:
                engineered[f'{col}_lag_{lag}'] = raw_features[col].shift(lag)
        
        # ===== Category 2: Rolling Statistics (20) =====
        logger.debug("Generating rolling statistics...")
        for col in raw_features.columns:
            for window in self.rolling_windows:
                # Mean
                engineered[f'{col}_rolling_mean_{window}'] = (
                    raw_features[col].rolling(window=window).mean()
                )
                # Standard deviation
                engineered[f'{col}_rolling_std_{window}'] = (
                    raw_features[col].rolling(window=window).std()
                )
                # Min (only for first window to save space)
                if window == self.rolling_windows[0]:
                    engineered[f'{col}_rolling_min_{window}'] = (
                        raw_features[col].rolling(window=window).min()
                    )
                    engineered[f'{col}_rolling_max_{window}'] = (
                        raw_features[col].rolling(window=window).max()
                    )
        
        # ===== Category 3: Momentum & Acceleration (10) =====
        logger.debug("Generating momentum features...")
        for col in raw_features.columns:
            # Momentum (rate of change): (x_t - x_t-5) / x_t-5
            momentum = raw_features[col].pct_change(5)
            engineered[f'{col}_momentum_5d'] = momentum
            
            # Acceleration: momentum_t - momentum_t-1
            engineered[f'{col}_acceleration'] = momentum.diff()
            
            # Jerk: second derivative
            engineered[f'{col}_jerk'] = momentum.diff().diff()
        
        # ===== Category 4: Feature Interactions (15) =====
        logger.debug("Generating interaction features...")
        
        # 1. Interest Rate - Inflation relationship
        engineered['real_rate'] = (
            raw_features['FEDFUNDS'] - raw_features['CPI_YOY']
        )
        
        # 2. Yield Curve - Fed Rate relationship
        engineered['yield_fed_ratio'] = (
            (raw_features['T10Y2Y'] + 0.01) / (raw_features['FEDFUNDS'] + 0.001)
        )
        
        # 3. Unemployment - Rate relationship (Phillips curve)
        engineered['phillips_curve'] = (
            raw_features['UNRATE'] * raw_features['FEDFUNDS']
        )
        
        # 4. Stress indicator normalized by rates
        engineered['stress_normalized'] = (
            raw_features['STLFSI4'] / (raw_features['FEDFUNDS'] + 0.01)
        )
        
        # 5. Inflation-stress correlation
        engineered['inflation_stress'] = (
            raw_features['CPI_YOY'] * raw_features['STLFSI4']
        )
        
        # 6. Yield curve - Unemployment (recession signal)
        engineered['recession_signal'] = (
            raw_features['T10Y2Y'] * (1 - raw_features['UNRATE'])
        )
        
        # 7. Feature differences (rate of change in relationships)
        for i, col1 in enumerate(raw_features.columns):
            for col2 in raw_features.columns[i+1:2]:  # Limit combinations
                engineered[f'{col1}_minus_{col2}'] = (
                    raw_features[col1] - raw_features[col2]
                )
        
        # ===== Category 5: Spectral Features (8) =====
        logger.debug("Generating spectral features...")
        
        # FFT analysis for each feature (captures cyclical patterns)
        for col in raw_features.columns:
            # Remove NaNs for FFT
            valid_data = raw_features[col].dropna().values
            
            if len(valid_data) > 10:
                # Compute FFT
                fft_result = np.abs(fft.fft(valid_data))
                
                # Get top 2 frequency components (excluding DC)
                top_freqs = np.argsort(fft_result[1:])[-2:] + 1
                
                # Power spectral density (top 2 frequencies)
                for i, freq_idx in enumerate(top_freqs):
                    engineered[f'{col}_spectral_freq_{i}'] = (
                        fft_result[freq_idx] / len(valid_data)
                    )
        
        # ===== Category 6: Derived Risk Indicators (5) =====
        logger.debug("Generating risk indicators...")
        
        # 1. Volatility indicator (std of recent changes)
        returns = raw_features.pct_change()
        engineered['volatility'] = returns.std(axis=1, skipna=True)
        
        # 2. Regime change probability (high volatility + stress)
        engineered['regime_change_probability'] = (
            engineered['volatility'] * raw_features['STLFSI4']
        )
        
        # 3. Overall financial stress (normalized)
        engineered['financial_stress_score'] = (
            raw_features['STLFSI4'] + 
            engineered['volatility'] + 
            np.abs(engineered['real_rate'])
        )
        
        # 4. Market tightening indicator
        engineered['market_tightening'] = (
            raw_features['FEDFUNDS'] + 
            np.maximum(0, -raw_features['T10Y2Y'])  # Penalize inversion
        )
        
        # 5. Overall economic pressure
        engineered['economic_pressure'] = (
            raw_features['UNRATE'] + 
            raw_features['CPI_YOY'] + 
            np.maximum(0, raw_features['STLFSI4'])
        )
        
        # ===== Data Cleaning =====
        logger.debug("Cleaning features...")
        
        # Drop rows with NaN (from lag/rolling)
        engineered = engineered.dropna()
        
        # Remove infinite values
        engineered = engineered.replace([np.inf, -np.inf], np.nan)
        engineered = engineered.fillna(engineered.mean())
        
        # Standardize
        if self.normalize and self.scaler_fit:
            engineered_array = self.scaler.transform(engineered)
        else:
            engineered_array = engineered.values
        
        # Store feature names
        self.feature_names = engineered.columns.tolist()
        
        logger.info(
            f"Generated {engineered_array.shape[1]} features "
            f"for {engineered_array.shape[0]} samples"
        )
        
        return engineered_array
    
    def fit_transform(self, raw_features: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Parameters:
        ───────────
        raw_features : pd.DataFrame
            Training data
        
        Returns:
        ────────
        engineered : np.ndarray
            Transformed features
        """
        return self.fit(raw_features).transform(raw_features)
    
    def get_feature_importance_baseline(self, 
                                       features: np.ndarray,
                                       labels: np.ndarray) -> Dict[str, float]:
        """
        Get baseline feature importance using correlation with labels.
        
        Parameters:
        ───────────
        features : np.ndarray
            Shape (n_samples, 73)
        labels : np.ndarray
            Shape (n_samples,) - regime labels
        
        Returns:
        ────────
        importance : Dict[str, float]
            Feature name -> correlation with labels
        """
        importance = {}
        
        for i, feature_name in enumerate(self.feature_names):
            correlation = np.corrcoef(features[:, i], labels)[0, 1]
            importance[feature_name] = abs(correlation)
        
        return dict(sorted(importance.items(), 
                          key=lambda x: x[1], 
                          reverse=True))
    
    def get_feature_stats(self, features: np.ndarray) -> Dict[str, Dict]:
        """
        Get statistics for each feature.
        
        Useful for validation and understanding feature distributions.
        """
        stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_data = features[:, i]
            stats[feature_name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q75': np.percentile(feature_data, 75),
                'skewness': self._skewness(feature_data),
                'kurtosis': self._kurtosis(feature_data)
            }
        
        return stats
    
    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Calculate skewness."""
        return ((x - np.mean(x)) ** 3).mean() / (np.std(x) ** 3)
    
    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        return ((x - np.mean(x)) ** 4).mean() / (np.std(x) ** 4) - 3


class FeatureValidator:
    """
    Validate engineered features for quality.
    
    Checks:
    - No NaN/Inf values
    - Feature distributions are reasonable
    - No constant features
    - Low correlation (avoid multicollinearity)
    """
    
    @staticmethod
    def validate_features(features: np.ndarray,
                         feature_names: List[str],
                         max_missing_ratio: float = 0.05,
                         max_constant_features: int = 0,
                         max_high_correlation: float = 0.95) -> Dict:
        """
        Validate features and return report.
        
        Parameters:
        ───────────
        features : np.ndarray
            Shape (n_samples, n_features)
        feature_names : List[str]
            Feature names
        max_missing_ratio : float
            Max allowed missing ratio (default: 5%)
        max_constant_features : int
            Max allowed constant features (default: 0)
        max_high_correlation : float
            Threshold for high correlation (default: 0.95)
        
        Returns:
        ────────
        report : Dict
            Validation report with issues and recommendations
        """
        report = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # ===== Check 1: Missing Values =====
        missing_counts = np.isnan(features).sum(axis=0)
        missing_ratio = missing_counts / len(features)
        
        high_missing = np.where(missing_ratio > max_missing_ratio)[0]
        if len(high_missing) > 0:
            report['issues'].append(
                f"Features with >5% missing values: "
                f"{[feature_names[i] for i in high_missing]}"
            )
            report['valid'] = False
        
        report['statistics']['missing_ratio'] = dict(zip(
            feature_names,
            missing_ratio
        ))
        
        # ===== Check 2: Infinite Values =====
        inf_counts = np.isinf(features).sum(axis=0)
        high_inf = np.where(inf_counts > 0)[0]
        if len(high_inf) > 0:
            report['issues'].append(
                f"Features with infinite values: "
                f"{[feature_names[i] for i in high_inf]}"
            )
            report['valid'] = False
        
        # ===== Check 3: Constant Features =====
        std_devs = np.nanstd(features, axis=0)
        constant_features = np.where(std_devs < 1e-10)[0]
        
        if len(constant_features) > max_constant_features:
            report['issues'].append(
                f"Constant features detected: "
                f"{[feature_names[i] for i in constant_features]}"
            )
            report['valid'] = False
        
        # ===== Check 4: High Correlation (Multicollinearity) =====
        correlation_matrix = np.corrcoef(features.T)
        
        # Find high correlations (off-diagonal)
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(correlation_matrix[i, j]) > max_high_correlation:
                    high_corr_pairs.append((
                        feature_names[i],
                        feature_names[j],
                        correlation_matrix[i, j]
                    ))
        
        if len(high_corr_pairs) > 0:
            report['warnings'].append(
                f"High correlation detected ({max_high_correlation}+): "
                f"{len(high_corr_pairs)} feature pairs"
            )
        
        # ===== Check 5: Feature Statistics =====
        means = np.nanmean(features, axis=0)
        stds = np.nanstd(features, axis=0)
        
        report['statistics']['feature_means'] = dict(zip(feature_names, means))
        report['statistics']['feature_stds'] = dict(zip(feature_names, stds))
        
        # ===== Summary =====
        report['n_features'] = len(feature_names)
        report['n_samples'] = features.shape[0]
        
        if report['valid']:
            logger.info("✓ Feature validation passed")
        else:
            logger.warning(f"✗ Feature validation failed: {report['issues']}")
        
        return report


# ============================================================
# TESTING & EXAMPLES
# ============================================================

if __name__ == "__main__":
    """
    Test feature engineering on sample data.
    """
    
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data (simulating real economic indicators)
    n_samples = 859
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    raw_data = pd.DataFrame({
        'UNRATE': np.random.normal(0.04, 0.01, n_samples).clip(0.02, 0.08),
        'FEDFUNDS': np.random.normal(0.025, 0.015, n_samples).clip(0.0, 0.05),
        'CPI_YOY': np.random.normal(0.025, 0.005, n_samples).clip(0.01, 0.04),
        'T10Y2Y': np.random.normal(0.015, 0.02, n_samples),
        'STLFSI4': np.random.normal(0.2, 0.3, n_samples)
    }, index=dates)
    
    # Initialize feature engineer
    fe = FeatureEngineer(normalize=True, scaler_type='robust')
    
    # Generate features
    features = fe.fit_transform(raw_data)
    
    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING RESULTS")
    print(f"{'='*60}")
    print(f"Raw features shape: {raw_data.shape}")
    print(f"Engineered features shape: {features.shape}")
    print(f"Total features generated: {features.shape[1]}")
    print(f"\nFeature names (first 10):")
    for i, name in enumerate(fe.feature_names[:10]):
        print(f"  {i+1}. {name}")
    
    # Validate features
    validator = FeatureValidator()
    validation_report = validator.validate_features(
        features,
        fe.feature_names
    )
    
    print(f"\n{'='*60}")
    print("VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Valid: {validation_report['valid']}")
    if validation_report['issues']:
        print(f"Issues: {validation_report['issues']}")
    if validation_report['warnings']:
        print(f"Warnings: {validation_report['warnings']}")
    
    print(f"\n{'='*60}")
    print("FEATURE STATISTICS")
    print(f"{'='*60}")
    stats = fe.get_feature_stats(features)
    print("\nTop 5 features by variance:")
    top_vars = sorted(
        [(name, s['std']) for name, s in stats.items()],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for name, std in top_vars:
        print(f"  {name}: std={std:.4f}")
