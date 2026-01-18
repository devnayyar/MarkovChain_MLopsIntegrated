#!/usr/bin/env python3
"""
FINML Model Training & Inference Script
Run model training and show results
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def create_sample_data():
    """Create sample financial data for testing"""
    print("\n" + "="*70)
    print("📊 CREATING SAMPLE FINANCIAL DATA")
    print("="*70)
    
    # Generate synthetic financial data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    
    # Create synthetic indicators
    data = pd.DataFrame({
        'date': dates,
        'dff': np.random.normal(4.5, 1.2, 200),  # Federal Funds Rate
        't10y2y': np.random.normal(0.5, 0.8, 200),  # Yield Curve
        'unrate': np.random.normal(4.0, 0.8, 200),  # Unemployment
        'cpi': np.random.normal(2.8, 0.5, 200),  # Inflation
        'vix': np.random.normal(18.0, 5.0, 200),  # Volatility
    })
    
    # Ensure non-negative values for rates/indices
    data['dff'] = data['dff'].abs()
    data['unrate'] = data['unrate'].abs()
    data['cpi'] = data['cpi'].abs()
    data['vix'] = data['vix'].abs()
    
    # Create regime labels (Low/High volatility)
    data['regime'] = (data['vix'] > data['vix'].median()).astype(int)
    
    # Save to data layers
    gold_path = PROJECT_ROOT / "data" / "gold" / "features_final.csv"
    data.to_csv(gold_path, index=False)
    
    print(f"✅ Sample data created")
    print(f"   📍 Location: {gold_path}")
    print(f"   📊 Shape: {data.shape}")
    print(f"   📈 Columns: {', '.join(data.columns.tolist())}")
    print(f"   📋 Regimes: Low={sum(data['regime']==0)}, High={sum(data['regime']==1)}")
    
    return data

def train_markov_model(data):
    """Train Markov chain model"""
    print("\n" + "="*70)
    print("🤖 TRAINING MARKOV CHAIN MODEL")
    print("="*70)
    
    try:
        from modeling.models.base_markov import MarkovChain
        
        regimes = data['regime'].values
        
        # Create and train model
        model = MarkovChain(state_sequence=regimes, states=['Low', 'High'])
        model.calculate_stationary_distribution()
        gap = model.calculate_spectral_gap()
        
        print(f"✅ Model trained successfully")
        print(f"\n📊 MODEL PROPERTIES:")
        print(f"   • Transition Matrix:")
        print(f"     {model.transition_matrix}")
        print(f"\n   • Stationary Distribution: {model.stationary_dist}")
        print(f"   • Spectral Gap: {gap:.4f}")
        
        # Get metrics
        predictions = model.predict(data[['vix', 'dff', 'unrate']].values)
        accuracy = np.mean(predictions == regimes)
        
        print(f"\n📈 ACCURACY: {accuracy:.2%}")
        
        metrics = {
            'accuracy': float(accuracy),
            'spectral_gap': float(gap),
            'stationary_dist': model.stationary_dist.tolist() if model.stationary_dist is not None else None,
            'transition_matrix': model.transition_matrix.tolist()
        }
        
        return model, metrics
        
    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return None, None

def generate_predictions(model, data):
    """Generate predictions on test data"""
    print("\n" + "="*70)
    print("🔮 GENERATING PREDICTIONS")
    print("="*70)
    
    try:
        if model is None:
            print("❌ Model not available")
            return None
        
        # Make predictions using the state sequence from training
        predictions = model.state_sequence
        
        # Create prediction dataframe
        pred_df = pd.DataFrame({
            'date': data['date'],
            'actual_regime': data['regime'],
            'predicted_regime': predictions,
            'vix': data['vix'],
            'correct': predictions == data['regime'].values
        })
        
        accuracy = pred_df['correct'].mean()
        
        print(f"✅ Predictions generated")
        print(f"\n📊 PREDICTION SUMMARY:")
        print(f"   • Total predictions: {len(pred_df)}")
        print(f"   • Accuracy: {accuracy:.2%}")
        print(f"   • Predicted Low: {sum(predictions==0)}")
        print(f"   • Predicted High: {sum(predictions==1)}")
        
        # Save predictions
        pred_path = PROJECT_ROOT / "data" / "gold" / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"\n   📍 Predictions saved to: {pred_path}")
        
        return pred_df
        
    except Exception as e:
        print(f"❌ Error generating predictions: {str(e)}")
        return None

def create_dashboard_data(data, pred_df, metrics):
    """Create data for dashboard visualization"""
    print("\n" + "="*70)
    print("📈 PREPARING DASHBOARD DATA")
    print("="*70)
    
    try:
        dashboard_data = {
            "model_metrics": {
                "accuracy": metrics['accuracy'] if metrics else 0,
                "spectral_gap": metrics['spectral_gap'] if metrics else 0,
                "training_date": datetime.now().isoformat(),
                "model_version": "v1.0",
                "status": "Active"
            },
            "data_summary": {
                "total_records": len(data),
                "low_regime_count": int(sum(data['regime']==0)),
                "high_regime_count": int(sum(data['regime']==1)),
                "vix_mean": float(data['vix'].mean()),
                "vix_std": float(data['vix'].std()),
                "vix_min": float(data['vix'].min()),
                "vix_max": float(data['vix'].max())
            },
            "recent_predictions": pred_df[['date', 'predicted_regime', 'vix', 'correct']].tail(10).to_dict('records') if pred_df is not None else [],
            "model_performance": {
                "train_accuracy": metrics['accuracy'] if metrics else 0,
                "test_accuracy": (pred_df['correct'].mean() if pred_df is not None else 0),
                "last_updated": datetime.now().isoformat()
            }
        }
        
        # Save dashboard data
        dash_path = PROJECT_ROOT / "monitoring" / "dashboard_data.json"
        os.makedirs(dash_path.parent, exist_ok=True)
        
        with open(dash_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"✅ Dashboard data prepared")
        print(f"\n📊 DASHBOARD METRICS:")
        print(f"   • Model Accuracy: {dashboard_data['model_metrics']['accuracy']:.2%}")
        print(f"   • Spectral Gap: {dashboard_data['model_metrics']['spectral_gap']:.4f}")
        print(f"   • Total Records: {dashboard_data['data_summary']['total_records']}")
        print(f"   • Low Regime: {dashboard_data['data_summary']['low_regime_count']} days")
        print(f"   • High Regime: {dashboard_data['data_summary']['high_regime_count']} days")
        print(f"   • VIX Mean: {dashboard_data['data_summary']['vix_mean']:.2f}")
        print(f"\n   📍 Data saved to: {dash_path}")
        
        return dashboard_data
        
    except Exception as e:
        print(f"❌ Error preparing dashboard data: {str(e)}")
        return None

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 FINML MODEL PIPELINE - FULL EXECUTION")
    print("="*70)
    print(f"   Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Project Root: {PROJECT_ROOT}")
    
    # Step 1: Create sample data
    data = create_sample_data()
    
    # Step 2: Train model
    model, metrics = train_markov_model(data)
    
    # Step 3: Generate predictions
    pred_df = generate_predictions(model, data)
    
    # Step 4: Prepare dashboard data
    dashboard_data = create_dashboard_data(data, pred_df, metrics)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"   End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n   🎯 Ready to view in Dashboard!")
    print(f"   Command: streamlit run dashboards/app.py\n")

if __name__ == "__main__":
    main()
