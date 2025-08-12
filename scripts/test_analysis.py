#!/usr/bin/env python3
"""
Simple test script for ARFIMA and DFA analysis.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.arfima_modelling import ARFIMAModel, arfima_simulation, estimate_arfima_order
from analysis.dfa_analysis import DFAModel, dfa, hurst_from_dfa_alpha, d_from_hurst
from analysis.rs_analysis import RSModel, rs_analysis, d_from_hurst_rs, alpha_from_hurst_rs
from analysis.mfdfa_analysis import MFDFAModel, mfdfa, hurst_from_mfdfa, alpha_from_mfdfa

def main():
    print("Testing ARFIMA and DFA Analysis")
    print("=" * 40)
    
    # Generate test data
    print("1. Generating test data...")
    np.random.seed(42)
    y = arfima_simulation(500, d=0.3, ar_params=np.array([0.5]), ma_params=np.array([0.3]))
    print(f"   Generated {len(y)} points")
    
    # Test DFA
    print("\n2. Testing DFA...")
    try:
        scales, flucts, summary = dfa(y, order=1)
        print(f"   DFA alpha: {summary.alpha:.3f}")
        print(f"   DFA r-value: {summary.rvalue:.3f}")
        print("   ✓ DFA test passed")
    except Exception as e:
        print(f"   ✗ DFA test failed: {e}")
        return
    
    # Test R/S
    print("\n3. Testing R/S...")
    try:
        scales, rs_values, summary = rs_analysis(y)
        print(f"   R/S Hurst: {summary.hurst:.3f}")
        print(f"   R/S r-value: {summary.rvalue:.3f}")
        print("   ✓ R/S test passed")
    except Exception as e:
        print(f"   ✗ R/S test failed: {e}")
        return
    
    # Test MFDFA
    print("\n4. Testing MFDFA...")
    try:
        scales, fq, summary = mfdfa(y)
        hq_2 = hurst_from_mfdfa(summary.hq, summary.q_values, q_target=2.0)
        print(f"   MFDFA h(q=2): {hq_2:.3f}")
        print(f"   MFDFA q range: {len(summary.q_values)} values")
        print("   ✓ MFDFA test passed")
    except Exception as e:
        print(f"   ✗ MFDFA test failed: {e}")
        return
    
    # Test ARFIMA
    print("\n5. Testing ARFIMA...")
    try:
        p, d_est, q = estimate_arfima_order(y, max_p=1, max_q=1)
        print(f"   Estimated order: ARFIMA({p}, {d_est:.3f}, {q})")
        
        model = ARFIMAModel(d=d_est, p=p, q=q)
        model.fit(y)
        
        print(f"   Fitted d: {model.params.d:.3f}")
        print(f"   Fitted p: {model.params.p}")
        print(f"   Fitted q: {model.params.q}")
        print("   ✓ ARFIMA test passed")
    except Exception as e:
        print(f"   ✗ ARFIMA test failed: {e}")
        return
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    main()
