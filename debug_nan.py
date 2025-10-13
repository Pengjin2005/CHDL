#!/usr/bin/env python3
"""
Debug script to test numerical stability fixes
"""

import torch
import torch.nn.functional as F
import math

def test_log_cosh_stability():
    """Test the numerical stability of log_cosh function"""
    print("Testing log_cosh stability...")
    
    # Original unstable version
    def _log_cosh_old(x):
        return F.softplus(2 * x) - math.log(2.0)
    
    # New stable version
    def _log_cosh_new(x):
        abs_x = torch.abs(x)
        abs_x_clamped = torch.clamp(abs_x, max=20.0)  
        return abs_x_clamped + torch.log1p(torch.exp(-2.0 * abs_x_clamped))
    
    # Test with extreme values
    test_values = torch.tensor([-100.0, -50.0, -10.0, 0.0, 10.0, 50.0, 100.0])
    
    print("Input values:", test_values)
    
    try:
        old_results = _log_cosh_old(test_values)
        print("Old function results:", old_results)
        print("Old function has NaN/Inf:", torch.isnan(old_results).any() or torch.isinf(old_results).any())
    except Exception as e:
        print("Old function failed:", e)
    
    try:
        new_results = _log_cosh_new(test_values)
        print("New function results:", new_results)
        print("New function has NaN/Inf:", torch.isnan(new_results).any() or torch.isinf(new_results).any())
    except Exception as e:
        print("New function failed:", e)
    
    print()

def test_hash_layer_stability():
    """Test HashLayer forward pass stability"""
    print("Testing HashLayer stability...")
    
    # Import the fixed HashLayer
    import sys
    sys.path.append('/home/test/pengjin/data1/HashVcmr')
    from method_tvr.model_components import HashLayer
    
    # Create a hash layer
    hash_layer = HashLayer(512, 1024)
    hash_layer.train()
    
    # Test with extreme input values
    extreme_input = torch.randn(4, 512) * 100  # Very large values
    normal_input = torch.randn(4, 512)
    
    print("Testing with normal input...")
    try:
        result_normal = hash_layer(normal_input, eta=1.0)
        print("Normal input - Success")
        print("Has NaN:", torch.isnan(result_normal.code).any() or torch.isnan(result_normal.bin_like).any())
    except Exception as e:
        print("Normal input failed:", e)
    
    print("Testing with extreme input...")
    try:
        result_extreme = hash_layer(extreme_input, eta=10.0)  # Large eta
        print("Extreme input - Success")
        print("Has NaN:", torch.isnan(result_extreme.code).any() or torch.isnan(result_extreme.bin_like).any())
    except Exception as e:
        print("Extreme input failed:", e)
    
    print()

def test_contrastive_stability():
    """Test contrastive loss stability"""
    print("Testing contrastive loss stability...")
    
    import sys
    sys.path.append('/home/test/pengjin/data1/HashVcmr')
    from method_tvr.contrastive import get_positive_expectation, get_negative_expectation
    
    # Test with extreme values
    extreme_pos = torch.tensor([100.0, -100.0, 50.0, -50.0])
    extreme_neg = torch.tensor([100.0, -100.0, 50.0, -50.0])
    
    measures = ['JSD', 'KL', 'H2', 'DV']
    
    for measure in measures:
        print(f"Testing measure: {measure}")
        try:
            pos_result = get_positive_expectation(extreme_pos, measure=measure)
            neg_result = get_negative_expectation(extreme_neg, measure=measure)
            print(f"  Positive: {pos_result}, Negative: {neg_result}")
            print(f"  Has NaN: {torch.isnan(pos_result) or torch.isnan(neg_result)}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print()

if __name__ == "__main__":
    print("=" * 50)
    print("NUMERICAL STABILITY DEBUG TESTS")
    print("=" * 50)
    
    test_log_cosh_stability()
    test_hash_layer_stability()
    test_contrastive_stability()
    
    print("Debug tests completed!")