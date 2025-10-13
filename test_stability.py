#!/usr/bin/env python3
"""
Test script to verify numerical stability fixes for training
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project path
sys.path.append('/home/test/pengjin/data1/HashVcmr')

def test_model_forward():
    """Test a simple forward pass to check for NaN issues"""
    print("Testing model forward pass...")
    
    from method_tvr.model_components import MILNCELoss, HashLayer
    from method_tvr.model import ReLoCLNet
    from easydict import EasyDict as edict
    
    # Create a minimal config for testing
    config = edict({
        'max_desc_l': 20,
        'max_ctx_l': 100,
        'hidden_size': 256,
        'input_drop': 0.1,
        'query_input_size': 300,
        'visual_input_size': 1024,
        'sub_input_size': 300,
        'n_heads': 8,
        'drop': 0.1,
        'initializer_range': 0.02,
        'conv_kernel_size': 5,
        'conv_stride': 1,
        'lw_fcl': 0.1,
        'lw_vcl': 0.1,
        'lw_st_ed': 1.0,
        'lw_neg_ctx': 0.1,
        'lw_neg_q': 0.1,
        'lw_q': 0.001,
        'lw_b': 0.001,
        'lw_rec': 1.0,
        'ranking_loss_type': 'hinge',
        'margin': 0.1,
        'use_hard_negative': False,
        'hard_pool_size': 16
    })
    
    # Create model
    model = ReLoCLNet(config)
    model.train()
    
    # Create dummy inputs
    batch_size = 2
    query_feat = torch.randn(batch_size, 10, 300)
    query_mask = torch.ones(batch_size, 10)
    video_feat = torch.randn(batch_size, 50, 1024) 
    video_mask = torch.ones(batch_size, 50)
    sub_feat = torch.randn(batch_size, 50, 300)
    sub_mask = torch.ones(batch_size, 50)
    st_ed_indices = torch.randint(0, 50, (batch_size, 2))
    match_labels = torch.zeros(batch_size, 50)
    # Set some positive labels
    match_labels[:, 10:15] = 1
    
    print("Input shapes:")
    print(f"  query_feat: {query_feat.shape}")
    print(f"  video_feat: {video_feat.shape}")
    print(f"  sub_feat: {sub_feat.shape}")
    
    try:
        # Forward pass
        loss, loss_dict = model(
            query_feat=query_feat,
            query_mask=query_mask,
            video_feat=video_feat,
            video_mask=video_mask,
            sub_feat=sub_feat,
            sub_mask=sub_mask,
            st_ed_indices=st_ed_indices,
            match_labels=match_labels
        )
        
        print(f"Forward pass successful!")
        print(f"Total loss: {loss}")
        print("Loss components:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v}")
            
        # Check for NaN/Inf
        has_nan = torch.isnan(loss) or torch.isinf(loss)
        print(f"Loss has NaN/Inf: {has_nan}")
        
        if not has_nan:
            # Test backward pass
            print("Testing backward pass...")
            loss.backward()
            print("Backward pass successful!")
            
            # Check gradients
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    print(f"NaN/Inf gradient in {name}")
                    has_nan_grad = True
            
            if not has_nan_grad:
                print("No NaN/Inf gradients detected!")
                return True
            else:
                print("NaN/Inf gradients detected!")
                return False
        else:
            print("NaN/Inf in loss!")
            return False
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_components():
    """Test specific components that were causing issues"""
    print("Testing specific components...")
    
    # Test MILNCELoss
    from method_tvr.model_components import MILNCELoss
    milnce = MILNCELoss()
    
    # Test with extreme values
    extreme_scores = torch.tensor([[100.0, -100.0], [-100.0, 100.0]])
    try:
        loss = milnce(q2ctx_scores=extreme_scores)
        print(f"MILNCELoss with extreme values: {loss}")
        print(f"Has NaN/Inf: {torch.isnan(loss) or torch.isinf(loss)}")
    except Exception as e:
        print(f"MILNCELoss failed: {e}")
        
    # Test HashLayer
    from method_tvr.model_components import HashLayer
    hash_layer = HashLayer(256, 1024)
    hash_layer.train()
    
    extreme_input = torch.randn(4, 256) * 100
    try:
        result = hash_layer(extreme_input, eta=10.0)
        print(f"HashLayer test successful")
        print(f"Has NaN: {torch.isnan(result.code).any() or torch.isnan(result.bin_like).any()}")
    except Exception as e:
        print(f"HashLayer failed: {e}")

def test_zero_inputs():
    """Test model behavior with all-zero inputs and/or zero masks"""
    print("Testing zero-input edge cases...")

    from method_tvr.model import ReLoCLNet
    from easydict import EasyDict as edict
    
    config = edict({
        'max_desc_l': 20,
        'max_ctx_l': 100,
        'hidden_size': 256,
        'input_drop': 0.1,
        'query_input_size': 300,
        'visual_input_size': 1024,
        'sub_input_size': 300,
        'n_heads': 8,
        'drop': 0.1,
        'initializer_range': 0.02,
        'conv_kernel_size': 5,
        'conv_stride': 1,
        'lw_fcl': 0.1,
        'lw_vcl': 0.1,
        'lw_st_ed': 1.0,
        'lw_neg_ctx': 0.1,
        'lw_neg_q': 0.1,
        'lw_q': 0.001,
        'lw_b': 0.001,
        'lw_rec': 1.0,
        'ranking_loss_type': 'hinge',
        'margin': 0.1,
        'use_hard_negative': False,
        'hard_pool_size': 16
    })

    model = ReLoCLNet(config)
    model.train()

    batch_size = 2
    Tq, Vv = 10, 50

    # Case A: zero features, valid masks
    query_feat = torch.zeros(batch_size, Tq, 300)
    query_mask = torch.ones(batch_size, Tq)
    video_feat = torch.zeros(batch_size, Vv, 1024)
    video_mask = torch.ones(batch_size, Vv)
    sub_feat = torch.zeros(batch_size, Vv, 300)
    sub_mask = torch.ones(batch_size, Vv)
    st_ed_indices = torch.randint(0, Vv, (batch_size, 2))
    match_labels = torch.zeros(batch_size, Vv)

    loss, loss_dict = model(
        query_feat=query_feat,
        query_mask=query_mask,
        video_feat=video_feat,
        video_mask=video_mask,
        sub_feat=sub_feat,
        sub_mask=sub_mask,
        st_ed_indices=st_ed_indices,
        match_labels=match_labels
    )
    print(f"Zero features - loss: {loss}")
    assert torch.isfinite(loss), "Loss is not finite for zero features"
    loss.backward()
    print("Zero features - backward OK")

    # Case B: valid features, zero masks
    query_feat = torch.randn(batch_size, Tq, 300)
    query_mask = torch.zeros(batch_size, Tq)
    video_feat = torch.randn(batch_size, Vv, 1024)
    video_mask = torch.zeros(batch_size, Vv)
    sub_feat = torch.randn(batch_size, Vv, 300)
    sub_mask = torch.zeros(batch_size, Vv)
    st_ed_indices = torch.randint(0, Vv, (batch_size, 2))
    match_labels = torch.zeros(batch_size, Vv)

    model.zero_grad()
    loss2, loss_dict2 = model(
        query_feat=query_feat,
        query_mask=query_mask,
        video_feat=video_feat,
        video_mask=video_mask,
        sub_feat=sub_feat,
        sub_mask=sub_mask,
        st_ed_indices=st_ed_indices,
        match_labels=match_labels
    )
    print(f"Zero masks - loss: {loss2}")
    assert torch.isfinite(loss2), "Loss is not finite for zero masks"
    loss2.backward()
    print("Zero masks - backward OK")

if __name__ == "__main__":
    print("=" * 60)
    print("NUMERICAL STABILITY INTEGRATION TEST")
    print("=" * 60)
    
    test_specific_components()
    print()
    
    success = test_model_forward()

    print()
    test_zero_inputs()
    
    if success:
        print("\n✅ All tests passed! The model should now be stable for training.")
    else:
        print("\n❌ Tests failed. There are still numerical stability issues.")