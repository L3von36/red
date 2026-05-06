#!/usr/bin/env python3
"""
Pre-flight validation for DualFlowTransformer Phase 1
Checks for common errors before training starts
"""

import torch
import numpy as np
from torch import nn
import sys

def validate_setup():
    """Validate GPU, imports, and basic setup"""
    print("=" * 80)
    print("PRE-FLIGHT VALIDATION FOR PHASE 1")
    print("=" * 80)

    # Check 1: CUDA availability
    print("\n✓ Check 1: CUDA/GPU")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  ✅ CUDA available: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"  ⚠️  WARNING: CUDA not available, using CPU (SLOW!)")

    # Check 2: Required imports
    print("\n✓ Check 2: Required imports")
    try:
        from torch_geometric.nn import ChebConv
        print("  ✅ torch_geometric.nn.ChebConv available")
    except:
        print("  ❌ ERROR: torch_geometric not installed")
        return False

    try:
        import torch.nn.functional as F
        print("  ✅ torch.nn.functional available")
    except:
        print("  ❌ ERROR: torch.nn.functional not found")
        return False

    # Check 3: Shape compatibility for typical batch
    print("\n✓ Check 3: Tensor shape compatibility")
    try:
        N, T, hidden = 200, 288, 64

        # Simulate hidden states
        h_fwd = torch.randn(N, T, hidden, device=device)
        h_bwd = torch.randn(N, T, hidden, device=device)
        w = torch.rand(N, T, 2, device=device)
        w = w / w.sum(dim=-1, keepdim=True)  # normalize

        # Test fusing
        h_fused = h_fwd * w[:, :, 0:1] + h_bwd * w[:, :, 1:2]
        assert h_fused.shape == (N, T, hidden), f"Shape mismatch: {h_fused.shape}"
        print(f"  ✅ Tensor shapes compatible: {h_fused.shape}")
    except Exception as e:
        print(f"  ❌ ERROR in shape compatibility: {e}")
        return False

    # Check 4: TransformerEnhancer forward pass
    print("\n✓ Check 4: TransformerEnhancer forward pass")
    try:
        class TransformerEnhancer(nn.Module):
            def __init__(self, hidden=64, num_layers=3, num_heads=4, ff_dim=256, dropout=0.1):
                super().__init__()
                self.hidden = hidden
                self.pos_encoder = nn.Embedding(512, hidden)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=num_heads, dim_feedforward=ff_dim,
                    dropout=dropout, batch_first=True, activation='relu'
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.out_proj = nn.Linear(hidden, 1)

            def forward(self, hidden_states, mask=None):
                N, T, H = hidden_states.shape
                positions = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(N, -1)
                pos_enc = self.pos_encoder(positions)
                hidden_with_pos = hidden_states + pos_enc
                refined = self.transformer(hidden_with_pos)
                refined_preds = self.out_proj(refined).squeeze(-1)
                return refined_preds

        trans = TransformerEnhancer(hidden=hidden).to(device)
        h_test = torch.randn(N, T, hidden, device=device)
        out = trans(h_test)
        assert out.shape == (N, T), f"Transformer output shape wrong: {out.shape}"
        print(f"  ✅ TransformerEnhancer works: input {h_test.shape} → output {out.shape}")
    except Exception as e:
        print(f"  ❌ ERROR in TransformerEnhancer: {e}")
        return False

    # Check 5: Device consistency
    print("\n✓ Check 5: Device consistency")
    try:
        x = torch.randn(N, T, device=device)
        m = torch.rand(N, T, device=device)
        assert x.device == m.device == device, "Device mismatch"
        print(f"  ✅ All tensors on same device: {device}")
    except Exception as e:
        print(f"  ❌ ERROR in device setup: {e}")
        return False

    # Check 6: Broadcasting rules
    print("\n✓ Check 6: Broadcasting rules")
    try:
        a = torch.randn(N, T, hidden, device=device)
        b = torch.randn(N, T, 1, device=device)
        c = a * b
        assert c.shape == (N, T, hidden), f"Broadcasting failed: {c.shape}"

        d = torch.randn(N, T, 2, device=device)
        e = d[:, :, 0:1]
        f = a * e
        assert f.shape == (N, T, hidden), f"Slicing broadcast failed: {f.shape}"
        print(f"  ✅ Broadcasting rules work correctly")
    except Exception as e:
        print(f"  ❌ ERROR in broadcasting: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("=" * 80)
    print("\nReady to run Phase 1 training!\n")
    return True

if __name__ == '__main__':
    success = validate_setup()
    sys.exit(0 if success else 1)
