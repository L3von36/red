# Phase 1 Implementation Plan: DualFlowTransformer

## Architecture Design

### Current Flow
```
Input [N, T] → DualFlowCell (bidirectional GRU) → Speed/Jam predictions [N, T]
```

### New Flow
```
Input [N, T] 
  ↓
DualFlowCell (bidirectional, WITH hidden states) → Features [N, T, hidden=64]
  ↓
TransformerEncoder (4 layers, 8 heads, temporal attention) → Refined features [N, T, hidden]
  ↓
Output projection → Predictions [N, T]
```

## Key Changes

### 1. Modify DualFlowCell to Support Hidden State Output
- Add `return_hidden=True` parameter
- Return both predictions AND hidden states from forward()
- Hidden states: [N, T, hidden_dim]

### 2. Create DualFlowTransformer Class
- Wrap the modified bidirectional DualFlowCells
- Add TransformerEncoder on top of GRU hidden states
- Temporal self-attention learns:
  - Which timepoints are informative
  - Long-range temporal dependencies
  - Which nodes to trust (via positional encoding)

### 3. Integrate Smoothly
- Keep all existing losses (S1, S2, S3)
- Keep training interface identical
- Just swap DualFlow → DualFlowTransformer in training code

## Implementation Steps

1. ✅ Modify DualFlowCell.forward() to optionally return hidden states
2. ✅ Create TransformerEncoderBlock class
3. ✅ Create DualFlowTransformer wrapper
4. ✅ Update training function to use DualFlowTransformer
5. ✅ Test and validate
6. ✅ Commit

## Expected Performance
- Baseline DualFlow: 3.41 km/h MAE
- DualFlowTransformer Phase 1: 2.9-3.0 km/h MAE (10-12% improvement)
- Target: Beat HA baseline (2.65 km/h) eventually with Phases 2-3

## Risk Mitigation
- Keep old DualFlow code intact (can revert if needed)
- Add `use_transformer=True/False` flag to enable/disable
- Validate on small batch first before full training
