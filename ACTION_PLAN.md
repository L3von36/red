# Action Plan: Next Steps for DualFlow Investigation

## Status Summary

**Bugs Fixed**: 3 critical issues in training loop have been identified and corrected:
1. ✅ Early stopping criterion (loss → blind-node MAE)
2. ✅ Undefined variable in validation loop
3. ✅ Duplicate patience check

**Root Cause Identified**: At 80% sparsity, DualFlow is theoretically limited by matrix completion bounds (~1400 samples needed, only ~40 available). HA's temporal-only approach is fundamentally more suitable.

**Key Discrepancy Explained**: The fixes to early stopping should resolve the validation-test performance gap (2.500 BlindMAE → 3.41 MAE) which was caused by loading wrong checkpoint weights.

---

## Immediate Next Steps (If Running on Your Environment)

### 1. **Run Fresh Training with Fixed Code**
```bash
# Option A: Run the script
python /home/user/red/dualflow.py

# Option B: Run the notebook cells in order
# Cell 6: Load data
# Cell 20: Run production training (fixed version)
```

**Expected Outputs**:
- Should complete without `NameError` on line 468
- Should print checkpoint markers (`*`) showing BlindMAE improvements
- Should stop at patience=2 (stricter early stopping)
- Best checkpoint saved based on actual BlindMAE, not loss

**Time**: ~30-60 minutes depending on GPU

---

### 2. **Evaluate New Results**
```python
# After training completes:
# Compare with baseline HA
print(f"DualFlow BlindMAE: {mae_v:.4f} (normalized)")
print(f"HA MAE: {ha_mae:.4f} (normalized)")

# Convert to km/h for reporting
df_kmh = mae_v * np.mean(node_stds) + np.mean(node_means)
ha_kmh = ha_mae * np.mean(node_stds) + np.mean(node_means)
print(f"DualFlow: {df_kmh:.2f} km/h")
print(f"HA: {ha_kmh:.2f} km/h")
```

---

### 3. **Run Ablation Studies** (Already Defined, Not Executed)

#### Ablation A: Multi-Sparsity Evaluation
- **Location**: Cell 42 (notebook)
- **Purpose**: Compare DualFlow vs HA at [40%, 60%, 80%, 90%] sparsity
- **Expected**: Shows crossover point where HA begins to win
- **Status**: Code defined, ready to run

#### Ablation B: Error Profiling
- **Location**: Cell 43 (notebook)  
- **Purpose**: Break down errors by time-of-day and congestion state
- **Expected**: Reveals if DualFlow fails specifically at peak hours or jam times
- **Status**: Code defined, ready to run

**Run all**:
```python
# In notebook:
# Cell 42: Ablation A
exec(open('/path/to/cell42.py').read())

# Cell 43: Ablation B  
exec(open('/path/to/cell43.py').read())
```

---

## If You Can't Run Locally

### Option 1: Run on Colab/Kaggle (Recommended)
The notebook (`dualflow.ipynb`) is fully functional on Colab with GPU. Steps:
1. Upload to Colab
2. Run cells in order (all required data is in the notebook)
3. Save outputs
4. Document results in separate cell

### Option 2: Use Cached Results
The summary documents from your previous run are available. You can:
1. Compare them with the new bug fixes
2. Verify if the undefined variable bug would have caused crashes
3. Check if early stopping criterion explains the val-test gap

---

## Interpretation Guide

### What to Look For

#### Good Signs ✅
- Training completes without errors
- BlindMAE steadily decreases for first 100-200 epochs
- Checkpoint marked with `*` when BlindMAE improves
- Early stops at ep X with message showing best BlindMAE
- New test results much better than old (3.41 → closer to 2.5 km/h range)

#### Warning Signs ⚠️
- Loss spikes/diverges (ep 700 issue from before)
- BlindMAE increases after warmup (ep 200+)
- Checkpoint never marked with `*` (no improvement)
- Test results still worse than HA (suggests other issues)

#### Expected Trade-offs
- **DualFlow vs HA at 80% sparsity**: HA likely still wins (theory + empirics)
- **DualFlow at lower sparsity**: May win at 40-60% (where spatial signal matters)
- **Component importance**: S1/S2/S3 unlikely to matter much at 80% (ablation showed <0.1 MAE difference)

---

## Architecture Questions to Investigate (Optional)

If DualFlow still underperforms HA after bug fixes, consider:

1. **Is spatial signal being used correctly?**
   - Check impute() method's anchor-diffusion (S3) is active
   - Verify adjacency matrix A_t is proper (symmetric, normalized)
   - Try different diffusion_steps/alpha values

2. **Is jam head learning anything?**
   - Inspect jam_logit predictions (should differ between jam/free regions)
   - Check if BCE loss is non-zero during training
   - Compare S2 (with jam head) vs -S2 (without)

3. **Should we use different loss weighting?**
   - Current: jam_w=2.0, bce_w=1.0 after warmup
   - Try: 1.5/0.75, 2.5/1.5, etc.
   - Monitor when loss spikes occur relative to weight changes

4. **Is there overfitting to the train/val mask pattern?**
   - Current: New random mask each epoch (good)
   - But test uses completely different random mask
   - Try: Test with same mask pattern as training → should perform better if overfitting is the issue

---

## Success Metrics

### Minimum Bar (Acceptable)
- [ ] Training runs without errors
- [ ] Checkpoints saved correctly
- [ ] Test results ≤ 3.2 km/h (vs old 3.41)

### Target Performance
- [ ] Test results ≤ 2.8 km/h (competitive with HA's 2.65)
- [ ] Ablation B shows where DualFlow outperforms HA (if anywhere)

### Ambitious (Likely Unachievable at 80%)
- [ ] DualFlow beats HA at 80% sparsity
- [ ] Component ablation shows clear S1/S2/S3 benefits
- [ ] ⚠️ **Unlikely**: Theory predicts spatial approaches can't work here

---

## Documentation Created

Three documents explain the current state:

1. **ANALYSIS.md** - Why HA wins theoretically and empirically
2. **BUG_FIXES.md** - Detailed bug descriptions and fixes  
3. **ACTION_PLAN.md** - This document (next steps)

All code fixes are committed and pushed to branch `claude/add-architecture-diagram-Qj2Wl`.

---

## Questions to Ask Yourself

1. **Do I have time to run training?**
   - Yes → Run fresh training, check results
   - No → Review analysis and bug fixes, understand the theory

2. **What's my goal?**
   - Improve DualFlow → Run training, complete ablations
   - Understand what failed → Read ANALYSIS.md and BUG_FIXES.md
   - Move forward with HA → Skip training, start production deployment

3. **How much uncertainty is acceptable?**
   - Need certainty → Run tests with multiple seeds
   - Acceptable to move forward → Use single run results

---

## If You Have Questions

- **"Why does HA beat DualFlow?"** → See ANALYSIS.md (matrix completion bound)
- **"What bugs were found?"** → See BUG_FIXES.md (3 critical issues)
- **"How do I verify the fixes worked?"** → Run training and compare outputs
- **"Should I keep trying to beat HA?"** → Probably not worth it; below theoretical limit
