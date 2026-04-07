# Batch-Level Progress Logging

## Overview

The training now includes **real-time batch progress logging** that shows you exactly where you are in each epoch.

---

## Sample Progress Output

### Console Output During Training

```
================================================================================
Starting training for 1 epochs...
================================================================================

Epoch [1/1]
    [Train] Batch 1/6250 (  0.0%) | VFL: 0.234567 | BBox: 0.456789 | GIoU: 0.123456
    [Train] Batch 11/6250 (  0.2%) | VFL: 0.198765 | BBox: 0.398765 | GIoU: 0.087654
    [Train] Batch 21/6250 (  0.3%) | VFL: 0.187654 | BBox: 0.387654 | GIoU: 0.076543
    [Train] Batch 31/6250 (  0.5%) | VFL: 0.176543 | BBox: 0.376543 | GIoU: 0.065432
    ...
    [Train] Batch 6241/6250 ( 99.9%) | VFL: 0.087654 | BBox: 0.287654 | GIoU: 0.003210
  Train - VFL Loss: 0.145678, BBox Loss: 0.345678, GIoU Loss: 0.065432
    [Val]   Batch 1/1250 (  0.1%) | Mean IoU: 0.234567
    [Val]   Batch 21/1250 (  1.7%) | Mean IoU: 0.265432
    [Val]   Batch 41/1250 (  3.3%) | Mean IoU: 0.278901
    ...
    [Val]   Batch 1241/1250 ( 99.3%) | Mean IoU: 0.654321
  Val   - Mean IoU: 0.654321
  Time: 1234.56s

================================================================================
Training Complete!
Best validation IoU: 0.654321 (Epoch 1)
Total epochs: 1
Log file: checkpoints/logs/training_20260407_190215.log
================================================================================
```

---

## What You Can See

### Training Progress [Train]
- **Batch number**: Current batch / total batches
- **Progress percentage**: 0-100% of epoch completion
- **Current losses**: VFL, BBox, GIoU losses for this batch
- **Updates every 10 batches** (configurable)

### Validation Progress [Val]
- **Batch number**: Current validation batch / total validation batches
- **Progress percentage**: 0-100% of validation completion
- **Current Mean IoU**: Running average of IoU metric
- **Updates every 20 batches** (configurable)

### Final Epoch Summary
- **Average losses** across all training batches
- **Final validation IoU**
- **Epoch duration** in seconds

---

## Log File Content

Same progress is saved to the log file with timestamps:

```
2026-04-07 19:17:32,100 - training - INFO - Epoch [1/1]
2026-04-07 19:17:32,150 - training - INFO -     [Train] Batch 1/6250 (  0.0%) | VFL: 0.234567 | BBox: 0.456789 | GIoU: 0.123456
2026-04-07 19:17:35,200 - training - INFO -     [Train] Batch 11/6250 (  0.2%) | VFL: 0.198765 | BBox: 0.398765 | GIoU: 0.087654
2026-04-07 19:17:38,300 - training - INFO -     [Train] Batch 21/6250 (  0.3%) | VFL: 0.187654 | BBox: 0.387654 | GIoU: 0.076543
...
2026-04-07 19:51:42,200 - training - INFO -   Train - VFL Loss: 0.145678, BBox Loss: 0.345678, GIoU Loss: 0.065432
2026-04-07 19:51:42,300 - training - INFO -     [Val]   Batch 1/1250 (  0.1%) | Mean IoU: 0.234567
...
2026-04-07 20:01:15,100 - training - INFO -   Val   - Mean IoU: 0.654321
```

---

## Progress Logging Intervals

### Training Batches
- **Logged every 10 batches** (or at epoch end)
- Typical COCO: 6250 batches per epoch = ~625 progress lines
- With batch_size=4, shows progress every ~40 images

### Validation Batches
- **Logged every 20 batches** (or at epoch end)
- Typical COCO: 1250 batches per epoch = ~62 progress lines
- Helps track validation without too much log noise

---

## Using Progress Information

### During Training

**In main terminal:**
```bash
./train.sh standard
# Watch progress updates in real-time
```

**In separate terminal (tail logs):**
```bash
tail -f checkpoints/logs/training_*.log | grep "\[Train\]"
```

### Analyze Progress After Training

**Find first batch:**
```bash
grep "\[Train\] Batch 1/" checkpoints/logs/training_*.log | head -1
```

**Find last batch:**
```bash
grep "\[Train\]" checkpoints/logs/training_*.log | tail -1
```

**Get training time from logs:**
```bash
echo "Start:" && grep "Epoch \[" checkpoints/logs/training_*.log | head -1
echo "End:" && grep "Training Complete" checkpoints/logs/training_*.log
```

**Track validation progress:**
```bash
grep "\[Val\]" checkpoints/logs/training_*.log
```

---

## Loss Interpretation

### VFL Loss (Focal Loss)
- Classification loss
- **Lower is better**
- Should decrease over batches

### BBox Loss
- Bounding box regression loss
- **Lower is better**
- Should decrease over batches

### GIoU Loss
- IoU-based regression loss
- **Lower is better**
- Should decrease over batches

### Example Trend (good training):
```
Batch 1:   VFL: 0.234 | BBox: 0.456 | GIoU: 0.123
Batch 10:  VFL: 0.198 | BBox: 0.398 | GIoU: 0.087  ✓ Decreasing
Batch 50:  VFL: 0.145 | BBox: 0.345 | GIoU: 0.065  ✓ Still decreasing
Batch 100: VFL: 0.123 | BBox: 0.321 | GIoU: 0.054  ✓ Converging
```

---

## IoU Interpretation

### Mean IoU (Validation)
- Intersection over Union metric
- **Higher is better** (range: 0-1)
- Should increase during validation

### Example Trend (good validation):
```
[Val] Batch 1:    Mean IoU: 0.234
[Val] Batch 50:   Mean IoU: 0.456  ✓ Increasing
[Val] Batch 100:  Mean IoU: 0.567  ✓ Continuing to improve
[Val] Batch Final: Mean IoU: 0.654  ✓ Final result
```

---

## Troubleshooting Progress Logs

### No Progress Output
- Check if logger is being passed to functions
- Ensure you're not using `--no_save` (logs still work)
- Check `checkpoints/logs/` directory for log files

### Losses Not Decreasing
- Training might not be converging
- Try reducing learning rate: `--learning_rate 5e-5`
- Increase epochs: `--num_epochs 20`
- Check data loading (are samples valid?)

### Too Much/Too Little Output
- Progress updates every 10 training batches, 20 validation batches
- This is hardcoded in train.py (can be adjusted if needed)
- File logs capture everything; console shows subset

---

## Example Commands

```bash
# Train and watch progress in real-time
./train.sh quick

# Train with verbose output
./train.sh custom --num_epochs 1 --batch_size 8 --verbose

# Monitor in separate terminal
tail -f checkpoints/logs/training_*.log

# Get only training progress lines
tail -f checkpoints/logs/training_*.log | grep "\[Train\]"

# Count how many batches trained so far
grep -c "\[Train\]" checkpoints/logs/training_*.log

# Get session summary
grep -E "Epoch \[|Training Complete|Best" checkpoints/logs/training_*.log
```

---

## Next Steps

1. **Start training**: `./train.sh quick` or `./train.sh standard`
2. **Watch progress**: Monitor batch updates in real-time
3. **Analyze losses**: Check if they're decreasing as expected
4. **Review results**: Check final metric in log file

The batch-level progress logging gives you full visibility into the training process! 🎯
