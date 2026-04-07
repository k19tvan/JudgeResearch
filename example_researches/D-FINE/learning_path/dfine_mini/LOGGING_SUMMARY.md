# Logging System - Summary

## ✅ What's Been Added

Your training script now includes **comprehensive logging** that tracks:

### 📊 **Per-Epoch Metrics**
- Training losses (VFL, BBox, GIoU)
- Validation IoU
- Epoch duration in seconds
- Best model indicator

### 📝 **Session Information**
- Training start/end times
- Device info (CPU/CUDA availability)
- Dataset paths and split names
- Model architecture details
- All hyperparameters used
- Checkpoint locations

### 📂 **Log Files**
- Automatically created in: `{save_dir}/logs/training_YYYYMMDD_HHMMSS.log`
- Timestamped entries for full audit trail
- Supports both console and file output

---

## 🚀 **Quick Start**

### Run Training (Logs Auto-Created)
```bash
./train.sh quick      # Logs to: ./checkpoints/logs/
./train.sh standard   # Or any other preset
./train.sh cpu

# Custom with specific save location
./train.sh custom --num_epochs 5 --save_dir ./my_experiment
# Logs go to: ./my_experiment/logs/
```

### View Logs

**Real-time during training** (in another terminal):
```bash
tail -f checkpoints/logs/training_*.log
```

**After training** (review complete log):
```bash
cat checkpoints/logs/training_YYYYMMDD_HHMMSS.log
```

**Find best results**:
```bash
grep "New best IoU" checkpoints/logs/training_*.log
```

---

## 📋 **Logged Information**

### Console Output (Real-Time)
```
Epoch [1/5]
  Train - VFL Loss: 0.234567, BBox Loss: 0.456789, GIoU Loss: 0.123456
  Val   - Mean IoU: 0.654321
  Time: 156.42s
  *** New best IoU: 0.654321 ***
  Checkpoint saved: checkpoint_epoch_1.pt
```

### Log File (Detailed)
```
2026-04-07 19:02:15,123 - INFO - DFINE Training Session Started
2026-04-07 19:02:15,200 - INFO - Device: cuda
2026-04-07 19:02:15,201 - INFO - CUDA available: True
2026-04-07 19:02:15,300 - INFO - Model parameters: 12,345,678
2026-04-07 19:02:15,400 - INFO - Train dataset: 25000 images (3125 batches)
2026-04-07 19:17:32,100 - INFO - Epoch [1/5]
2026-04-07 19:17:32,102 - INFO -   Train - VFL Loss: 0.234567, ...
2026-04-07 19:33:07,201 - INFO - Training Complete!
2026-04-07 19:33:07,202 - INFO - Best validation IoU: 0.678901 (Epoch 2)
```

---

## 🔍 **Log Analysis Examples**

### Get Training Summary
```bash
tail -20 checkpoints/logs/training_*.log
```

### Count How Many Epochs Completed
```bash
grep -c "Epoch \[" checkpoints/logs/training_*.log
```

### Find Warnings or Errors
```bash
grep "WARNING\|ERROR" checkpoints/logs/training_*.log
```

### Track Best Model Progress
```bash
grep "New best IoU\|Epoch \[" checkpoints/logs/training_*.log
```

### Extract Just Losses
```bash
grep "Train - VFL" checkpoints/logs/training_*.log
```

### Get Training Duration
```bash
head -1 checkpoints/logs/training_*.log  # Start time
tail -5 checkpoints/logs/training_*.log  # End time
```

---

## 📚 **Documentation**

For complete logging details, see: **[LOGGING.md](LOGGING.md)**

### Other Useful Docs:
- **[QUICK_START.md](QUICK_START.md)** - Quick reference for train.sh
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Full argument documentation

---

## 💡 **Tips**

✓ Logs saved to files for review even if terminal closes  
✓ Use `--verbose` flag for more console output  
✓ Logs include timestamps for performance analysis  
✓ Easy to grep for specific metrics or epochs  
✓ Complete session information for reproducibility  

---

## 🎯 **Next Steps**

1. **Start Training**: `./train.sh quick` or `./train.sh standard`
2. **Monitor Progress**: Check console output or `tail -f` log file
3. **Review Results**: Check log file after training completes
4. **Analyze Performance**: Use grep commands to extract specific metrics

Enjoy your training! 🚀
