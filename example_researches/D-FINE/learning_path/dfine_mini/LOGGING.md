# Training Logs Documentation

## Overview

The training script now includes comprehensive logging that records all important training information to both console and log files.

## Log File Location

Logs are automatically saved to:
```
{save_dir}/logs/training_YYYYMMDD_HHMMSS.log
```

**Default location:** `./checkpoints/logs/training_*.log`

**Example:**
```
checkpoints/
├── logs/
│   └── training_20260407_190215.log
├── checkpoint_epoch_1.pt
└── checkpoint_epoch_2.pt
```

## Console Output (Real-time)

The script displays essential information in the terminal as training progresses.

### Example Console Output:
```
Epoch [1/5]
  Train - VFL Loss: 0.234567, BBox Loss: 0.456789, GIoU Loss: 0.123456
  Val   - Mean IoU: 0.654321
  Time: 156.42s

Epoch [2/5]
  Train - VFL Loss: 0.198765, BBox Loss: 0.398765, GIoU Loss: 0.087654
  Val   - Mean IoU: 0.678901
  Time: 155.89s
  *** New best IoU: 0.678901 ***
  Checkpoint saved: checkpoint_epoch_2.pt
```

## Log File Contents

The log file contains detailed information including timestamps for all events:

### Session Start
```
2026-04-07 19:02:15,123 - training - INFO - ================================================================================
2026-04-07 19:02:15,124 - training - INFO - DFINE Training Session Started
2026-04-07 19:02:15,125 - training - INFO - Log file: /path/to/training_20260407_190215.log
2026-04-07 19:02:15,126 - training - INFO - ================================================================================
```

### Configuration Info
```
2026-04-07 19:02:15,200 - training - INFO - Random seed: 42
2026-04-07 19:02:15,300 - training - INFO - Device: cuda
2026-04-07 19:02:15,301 - training - INFO - CUDA available: True

2026-04-07 19:02:15,400 - training - INFO - --- Model Configuration ---
2026-04-07 19:02:15,401 - training - INFO - Number of classes: 80
2026-04-07 19:02:15,402 - training - INFO - Number of queries: 50
2026-04-07 19:02:15,403 - training - INFO - Model dimension: 128
2026-04-07 19:02:15,404 - training - INFO - Encoder layers: 2
2026-04-07 19:02:15,405 - training - INFO - Decoder layers: 3

2026-04-07 19:02:15,500 - training - INFO - Model parameters: 12,345,678

2026-04-07 19:02:15,600 - training - INFO - --- Training Configuration ---
2026-04-07 19:02:15,601 - training - INFO - Epochs: 5
2026-04-07 19:02:15,602 - training - INFO - Batch size: 8
2026-04-07 19:02:15,603 - training - INFO - Learning rate: 0.0001
2026-04-07 19:02:15,604 - training - INFO - Weight decay: 0.0001
2026-04-07 19:02:15,605 - training - INFO - Data workers: 4
```

### Dataset Information
```
2026-04-07 19:02:15,700 - training - INFO - --- Loading Dataset ---
2026-04-07 19:02:15,701 - training - INFO - Data directory: /path/to/data
2026-04-07 19:02:15,702 - training - INFO - Training split: train2017
2026-04-07 19:02:15,703 - training - INFO - Validation split: val2017
2026-04-07 19:02:16,200 - training - INFO - Train dataset: 25000 images (3125 batches)
2026-04-07 19:02:16,300 - training - INFO - Val dataset: 5000 images (625 batches)
2026-04-07 19:02:16,301 - training - INFO - Checkpoints will be saved to: ./checkpoints
```

### Training Progress
```
2026-04-07 19:02:16,400 - training - INFO - ================================================================================
2026-04-07 19:02:16,401 - training - INFO - Starting training for 5 epochs...
2026-04-07 19:02:16,402 - training - INFO - ================================================================================

2026-04-07 19:17:32,100 - training - INFO - Epoch [1/5]
2026-04-07 19:17:32,101 - training - DEBUG -   Processed 3125 batches in training epoch
2026-04-07 19:17:32,102 - training - INFO -   Train - VFL Loss: 0.234567, BBox Loss: 0.456789, GIoU Loss: 0.123456
2026-04-07 19:33:07,200 - training - DEBUG -   Processed 625 batches in evaluation epoch
2026-04-07 19:33:07,201 - training - INFO -   Val   - Mean IoU: 0.654321
2026-04-07 19:33:07,202 - training - INFO -   Time: 936.10s
2026-04-07 19:33:07,203 - training - INFO -   *** New best IoU: 0.654321 ***
2026-04-07 19:33:15,100 - training - INFO -   Checkpoint saved: checkpoint_epoch_1.pt
```

### Training Summary
```
2026-04-07 20:55:30,500 - training - INFO - ================================================================================
2026-04-07 20:55:30,501 - training - INFO - Training Complete!
2026-04-07 20:55:30,502 - training - INFO - Best validation IoU: 0.678901 (Epoch 2)
2026-04-07 20:55:30,503 - training - INFO - Total epochs: 5
2026-04-07 20:55:30,504 - training - INFO - Log file: /path/to/training_20260407_190215.log
2026-04-07 20:55:30,505 - training - INFO - ================================================================================
```

## Logged Information

### Per-Epoch Information
- **Epoch number** and total epochs
- **Training metrics**: VFL Loss, BBox Loss, GIoU Loss (averaged across batches)
- **Validation metrics**: Mean IoU
- **Epoch duration** in seconds
- **New best model** indicator
- **Checkpoint save** confirmation

### Debug Information (file only)
- Number of batches processed in training
- Number of batches processed in validation
- (Enable with `--verbose` flag for more details in console)

### Session Information
- Start/end time with timestamps
- Device information (CPU/CUDA)
- Random seed for reproducibility
- Model architecture details
- Training parameters
- Dataset information
- Checkpoint save location

## Viewing Logs

### Real-time In Terminal
Logs are printed during training. Use `--verbose` flag for more detailed console output:
```bash
./train.sh custom --num_epochs 5 --verbose
```

### After Training
View the complete log file:
```bash
cat checkpoints/logs/training_20260407_190215.log
```

### Last 50 Lines
```bash
tail -50 checkpoints/logs/training_20260407_190215.log
```

### Search for Specific Epochs
```bash
grep "Epoch \[3/" checkpoints/logs/training_20260407_190215.log
```

### Find Best IoU
```bash
grep "New best IoU" checkpoints/logs/training_20260407_190215.log
```

## Log Format

Each log line has the format:
```
TIMESTAMP - LOGGER_NAME - LOG_LEVEL - MESSAGE
```

**Example:**
```
2026-04-07 19:02:15,123 - training - INFO - Device: cuda
```

## Log Levels

- **INFO**: Important training events, metrics, milestones
- **DEBUG**: Detailed batch processing information (file only)
- **WARNING**: Non-critical issues (e.g., CUDA not available)
- **ERROR**: Critical failures (if any occur)

## Tips

1. **Monitor Training Progress**: Check console output for real-time updates
2. **Post-Training Analysis**: Review log file for complete history
3. **Find Issues**: Search for "WARNING" or "ERROR" in logs
4. **Track Best Model**: Look for "New best IoU" lines
5. **Reproducibility**: Log file stores seed and all hyperparameters

## Example Commands

```bash
# Train with logging to default location
./train.sh standard

# Train with custom save location (logs go there too)
./train.sh custom --num_epochs 10 --save_dir ./my_run

# View logs in real-time (in another terminal)
tail -f checkpoints/logs/training_*.log | grep Epoch

# Get training summary
grep -E "Training Complete|Best validation|Epoch \[" checkpoints/logs/training_*.log

# Count epochs trained
grep -c "Epoch \[" checkpoints/logs/training_*.log
```
