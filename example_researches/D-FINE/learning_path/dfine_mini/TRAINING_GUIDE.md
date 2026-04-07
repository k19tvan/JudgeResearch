# Training Guide for DFINE Object Detection

## Quick Start

### Basic Training (Default Configuration)
```bash
python train.py
```

### With Custom Hyperparameters
```bash
python train.py \
    --num_epochs 10 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --device cuda
```

---

## Command-Line Arguments

### Model Architecture Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_classes` | 80 | Number of object classes (80 for COCO) |
| `--num_queries` | 50 | Number of queries in DETR decoder |
| `--d_model` | 128 | Model embedding dimension |
| `--num_encoder_layers` | 2 | Number of transformer encoder layers |
| `--num_decoder_layers` | 3 | Number of transformer decoder layers |

**Example - Larger Model:**
```bash
python train.py \
    --d_model 256 \
    --num_encoder_layers 4 \
    --num_decoder_layers 6
```

---

### Training Hyperparameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Batch size for training |
| `--learning_rate` | 1e-4 | Initial learning rate for AdamW |
| `--weight_decay` | 1e-4 | Weight decay regularization |
| `--num_workers` | 4 | Number of data loading workers |
| `--seed` | 42 | Random seed for reproducibility |

**Example - Fast Training (1 epoch for testing):**
```bash
python train.py --num_epochs 1 --batch_size 32
```

**Example - Longer Training with Lower Learning Rate:**
```bash
python train.py \
    --num_epochs 20 \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4
```

---

### Data Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `./data` | Path to dataset directory |
| `--train_split` | train2017 | Training dataset split name |
| `--val_split` | val2017 | Validation dataset split name |

**Example - Custom Data Path:**
```bash
python train.py --data_dir /path/to/coco_dataset
```

---

### Device & Output Parameters

| Argument | Default | Description |
| `--device` | cuda | Device (cuda or cpu) |
| `--save_dir` | checkpoints | Directory for saving checkpoints |
| `--save_interval` | 1 | Save checkpoint every N epochs |
| `--no_save` | False | Don't save checkpoints |
| `--verbose` | False | Print detailed training progress |

**Example - CPU Training:**
```bash
python train.py --device cpu --num_workers 2
```

**Example - Save Every 5 Epochs with Detailed Output:**
```bash
python train.py \
    --save_interval 5 \
    --save_dir ./my_checkpoints \
    --verbose
```

---

## Common Training Scenarios

### 1. **Quick Test Run (1 epoch on GPU)**
```bash
python train.py --num_epochs 1 --batch_size 8 --save_dir test_run
```

### 2. **Production Training (20 epochs, larger batch)**
```bash
python train.py \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --num_workers 8 \
    --save_interval 2 \
    --verbose
```

### 3. **Smaller Model for Resource-Constrained Environment**
```bash
python train.py \
    --d_model 64 \
    --num_encoder_layers 1 \
    --num_decoder_layers 2 \
    --batch_size 4 \
    --num_workers 2
```

### 4. **Larger Model for Better Performance**
```bash
python train.py \
    --d_model 256 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --num_queries 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 5. **No Checkpoint Saving (Fast Prototyping)**
```bash
python train.py --num_epochs 5 --no_save
```

---

## Output Structure

### Training Progress
The script prints epoch-wise metrics:
```
Epoch 1/5:
  Train: avg_loss_vfl=0.1234, avg_loss_bbox=0.5678, avg_loss_giou=0.9012
  Val:   mean_iou=0.5123
```

### Saved Checkpoints
Checkpoints are saved in `--save_dir` directory:
```
checkpoints/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
└── ...
```

Each checkpoint contains:
- Model state dictionary
- Optimizer state dictionary
- Epoch number
- Validation IoU
- Model configuration

---

## Tips & Recommendations

### GPU Memory Optimization
- Reduce `--batch_size` if running out of memory
- Increase `--num_workers` for faster data loading
- Use smaller `--d_model` for memory-constrained GPUs

### Faster Training
- Increase `--num_workers` (parallel data loading)
- Increase `--batch_size` (larger batches = fewer iterations)
- Use `--device cuda` (ensure CUDA is available)

### Better Model Performance
- Increase `--num_epochs`
- Use smaller `--learning_rate` with more epochs
- Increase `--d_model` and num_decoder_layers
- Accumulate more training data

### Reproducibility
- Set `--seed` to fixed value (default: 42)
- Results will be reproducible across runs

---

## Loading Saved Checkpoints

To resume training from a checkpoint:

```python
import torch
from model import DFINEMini

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_10.pt')

# Recreate model with saved config
config = checkpoint['config']
model = DFINEMini(
    num_classes=config['num_classes'],
    num_queries=config['num_queries'],
    d_model=config['d_model'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])

# For inference
model.eval()
```

---

## Troubleshooting

### CUDA Out of Memory
```bash
python train.py --batch_size 4 --device cuda
```

### Slow Data Loading
```bash
python train.py --num_workers 8  # Increase workers
```

### Not Converging
```bash
python train.py --learning_rate 1e-5 --num_epochs 20
```

### Training on CPU
```bash
python train.py --device cpu --num_workers 2
```
