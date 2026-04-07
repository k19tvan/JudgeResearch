# Quick Reference: train.sh

## Usage
```bash
./train.sh [option]
```

## Quick Start Options

| Command | Purpose | Epochs | Batch Size | Use Case |
|---------|---------|--------|-----------|----------|
| `./train.sh quick` | 1-epoch test | 1 | 8 | ✅ Quick testing |
| `./train.sh test` | 3-epoch validation | 3 | 8 | ✅ Validate setup |
| `./train.sh standard` | Default training | 5 | 8 | ✅ Regular training |
| `./train.sh long` | Extended training | 10 | 16 | ✅ Better convergence |
| `./train.sh production` | Full training | 20 | 16 | ✅ Best quality model |
| `./train.sh small` | Resource-limited | 5 | 4 | ✅ Low memory/GPU |
| `./train.sh large` | High performance | 10 | 32 | ✅ Better accuracy |
| `./train.sh cpu` | CPU-only training | 2 | 4 | ✅ No GPU needed |

## Custom Training
```bash
./train.sh custom --num_epochs 15 --batch_size 16 --learning_rate 5e-5
```

## Help
```bash
./train.sh --help
./train.sh -h
```

## Output
- **Logs**: Printed to terminal in real-time
- **Checkpoints**: Saved to `./checkpoints/[scenario_name]/`
- **Each checkpoint** contains: model weights, optimizer state, epoch, validation IoU

## Features
✅ Predefined training scenarios  
✅ Color-coded output  
✅ Environment auto-detection  
✅ Custom argument support  
✅ Automatic checkpoint saving  

## For Full Documentation
See: `TRAINING_GUIDE.md`
