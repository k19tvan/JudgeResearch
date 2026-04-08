"""Problem 12: Training and Evaluation Loop"""
import torch
import torch.nn as nn
from typing import List
from torch.utils.data import DataLoader
import logging
import sys
from pathlib import Path
from datetime import datetime

from model import DFINEMini
from matcher import HungarianMatcher
from criterion import SetCriterion
from box_ops import box_cxcywh_to_xyxy
from iou import box_iou
from dataset import COCODataset, collate_fn


def setup_logger(log_dir, name='training'):
    """Setup logger that writes to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove existing handlers
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - includes all info
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler - less verbose
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, str(log_file)


def train_one_epoch(model, criterion, matcher, optimizer, data_loader, device, logger=None, use_multi_layer=False):
    """Run one full training epoch.
    
    Args:
        model: DFINEMini model
        criterion: SetCriterion for loss computation
        matcher: HungarianMatcher for bipartite matching
        optimizer: Optimization algorithm
        data_loader: Training data loader
        device: Device to train on
        logger: Optional logger for progress
        use_multi_layer: If True, use multi-layer supervision (matcher passed to criterion).
                        If False, call matcher externally and pass indices to criterion.
    """
    model.train()
    
    # Track main losses (will accumulate all losses)
    loss_keys = ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl"]
    totals = {k: 0.0 for k in loss_keys}
    batch_count = 0
    total_batches = len(data_loader)
    
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        out = model(images)
        
        # Multi-layer or single-layer loss computation
        if use_multi_layer:
            # Mode 2: Pass matcher to criterion for internal multi-layer handling
            losses = criterion(out, targets, matcher=matcher)
        else:
            # Mode 1: Traditional mode - compute indices separately
            indices = matcher(out, targets)["indices"]
            losses = criterion(out, targets, indices=indices)
        
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for loss_key in loss_keys:
            if loss_key in losses:
                totals[loss_key] += losses[loss_key].item()
        batch_count += 1
        
        # Log batch progress every 10 batches or at end
        if logger and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
            progress_pct = 100.0 * (batch_idx + 1) / total_batches
            log_msg = f"[Train] Batch {batch_idx+1}/{total_batches} ({progress_pct:5.1f}%) | "
            
            # Format available losses
            loss_strs = []
            for k in ["loss_vfl", "loss_bbox", "loss_giou", "loss_fgl"]:
                if k in losses:
                    loss_strs.append(f"{k.split('_')[1][:3]}: {losses[k].item():.6f}")
            
            log_msg += " | ".join(loss_strs)
            logger.info(log_msg)
    
    # Compute average metrics
    n = max(len(data_loader), 1)
    avg_metrics = {f"avg_{k}": v / n for k, v in totals.items() if v > 0}
    
    return avg_metrics


def evaluate(model, data_loader, device, logger=None):
    """Evaluate model with mean IoU metric."""
    model.eval()
    all_ious = []
    batch_count = 0
    total_batches = len(data_loader)
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images  = images.to(device)
            targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
            out     = model(images)
            pb      = out["pred_boxes"]  # (B, N, 4)
            
            for b, tgt in enumerate(targets):
                gt_boxes = tgt["boxes"]  # (T, 4)
                if gt_boxes.shape[0] == 0: continue
                
                pred_boxes = pb[b]  # (N, 4)
                # Compute IoU between all pred (N,) and all gt (T,)
                iou_mat, _ = box_iou(
                    box_cxcywh_to_xyxy(pred_boxes), 
                    box_cxcywh_to_xyxy(gt_boxes)
                )
                # For each GT, take max IoU across predictions
                best_iou_per_gt = iou_mat.max(0)[0]  # (T,)
                all_ious.append(best_iou_per_gt.mean().item())
            
            batch_count += 1
            
            # Log batch progress every 20 batches or at end
            if logger and (batch_idx % 20 == 0 or batch_idx == total_batches - 1):
                progress_pct = 100.0 * (batch_idx + 1) / total_batches
                current_mean_iou = sum(all_ious) / max(len(all_ious), 1)
                logger.info(f"    [Val]   Batch {batch_idx+1}/{total_batches} ({progress_pct:5.1f}%) | "
                           f"Mean IoU: {current_mean_iou:.6f}")
    
    return {"mean_iou": sum(all_ious) / max(len(all_ious), 1)}


def parse_args():
    """Parse command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train DFINE object detection model on COCO dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model architecture parameters
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--num_classes", type=int, default=80,
                            help="Number of object classes (80 for COCO)")
    model_group.add_argument("--num_queries", type=int, default=50,
                            help="Number of queries in DETR decoder")
    model_group.add_argument("--d_model", type=int, default=128,
                            help="Model embedding dimension")
    model_group.add_argument("--num_encoder_layers", type=int, default=2,
                            help="Number of transformer encoder layers")
    model_group.add_argument("--num_decoder_layers", type=int, default=3,
                            help="Number of transformer decoder layers")
    
    # Training hyperparameters
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--num_epochs", type=int, default=5,
                            help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=8,
                            help="Batch size for training")
    train_group.add_argument("--learning_rate", type=float, default=1e-4,
                            help="Initial learning rate")
    train_group.add_argument("--weight_decay", type=float, default=1e-4,
                            help="Weight decay for AdamW optimizer")
    train_group.add_argument("--num_workers", type=int, default=4,
                            help="Number of data loading workers")
    train_group.add_argument("--seed", type=int, default=42,
                            help="Random seed for reproducibility")
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--data_dir", type=str, default=None,
                           help="Path to dataset directory (default: ./data)")
    data_group.add_argument("--train_split", type=str, default="train2017",
                           help="Training dataset split name")
    data_group.add_argument("--val_split", type=str, default="val2017",
                           help="Validation dataset split name")
    
    # Device and output parameters
    device_group = parser.add_argument_group("Device & Output")
    device_group.add_argument("--device", type=str, default="cuda",
                             choices=["cuda", "cpu"],
                             help="Device to use for training")
    device_group.add_argument("--save_dir", type=str, default="checkpoints",
                             help="Directory to save model checkpoints")
    device_group.add_argument("--save_interval", type=int, default=1,
                             help="Save checkpoint every N epochs")
    device_group.add_argument("--no_save", action="store_true",
                             help="Don't save checkpoints during training")
    device_group.add_argument("--verbose", action="store_true",
                             help="Print detailed training progress")
    
    return parser.parse_args()


if __name__ == "__main__":
    import os
    
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    log_dir = Path(args.save_dir) / "logs" if not args.no_save else Path("./logs")
    logger, log_file = setup_logger(log_dir)
    
    logger.info("="*80)
    logger.info("DFINE Training Session Started")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Model configuration
    logger.info("\n--- Model Configuration ---")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Number of queries: {args.num_queries}")
    logger.info(f"Model dimension: {args.d_model}")
    logger.info(f"Encoder layers: {args.num_encoder_layers}")
    logger.info(f"Decoder layers: {args.num_decoder_layers}")
    
    # Create model components
    logger.info("\nCreating model...")
    model = DFINEMini(
        num_classes=args.num_classes, 
        num_queries=args.num_queries, 
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    ).to(device)
    
    matcher = HungarianMatcher()
    # Enable multi-layer supervision by passing matcher to criterion
    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_vfl=1.0,
        weight_bbox=5.0,
        weight_giou=2.0,
        weight_fgl=0.5,  # Enable FGL loss
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Training hyperparameters
    logger.info("\n--- Training Configuration ---")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Data workers: {args.num_workers}")
    
    # Build data loaders from real COCO dataset
    logger.info("\n--- Loading Dataset ---")
    if args.data_dir is None:
        data_dir = Path(__file__).parent / "data"
    else:
        data_dir = Path(args.data_dir)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Training split: {args.train_split}")
    logger.info(f"Validation split: {args.val_split}")
    
    train_dataset = COCODataset(str(data_dir), split=args.train_split)
    val_dataset = COCODataset(str(data_dir), split=args.val_split)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} images ({len(train_loader)} batches)")
    logger.info(f"Val dataset: {len(val_dataset)} images ({len(val_loader)} batches)")
    
    # Setup checkpoint directory
    if not args.no_save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {save_dir}")
    else:
        logger.info("Checkpoint saving disabled (--no_save)")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    logger.info("="*80 + "\n")
    
    best_iou = 0.0
    best_epoch = 0
    
    for epoch in range(args.num_epochs):
        epoch_start = datetime.now()
        
        logger.info(f"Epoch [{epoch+1}/{args.num_epochs}]")
        
        # Training
        train_metrics = train_one_epoch(model, criterion, matcher, optimizer, train_loader, device, logger)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device, logger)
        
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        
        # Log metrics
        logger.info(f"  Train - VFL Loss: {train_metrics['avg_loss_vfl']:.6f}, "
                   f"BBox Loss: {train_metrics['avg_loss_bbox']:.6f}, "
                   f"GIoU Loss: {train_metrics['avg_loss_giou']:.6f}")
        logger.info(f"  Val   - Mean IoU: {val_metrics['mean_iou']:.6f}")
        logger.info(f"  Time: {epoch_time:.2f}s")
        
        # Track best model
        if val_metrics['mean_iou'] > best_iou:
            best_iou = val_metrics['mean_iou']
            best_epoch = epoch + 1
            logger.info(f"  *** New best IoU: {best_iou:.6f} ***")
        
        # Save checkpoint
        if not args.no_save and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = Path(args.save_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_metrics['mean_iou'],
                'config': {
                    'num_classes': args.num_classes,
                    'num_queries': args.num_queries,
                    'd_model': args.d_model,
                    'num_encoder_layers': args.num_encoder_layers,
                    'num_decoder_layers': args.num_decoder_layers,
                }
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path.name}")
        
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("Training Complete!")
    logger.info(f"Best validation IoU: {best_iou:.6f} (Epoch {best_epoch})")
    logger.info(f"Total epochs: {args.num_epochs}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
