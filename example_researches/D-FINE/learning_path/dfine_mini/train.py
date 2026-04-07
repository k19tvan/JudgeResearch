"""Problem 12: Training and Evaluation Loop"""
import torch
import torch.nn as nn
from typing import List

from model import DFINEMini
from matcher import HungarianMatcher
from criterion import SetCriterion
from box_ops import box_cxcywh_to_xyxy
from iou import box_iou


def make_synthetic_batch(B: int, num_gt: int, num_classes: int = 10,
                          device: torch.device = torch.device("cpu")):
    """Generate synthetic batch of images and targets for testing."""
    images = torch.randn(B, 3, 256, 256, device=device)
    targets = []
    for _ in range(B):
        cx = torch.rand(num_gt, device=device) * 0.6 + 0.2
        cy = torch.rand(num_gt, device=device) * 0.6 + 0.2
        w  = torch.rand(num_gt, device=device) * 0.25 + 0.05
        h  = torch.rand(num_gt, device=device) * 0.25 + 0.05
        # Clamp so boxes stay in [0,1]
        cx = cx.clamp(w/2+0.01, 1-w/2-0.01)
        cy = cy.clamp(h/2+0.01, 1-h/2-0.01)
        boxes  = torch.stack([cx, cy, w, h], -1)
        labels = torch.randint(0, num_classes, (num_gt,), device=device)
        targets.append({"labels": labels, "boxes": boxes})
    return images, targets


def train_one_epoch(model, criterion, matcher, optimizer, data_loader, device):
    """Run one full training epoch."""
    model.train()
    totals = {"loss_vfl": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
    
    for images, targets in data_loader:
        images  = images.to(device)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        out = model(images)
        indices = matcher(out, targets)["indices"]
        losses  = criterion(out, targets, indices)
        total   = sum(losses.values())
        total.backward()
        optimizer.step()
        
        for k in totals:
            totals[k] += losses[k].item()
    
    n = max(len(data_loader), 1)
    return {f"avg_{k}": v / n for k, v in totals.items()}


def evaluate(model, data_loader, device):
    """Evaluate model with mean IoU metric."""
    model.eval()
    all_ious = []
    
    with torch.no_grad():
        for images, targets in data_loader:
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
    
    return {"mean_iou": sum(all_ious) / max(len(all_ious), 1)}


if __name__ == "__main__":
    import sys
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 10
    NUM_QUERIES = 50
    D_MODEL = 128
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 3
    BATCH_SIZE = 4
    NUM_EPOCHS = 5
    NUM_TRAIN_BATCHES = 20
    NUM_VAL_BATCHES = 5
    
    print(f"Device: {device}")
    print(f"Creating model with {NUM_CLASSES} classes, {NUM_QUERIES} queries, d_model={D_MODEL}")
    
    # Create model components
    model     = DFINEMini(
        num_classes=NUM_CLASSES, 
        num_queries=NUM_QUERIES, 
        d_model=D_MODEL,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS
    ).to(device)
    
    matcher   = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Build data loaders
    print(f"Building synthetic data loaders...")
    train_loader = [make_synthetic_batch(BATCH_SIZE, 3, NUM_CLASSES, device) 
                    for _ in range(NUM_TRAIN_BATCHES)]
    val_loader   = [make_synthetic_batch(BATCH_SIZE, 3, NUM_CLASSES, device) 
                    for _ in range(NUM_VAL_BATCHES)]
    
    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_one_epoch(model, criterion, matcher, optimizer, train_loader, device)
        val_metrics   = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train: avg_loss_vfl={train_metrics['avg_loss_vfl']:.4f}, "
              f"avg_loss_bbox={train_metrics['avg_loss_bbox']:.4f}, "
              f"avg_loss_giou={train_metrics['avg_loss_giou']:.4f}")
        print(f"  Val:   mean_iou={val_metrics['mean_iou']:.4f}")
    
    print("\nTraining complete!")
