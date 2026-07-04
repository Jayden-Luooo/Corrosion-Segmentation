from pathlib import Path

import torch

import os
import random
import numpy as np

def set_seed(seed=42, deterministic=True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    
def dice_score(logits, masks, threshold=0.58, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten(start_dim=1)
    masks = masks.flatten(start_dim=1)

    intersection = (preds * masks).sum(dim=1)
    union = preds.sum(dim=1) + masks.sum(dim=1)

    return ((2.0 * intersection + eps) / (union + eps)).mean().item()


def iou_recall_score(logits, masks, threshold=0.58, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.flatten()
    masks = masks.flatten()

    tp = (preds * masks).sum()
    fp = (preds * (1 - masks)).sum()
    fn = ((1 - preds) * masks).sum()

    iou = (tp + eps) / (tp + fp + fn + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return iou.item(), recall.item()


def train_epoch(
    model,
    train_loader,
    loss_fn,
    optimizer,
    device,
    scaler=None,
    use_amp=False,
    metric_threshold=0.58,
):
    model.train()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_recall = 0.0

    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, masks)

        optimizer.zero_grad(set_to_none=True)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        dice = dice_score(logits.detach(), masks, threshold=metric_threshold)
        iou, recall = iou_recall_score(logits.detach(), masks, threshold=metric_threshold)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou
        total_recall += recall

    return {
        "loss": total_loss / len(train_loader),
        "dice": total_dice / len(train_loader),
        "iou": total_iou / len(train_loader),
        "recall": total_recall / len(train_loader),
    }


def validate(model, val_loader, loss_fn, device, use_amp=False, metric_threshold=0.58):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_recall = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, masks)

            dice = dice_score(logits, masks, threshold=metric_threshold)
            iou, recall = iou_recall_score(logits, masks, threshold=metric_threshold)
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_recall += recall

    return {
        "loss": total_loss / len(val_loader),
        "dice": total_dice / len(val_loader),
        "iou": total_iou / len(val_loader),
        "recall": total_recall / len(val_loader),
    }


def train_model(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    device,
    num_epochs,
    save_path,
    latest_path=None,
    scheduler=None,
    use_amp=True,
    metric_threshold=0.58,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if latest_path is None:
        latest_path = save_path.with_name("latest_unet.pth")
    latest_path = Path(latest_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "train_dice": [],
        "train_iou": [],
        "train_recall": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_recall": [],
    }
    
    
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_dice = 0.0

    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            metric_threshold=metric_threshold,
        )
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            metric_threshold=metric_threshold,
        )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])

        history["train_loss"].append(train_metrics["loss"])
        history["train_dice"].append(train_metrics["dice"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_recall"].append(val_metrics["recall"])

        is_best = False

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), save_path)
            is_best = True

        torch.save(model.state_dict(), latest_path)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Dice: {train_metrics['dice']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Train Recall: {train_metrics['recall']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
            + ("  Best" if is_best else "")
        )

    return history
