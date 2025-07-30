import os
import warnings
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler
from torchmetrics.detection import MeanAveragePrecision 
from torchvision import transforms  # <-- required for ToTensor
from utils import get_latest_checkpoint 
from dataloader import Yolo8Dataset
from model import YOLOv8n
from loss import v8DetectionLoss

# -------------------------------
# Global Configurations
# -------------------------------
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore")

# -------------------------------
# Custom collate function to handle variable targets per image
# -------------------------------
to_tensor = transforms.ToTensor()

def yolo8_collate_fn(batch):
    images = torch.stack([
        to_tensor(item[0]) if not isinstance(item[0], torch.Tensor) else item[0]
        for item in batch
    ])
    targets = [item[1] for item in batch]
    return images, targets

# -------------------------------
# Training Function
# -------------------------------
def train_model(
    train_images_dir,
    train_labels_dir,
    val_images_dir,
    val_labels_dir,
    num_epochs=100,
    batch_size=2,
    lr=0.001,
    weights_dir='weights'
):
    os.makedirs(weights_dir, exist_ok=True)
    last_weights_path, last_epoch = get_latest_checkpoint(weights_dir)
    start_epoch = int(last_epoch) if last_epoch is not None else 0

    last_scheduler_path = os.path.join(weights_dir, f"scheduler_state_{start_epoch}.pth")
    if not os.path.exists(last_scheduler_path):
        last_scheduler_path = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    writer = SummaryWriter()
    model = YOLOv8n(nc=1, reg_max=16).to(device)
    loss_fn = v8DetectionLoss(model)

    if last_weights_path:
        print(f"Loading weights from {last_weights_path}")
        checkpoint = torch.load(last_weights_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if last_scheduler_path:
        print(f"Loading scheduler state from {last_scheduler_path}")
        scheduler.load_state_dict(torch.load(last_scheduler_path, map_location=device))

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_dataset = Yolo8Dataset(train_images_dir, train_labels_dir)
    val_dataset = Yolo8Dataset(val_images_dir, val_labels_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=yolo8_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=yolo8_collate_fn
    )

    print(f"{'-'*30}\nBatch size: {batch_size}\n{'-'*30}")
    map_metric = MeanAveragePrecision(iou_type="bbox").to(device)  

    for epoch in range(start_epoch + 1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch} - LR: {current_lr:.8f}")
        model.train()
        epoch_losses = []
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Training [{epoch}/{num_epochs}]")):
            images = images.to(device, memory_format=torch.channels_last)
            batch_idx, cls_list, bbox_list = [], [], []

            for b_i, t in enumerate(targets):
                if not isinstance(t, torch.Tensor) or t.numel() == 0:
                    continue
                batch_idx.append(torch.full((t.shape[0], 1), b_i, dtype=torch.long, device=device))
                cls_list.append(t[:, 0:1].to(device))
                bbox_list.append(t[:, 1:].to(device))

            if len(batch_idx) > 0:
                batch_idx = torch.cat(batch_idx, dim=0)
                cls = torch.cat(cls_list, dim=0)
                bboxes = torch.cat(bbox_list, dim=0)
            else:
                batch_idx = torch.empty((0,1), dtype=torch.long, device=device)
                cls = torch.empty((0,1), dtype=torch.float, device=device)
                bboxes = torch.empty((0,4), dtype=torch.float, device=device)

            batch = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

            with torch.amp.autocast(device_type='cuda'):
                preds = model(images)
                loss, loss_items = loss_fn(preds, batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

        avg_train_loss = np.mean(epoch_losses)
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Validation
        model.eval()
        total_val_loss, total_acc, num_batches = 0.0, 0.0, 0 
        map_metric.reset()

        with torch.no_grad():
            for i, (images, targets) in enumerate(tqdm(val_loader, desc="Validating")):
                images = images.to(device, memory_format=torch.channels_last)
                batch_idx, cls_list, bbox_list = [], [], []

                for b_i, t in enumerate(targets):
                    if not isinstance(t, torch.Tensor) or t.numel() == 0:
                        continue
                    batch_idx.append(torch.full((t.shape[0], 1), b_i, dtype=torch.long, device=device))
                    cls_list.append(t[:, 0:1].to(device))
                    bbox_list.append(t[:, 1:].to(device))

                if len(batch_idx) > 0:
                    batch_idx = torch.cat(batch_idx, dim=0)
                    cls = torch.cat(cls_list, dim=0)
                    bboxes = torch.cat(bbox_list, dim=0)
                else:
                    batch_idx = torch.empty((0,1), dtype=torch.long, device=device)
                    cls = torch.empty((0,1), dtype=torch.float, device=device)
                    bboxes = torch.empty((0,4), dtype=torch.float, device=device)

                batch = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}
                preds = model(images)
                val_loss, _ = loss_fn(preds, batch)
                total_val_loss += val_loss.item()

                feats = preds[1] if isinstance(preds, tuple) else preds
                pred_scores = torch.cat([xi.view(feats[0].shape[0], loss_fn.no, -1) for xi in feats], 2)[:, -1, :].sigmoid()
                pred_bboxes = loss_fn.bbox_decode(loss_fn.assigner.anc_points, torch.cat([xi.view(feats[0].shape[0], loss_fn.no, -1) for xi in feats], 2)[:, :-1, :])
                pred_bboxes = pred_bboxes * 1024
                batch_preds = [
                    {
                        "boxes": pred_bboxes[b, pred_scores[b] > 0.5],
                        "scores": pred_scores[b, pred_scores[b] > 0.5],
                        "labels": torch.zeros_like(pred_scores[b, pred_scores[b] > 0.5], dtype=torch.int64)
                    } for b in range(images.size(0))
                ]
                batch_targets = [
                    {
                        "boxes": batch["bboxes"][batch["batch_idx"] == b] * 1024,
                        "labels": batch["cls"][batch["batch_idx"] == b].squeeze(-1).to(torch.int64)
                    } for b in range(images.size(0))
                ]
                map_metric.update(batch_preds, batch_targets)

                _, _, target_scores, fg_mask, _ = loss_fn.assigner(
                    pred_scores,
                    (loss_fn.bbox_decode(loss_fn.assigner.anc_points, feats[0][:, :-1, :]).detach() * loss_fn.assigner.stride_tensor).type(cls.dtype),
                    loss_fn.assigner.anc_points * loss_fn.assigner.stride_tensor,
                    cls.view(-1, 1).to(device),
                    bboxes.to(device),
                    torch.ones_like(cls).to(device)
                )
                if fg_mask.sum() > 0:
                    pred_labels = (pred_scores[fg_mask] > 0.5).float()
                    target_labels = target_scores[fg_mask, 0]
                    batch_acc = (pred_labels == target_labels).float().mean().item()
                else:
                    batch_acc = 0.0
                total_acc += batch_acc
                num_batches += 1

        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        avg_acc = total_acc / max(num_batches, 1)
        map_result = map_metric.compute()
        avg_map = map_result["map"].item()

        print(f"Validation Loss: {avg_val_loss:.4f} | mAP: {avg_map:.4f} | Accuracy: {avg_acc:.4f}") 
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('mAP/validation', avg_map, epoch)
        writer.add_scalar('Accuracy/validation', avg_acc, epoch)

        scheduler.step(avg_val_loss)

        if (epoch % 2 == 0) or (epoch == num_epochs):
            ckpt_path = os.path.join(weights_dir, f"yolov8_model_{epoch}.pth")
            sched_path = os.path.join(weights_dir, f"scheduler_state_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(scheduler.state_dict(), sched_path)
            print(f"Checkpoint saved: {ckpt_path}")
            print(f"Scheduler state saved: {sched_path}")

    writer.close()
    print("\nTraining complete!")
