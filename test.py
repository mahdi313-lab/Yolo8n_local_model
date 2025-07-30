import torch
import numpy as np
# --------
from model import YOLOv8_aircraft
from loss import v8DetectionLoss

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory Allocated: {allocated:.3f} GB")
        print(f"GPU Memory Reserved: {reserved:.3f} GB")
    else:
        print("CUDA GPU not available.")

def create_dummy_batch(batch_size=4, img_size=1024, num_classes=1, max_objs=10, device='cuda'):  # 🔴 Changed img_size=512 to 1024
    """
    Create dummy input images and targets batch for testing.
    - Inputs: random float tensors normalized [0,1]
    - Targets: (batch_idx, class, x_center, y_center, w, h), normalized coordinates [0,1]
    """
    imgs = torch.rand(batch_size, 3, img_size, img_size, device=device)

    batch_idx = []
    classes = []
    bboxes = []
    for i in range(batch_size):
        n_objs = np.random.randint(1, max_objs+1)
        batch_idx.append(torch.full((n_objs, 1), i, device=device))
        classes.append(torch.zeros((n_objs, 1), dtype=torch.float32, device=device))  # class 0 always (single class)
        # Random bboxes: x_center, y_center, w, h between 0.1 and 0.8 for stability
        boxes = 0.1 + 0.7 * torch.rand(n_objs, 4, device=device)
        # Ensure valid boxes: clamp w, h to avoid negative or zero dimensions
        boxes[:, 2:] = torch.clamp(boxes[:, 2:], min=1e-3)  # 🔴 Added to ensure valid w, h
        bboxes.append(boxes)

    batch_idx = torch.cat(batch_idx, 0)
    classes = torch.cat(classes, 0)
    bboxes = torch.cat(bboxes, 0)

    targets = torch.cat([batch_idx, classes, bboxes], 1)
    return imgs, targets

def test_model_forward():
    print("Testing model forward pass...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLOv8_aircraft(nc=1, reg_max=16).to(device)
    model.eval()

    imgs, targets = create_dummy_batch(batch_size=2, img_size=1024, device=device)  # 🔴 Changed img_size=512 to 1024
    print(f"Input batch shape: {imgs.shape}")
    with torch.no_grad():
        outputs = model(imgs)

    print(f"Model outputs: {len(outputs)} feature maps")
    for i, out in enumerate(outputs):
        print(f"  Output {i} shape: {out.shape}")
    print("Model forward pass succeeded.\n")
    return model, imgs, targets

def test_loss_function(model, imgs, targets):
    print("Testing loss function...")
    device = next(model.parameters()).device
    loss_fn = v8DetectionLoss(model, tal_topk=10)

    # Prepare batch dictionary as expected by loss
    batch = {
        "batch_idx": targets[:, 0].long(),  # 🔴 Ensured long type
        "cls": targets[:, 1].float(),  # 🔴 Changed to float for v8DetectionLoss
        "bboxes": targets[:, 2:].float()  # Already float, ensured consistency
    }

    model.eval()
    with torch.no_grad():
        preds = model(imgs)
        loss, loss_items = loss_fn(preds, batch)  # 🔴 Clarified return values

    print(f"Loss components (box, cls, dfl): {loss_items.cpu().numpy()}")
    print(f"Total loss: {loss.item():.4f}")
    print("Loss computation succeeded.\n")
    return loss, loss_items  # 🔴 Return both for clarity

def main():
    print("==== Starting full test script ====\n")
    print_gpu_memory()

    model, imgs, targets = test_model_forward()
    print_gpu_memory()

    loss, loss_items = test_loss_function(model, imgs, targets)  # 🔴 Updated to receive both loss and loss_items
    print_gpu_memory()

    print("\n==== All tests completed successfully! ====")

if __name__ == "__main__":
    main()
