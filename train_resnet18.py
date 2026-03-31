import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import time

# ── PATHS ──────────────────────────────────────────────────────────
TRAIN_PATH   = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_augmented\train"
VAL_PATH     = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_augmented\val"
SAVE_PATH    = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\weights_resnet18"
os.makedirs(SAVE_PATH, exist_ok=True)

# ── HYPERPARAMETERS ────────────────────────────────────────────────
MODEL_NAME   = 'resnet18'                              # ← changed
NUM_CLASSES  = 4
BATCH_SIZE   = 16
IMAGE_SIZE   = 224
LR           = 0.0001
NUM_EPOCHS   = 5
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── TRANSFORMS ─────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── DATASETS & DATALOADERS ─────────────────────────────────────────
train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)
val_dataset   = datasets.ImageFolder(VAL_PATH,   transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

print(f"Classes: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

# ── MODEL ──────────────────────────────────────────────────────────
model    = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ← changed
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)              # 512 → 4 for ResNet18
model    = model.to(DEVICE)

# ── LOSS & OPTIMIZER ───────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=7, gamma=0.1)

# ── LOG FILE ───────────────────────────────────────────────────────
log_path = os.path.join(SAVE_PATH, 'train.log')
log_file = open(log_path, 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log(f"Model: {MODEL_NAME}")
log(f"Classes: {train_dataset.classes}")
log(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
log(f"Epochs: {NUM_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
log("="*60)

# ── TRAINING LOOP ──────────────────────────────────────────────────
best_val_acc = 0.0
best_epoch   = 0

for epoch in range(1, NUM_EPOCHS + 1):
    start = time.time()

    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * images.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    train_loss /= train_total
    train_acc   = train_correct / train_total * 100

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            val_loss   += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_loss /= val_total
    val_acc   = val_correct / val_total * 100
    elapsed   = time.time() - start

    log(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
        f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  "
        f"Time: {elapsed:.1f}s")

    epoch_path = os.path.join(SAVE_PATH, f'{MODEL_NAME}_epoch_{epoch}.pth')
    torch.save(model.state_dict(), epoch_path)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        best_path    = os.path.join(SAVE_PATH, 'best_model.pth')
        torch.save(model.state_dict(), best_path)
        log(f" **New best model saved (Val Acc: {best_val_acc:.2f}%)")

    scheduler.step()

log("="*60)
log(f"Training complete. Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
log(f"Best model saved to: {best_path}")
log_file.close()