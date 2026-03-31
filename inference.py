import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import csv

# ══════════════════════════════════════════════════════════════════
# CONFIGURE HERE — change MODEL to switch between resnet50/resnet18
# ══════════════════════════════════════════════════════════════════
MODEL        = 'resnet18'   # ← change to 'resnet18' for ResNet-18

WEIGHTS_PATH = r"C:\Users\Avilasha\Desktop\Image-based Drone Classification\Online-data\New_Attempt\weights_resnet18\best_model.pth"
# For ResNet-18 use:
# WEIGHTS_PATH = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\weights_resnet18\best_model.pth"

TEST_DIR     = r"C:\Users\Avilasha\Desktop\Image-based Drone Classification\Online-data\New_Attempt\test_spectograms_v2"
RESULTS_DIR  = r"C:\Users\Avilasha\Desktop\Image-based Drone Classification\Online-data\New_Attempt\test_results_new"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
CLASS_NAMES  = ['DEVENTION', 'FATUBA', 'FLYSKY', 'YUNZHOU']
NUM_CLASSES  = 4
DEVICE       = torch.device('cpu')
# ══════════════════════════════════════════════════════════════════

# ── Load model ────────────────────────────────────────────────────
if MODEL == 'resnet50':
    model = models.resnet50(weights=None)
elif MODEL == 'resnet18':
    model = models.resnet18(weights=None)
else:
    raise ValueError(f"Unknown model: {MODEL}")

model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()
print(f"Model:   {MODEL}")
print(f"Weights: {WEIGHTS_PATH}")
print(f"Classes: {CLASS_NAMES}\n")

# ── Transform — must match training exactly ───────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ── Collect all test images ───────────────────────────────────────
# Handles both flat folder and subfolders
test_images = []
for root, dirs, files in os.walk(TEST_DIR):
    for fname in sorted(files):
        if fname.lower().endswith(('.jpg', '.png')):
            test_images.append(os.path.join(root, fname))

print(f"Found {len(test_images)} test images in: {TEST_DIR}\n")

# ── Run inference ─────────────────────────────────────────────────
results = []

for fpath in test_images:
    img    = Image.open(fpath).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()

    top_idx    = probs.argmax()
    top_class  = CLASS_NAMES[top_idx]
    top_conf   = probs[top_idx] * 100
    fname      = os.path.basename(fpath)

    # Print result
    print(f"{fname}")
    print(f"  Predicted : {top_class} ({top_conf:.2f}%)")
    for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
        bar    = '█' * int(p * 40)
        marker = ' ◄' if i == top_idx else ''
        print(f"  {name:<12} {p*100:5.2f}%  {bar}{marker}")
    print()

    results.append({
        'file'       : fname,
        'predicted'  : top_class,
        'confidence' : f"{top_conf:.2f}",
        'DEVENTION'  : f"{probs[0]*100:.2f}",
        'FATUBA'     : f"{probs[1]*100:.2f}",
        'FLYSKY'     : f"{probs[2]*100:.2f}",
        'YUNZHOU'    : f"{probs[3]*100:.2f}",
    })

# ── Save CSV with all probabilities ──────────────────────────────
csv_path = os.path.join(RESULTS_DIR, f'inference_{MODEL}.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# ── Summary ───────────────────────────────────────────────────────
print("=" * 55)
print(f"SUMMARY — {MODEL}")
print("=" * 55)
pred_counts = {c: 0 for c in CLASS_NAMES}
for r in results:
    pred_counts[r['predicted']] += 1

for name, count in pred_counts.items():
    pct = count / len(results) * 100
    bar = '█' * int(pct / 2)
    print(f"  {name:<12} {count:4d} images  ({pct:.1f}%)  {bar}")

print(f"\nTotal images tested : {len(results)}")
print(f"Results CSV saved  : {csv_path}")
print(f"\nDone — {MODEL} inference complete")