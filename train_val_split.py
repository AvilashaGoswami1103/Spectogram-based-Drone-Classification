import os, glob, shutil, random

# ── Your new spectrogram output folder ────────────────────────────
base        = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_matlab_v2"
dataset_dir = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_new"

# ── Map class names to source IQ file prefixes ────────────────────
# These match the subfolders created by organize_datasets.py
# OR just the flat files if you didn't organize into subfolders
classes = {
    "DEVENTION": ["DEVENTION_pack1_0-1s", "DEVENTION_pack1_1-2s",
                  "DEVENTION_pack1_2-3s", "DEVENTION_pack1_3-4s",
                  "DEVENTION_pack1_4-5s", "DEVENTION_pack1_5-6s"],
    "FATUBA":    ["FATUBA_pack1_0-1s",    "FATUBA_pack1_1-2s",
                  "FATUBA_pack1_2-3s",    "FATUBA_pack1_3-4s",
                  "FATUBA_pack1_4-5s",    "FATUBA_pack1_5-6s",
                  "FATUBA_pack1_6-7s"],
    "FLYSKY":    ["FLYSKY_pack1_0-1s",    "FLYSKY_pack1_1-2s",
                  "FLYSKY_pack1_2-3s",    "FLYSKY_pack1_3-4s",
                  "FLYSKY_pack1_4-5s",    "FLYSKY_pack1_5-6s"],
    "YUNZHOU":   ["YUNZHOU_pack1_0-1s",   "YUNZHOU_pack1_1-2s",
                  "YUNZHOU_pack2_0-1s",   "YUNZHOU_pack2_1-2s"],
}

random.seed(42)  # reproducible split

for class_name, source_prefixes in classes.items():

    # ── Output folders — no 'imgs' subfolder needed ───────────────
    train_dir = os.path.join(dataset_dir, "train", class_name)
    val_dir   = os.path.join(dataset_dir, "val",   class_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir,   exist_ok=True)

    # ── Collect all images for this class ─────────────────────────
    # New structure: dataset_matlab_v2/FLYSKY/FLYSKY_pack1_0-1s_w000.jpg
    # OR:           dataset_matlab_v2/FLYSKY_pack1_0-1s_w000.jpg (flat)
    all_images = []

    # Try class subfolder first (if you ran organize_datasets.py)
    class_folder = os.path.join(base, class_name)
    if os.path.exists(class_folder):
        imgs = glob.glob(os.path.join(class_folder, "*.jpg")) + \
               glob.glob(os.path.join(class_folder, "*.png"))
        all_images.extend(imgs)
        print(f"  [{class_name}] Found {len(imgs)} images in class subfolder")

    else:
        # Flat structure — match by filename prefix
        for prefix in source_prefixes:
            imgs = glob.glob(os.path.join(base, f"{prefix}*.jpg")) + \
                   glob.glob(os.path.join(base, f"{prefix}*.png"))
            all_images.extend(imgs)
            print(f"  [{class_name}] {prefix}: {len(imgs)} images")

    if not all_images:
        print(f"  ⚠ No images found for {class_name} — check base path")
        continue

    # ── 80/20 split ───────────────────────────────────────────────
    random.shuffle(all_images)
    split      = int(0.8 * len(all_images))
    train_imgs = all_images[:split]
    val_imgs   = all_images[split:]

    for f in train_imgs:
        shutil.copy(f, train_dir)
    for f in val_imgs:
        shutil.copy(f, val_dir)

    print(f"  → {class_name}: Train={len(train_imgs)} | Val={len(val_imgs)}\n")

print("="*50)
print("Split complete →", dataset_dir)
print("Structure:")
for split in ['train', 'val']:
    for cls in classes:
        d = os.path.join(dataset_dir, split, cls)
        if os.path.exists(d):
            n = len(os.listdir(d))
            print(f"  {split}/{cls}: {n} images")