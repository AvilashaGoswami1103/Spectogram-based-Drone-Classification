import albumentations as A
import cv2
import os
import shutil

# ── PATHS ─────────────────────────────────────────────────────────
dataset_path = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_new"
output_path  = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_augmented"

# ── AUGMENTATION PIPELINE ─────────────────────────────────────────
# Only physically meaningful augmentations for RF spectrograms
# NO flips, NO rotations — they are meaningless for frequency data
methods = [
    [A.GaussNoise(var_limit=(100, 500), mean=0, p=1)],
    [A.AdvancedBlur(blur_limit=(7, 13), sigma_x_limit=(7, 13),
                    sigma_y_limit=(7, 13), rotate_limit=(-90, 90),
                    beta_limit=(0.5, 8), noise_limit=(2, 10), p=1)],
    [A.CLAHE(clip_limit=3, tile_grid_size=(13, 13), p=1)],
    [A.ISONoise(intensity=(0.2, 0.5), color_shift=(0.01, 0.05), p=1)],
    [A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1), p=1)],
    [A.ColorJitter(brightness=(0.5, 1.5), contrast=(1, 1),
                   saturation=(1, 1), hue=(0, 0), p=1)],
]

# ── PROCESS TRAIN AND VAL ─────────────────────────────────────────
for split in ['train', 'val']:
    split_path = os.path.join(dataset_path, split)

    if not os.path.exists(split_path):
        print(f"⚠ Skipping '{split}' — folder not found")
        continue

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue

        # ── New structure: images sit directly in class folder ────
        # No imgs/ subfolder needed
        save_dir = os.path.join(output_path, split, class_name)
        os.makedirs(save_dir, exist_ok=True)

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.png'))]

        if not images:
            print(f"⚠ No images found in {class_path}")
            continue

        for image_file in images:
            img_path = os.path.join(class_path, image_file)
            original = cv2.imread(img_path)

            if original is None:
                print(f"⚠ Could not read {img_path}, skipping")
                continue

            # Always copy the original
            cv2.imwrite(os.path.join(save_dir, image_file), original)

            # Only augment train, copy val as-is
            if split == 'train':
                for i, method in enumerate(methods):
                    transform = A.Compose(method)
                    augmented = transform(image=original)['image']
                    aug_name  = f"{os.path.splitext(image_file)[0]}_aug{i}.jpg"
                    cv2.imwrite(os.path.join(save_dir, aug_name), augmented)

        total = len(os.listdir(save_dir))
        print(f"  {split}/{class_name}: {total} images saved  "
              f"({'original + 6 aug each' if split == 'train' else 'original only'})")

# ── SUMMARY ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("Augmentation complete →", output_path)
print("\nFinal image counts:")
for split in ['train', 'val']:
    for class_name in sorted(os.listdir(os.path.join(output_path, split))):
        d = os.path.join(output_path, split, class_name)
        if os.path.isdir(d):
            n = len(os.listdir(d))
            print(f"  {split}/{class_name}: {n} images")




## What this produces
'''
With 50 images per IQ file and 80/20 split:
```
dataset_new_augmented/
├── train/
│   ├── DEVENTION/  ← ~240 orig × 7 = ~1,680 images
│   ├── FATUBA/     ← ~280 orig × 7 = ~1,960 images
│   ├── FLYSKY/     ← ~240 orig × 7 = ~1,680 images
│   └── YUNZHOU/    ← ~160 orig × 7 = ~1,120 images
└── val/
    ├── DEVENTION/  ← ~60 images (originals only)
    ├── FATUBA/     ← ~70 images
    ├── FLYSKY/     ← ~60 images
    └── YUNZHOU/    ← ~40 images
'''