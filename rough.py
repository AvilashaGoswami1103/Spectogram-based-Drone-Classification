import numpy as np
import os

DATASETS = r"C:\Users\Avilasha\Desktop\Online-data\Datasets"

# Try different window sizes and see how many images each would produce
window_options = [50000, 100000, 500000, 1000000]

# First show file sizes
print("File sizes:")
print(f"{'File':<35} {'IQ samples':>15}")
print("-" * 52)

file_sizes = {}
for root, dirs, files in os.walk(DATASETS):
    for fname in sorted(files):
        if not fname.endswith('.iq'):
            continue
        fpath = os.path.join(root, fname)
        raw   = np.fromfile(fpath, dtype=np.float32)
        n_iq  = len(raw) // 2
        file_sizes[fname] = n_iq
        print(f"{fname:<35} {n_iq:>15,}")

print("\n\nImages per file at different window sizes:")
print(f"{'File':<35}", end='')
for w in window_options:
    print(f"  w={w//1000}k", end='')
print()
print("-" * 80)

for fname, n_iq in file_sizes.items():
    print(f"{fname:<35}", end='')
    for w in window_options:
        n_img = n_iq // w
        print(f"  {n_img:>6}", end='')
    print()