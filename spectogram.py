import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import os

def save_spectrogram_matlab_style(iq_path, save_dir, sample_rate=100e6,
                                   winLen=1024, overlap=512, Nfft=1024):    # Window length (number of samples) used in the STFT (Short‑Time Fourier Transform).
    """
    Reads float32 IQ data, applies DC removal + normalization,
    computes STFT with 1024-point Hamming window, fftshift to center 0Hz,
    uses top 60dB dynamic range relative to signal max.
    Saves clean 224x224 image for CNN input.
    """
    # ── Load as float32 (same as repo) ────────────────────────────
    raw = np.fromfile(iq_path, dtype=np.float32)
    if len(raw) < 2:
        print(f"  Skipping {os.path.basename(iq_path)} — empty file")
        return

    I  = raw[0::2].astype(np.float64)
    Q  = raw[1::2].astype(np.float64)
    iq = I + 1j * Q

    # ── DC removal + normalize to ±1 ──────────────────────────────
    iq = iq - np.mean(iq)
    max_val = np.max(np.abs(iq))
    if max_val > 0:
        iq = iq / max_val

    # ── STFT with 1024-point Hamming window ───────────────────────
    window = np.hamming(winLen)
    f, t, Zxx = stft(iq, fs=1.0, window=window, nperseg=winLen,
                     noverlap=overlap, nfft=Nfft, return_onesided=False)

    # ── fftshift to center 0 Hz (MATLAB-style) ────────────────────
    Zxx = np.fft.fftshift(Zxx, axes=0)

    # ── Power in dB, top 60dB relative to signal max ──────────────
    SdB  = 20 * np.log10(np.abs(Zxx) + 1e-12)
    vmax = SdB.max()
    vmin = vmax - 60

    # ── Save clean 224x224 image — no axes, no colorbar ───────────
    os.makedirs(save_dir, exist_ok=True)
    fname     = os.path.splitext(os.path.basename(iq_path))[0] + '.jpg'
    save_path = os.path.join(save_dir, fname)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)  # → 224x224 px
    ax.pcolormesh(SdB, cmap='jet', vmin=vmin, vmax=vmax, shading='gouraud')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def process_all_iq_files(input_root, output_root,
                          winLen=1024, overlap=512, Nfft=1024,
                          samples_per_window=10000, step_samples=None, max_images_per_file = 50):
    """
    Walks input_root, finds all .iq files.
    Generates MULTIPLE spectrogram images per file using a sliding window.

    samples_per_window: how many IQ samples per spectrogram image
                        10000 samples @ 100MHz = 0.1ms window
                        increase to get more time coverage per image
    step_samples:       how many samples to advance between windows
                        None = non-overlapping (step = samples_per_window)
                        set to samples_per_window//2 for 50% overlap between images
    """
    if step_samples is None:
        step_samples = samples_per_window  # non-overlapping by default

    total_files  = 0
    total_images = 0

    for root, dirs, files in os.walk(input_root):
        iq_files = [f for f in sorted(files) if f.endswith('.iq')]
        if not iq_files:
            continue

        rel_path = os.path.relpath(root, input_root)
        print(f"\nProcessing folder: {rel_path}  ({len(iq_files)} IQ files)")

        for fname in iq_files:
            fpath = os.path.join(root, fname)
            raw   = np.fromfile(fpath, dtype=np.float32)
            n_iq_samples = len(raw) // 2  # each IQ pair = 2 floats

            print(f"  {fname}: {n_iq_samples} IQ samples", end='')

            # ── Sliding window across the file ────────────────────
            img_count = 0
            window_start = 0
            while window_start + samples_per_window <= n_iq_samples:
                # Extract window
                if img_count >= max_images_per_file:   # ← add this check
                    break
                start_float = window_start * 2
                end_float   = (window_start + samples_per_window) * 2
                chunk = raw[start_float:end_float]

                I  = chunk[0::2].astype(np.float64)
                Q  = chunk[1::2].astype(np.float64)
                iq = I + 1j * Q

                # DC removal + normalize
                iq = iq - np.mean(iq)
                max_val = np.max(np.abs(iq))
                if max_val > 0:
                    iq = iq / max_val

                # STFT
                window_fn = np.hamming(winLen)
                _, _, Zxx = stft(iq, fs=1.0, window=window_fn,
                                 nperseg=winLen, noverlap=overlap,
                                 nfft=Nfft, return_onesided=False)

                # fftshift
                Zxx = np.fft.fftshift(Zxx, axes=0)

                # dB + dynamic range
                SdB  = 20 * np.log10(np.abs(Zxx) + 1e-12)
                vmax = SdB.max()
                vmin = vmax - 60

                # Save
                rel_save = os.path.relpath(root, input_root)
                save_dir  = os.path.join(output_root, rel_save)
                os.makedirs(save_dir, exist_ok=True)

                base      = os.path.splitext(fname)[0]
                img_name  = f"{base}_w{img_count:03d}.jpg"
                save_path = os.path.join(save_dir, img_name)

                fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
                ax.pcolormesh(SdB, cmap='jet', vmin=vmin, vmax=vmax,
                              shading='gouraud')
                ax.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(save_path, dpi=100,
                            bbox_inches='tight', pad_inches=0)
                plt.close()

                img_count    += 1
                window_start += step_samples

            print(f"  →  {img_count} images saved")
            total_files  += 1
            total_images += img_count

    print(f"\n{'='*50}")
    print(f"Done — {total_files} IQ files → {total_images} spectrogram images")
    print(f"{'='*50}")


# ── RUN ───────────────────────────────────────────────────────────────────────

# Training data — organized class subfolders

process_all_iq_files(
    input_root       = r"C:\Users\Avilasha\Desktop\Online-data\Datasets",
    output_root      = r"C:\Users\Avilasha\Desktop\Online-data\New_Attempt\dataset_matlab_v2",
    winLen           = 1024,
    overlap          = 512,
    Nfft             = 1024,
    samples_per_window = 50000,   # adjust: more = wider time coverage per image
    step_samples       = 50000,    # non-overlapping windows
    max_images_per_file = 50
)

# Test data 
process_all_iq_files(
    input_root       = r"C:\Users\Avilasha\Desktop\Image-based Drone Classification\Online-data\New_Attempt\test_iq",
    output_root      = r"C:\Users\Avilasha\Desktop\Image-based Drone Classification\Online-data\New_Attempt\test_spectograms_v2",
    winLen           = 1024,
    overlap          = 512,
    Nfft             = 1024,
    samples_per_window = 1000000,
    step_samples       = 1000000,
    max_images_per_file = 50
)