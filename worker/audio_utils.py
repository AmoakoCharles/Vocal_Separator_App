import os
import shutil
import tempfile
import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

# Paste your existing functions here:
# - gentle_bandpass_filter
# - separate_vocals
# - process_song

# For example:

def gentle_bandpass_filter(y, sr, lowcut, highcut, boost_amount=1.5):
    sos = butter(3, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, y)
    mixed = (filtered * boost_amount + y * 0.3) / (boost_amount + 0.3)
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed * 0.95 / max_val
    return mixed


def separate_vocals(audio_path):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    temp_dir = tempfile.mkdtemp(prefix="demucs_")
    audio, sr = librosa.load(audio_path, sr=44100, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    audio_tensor = torch.from_numpy(audio).float()
    if audio_tensor.ndim == 2:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    model = get_model('htdemucs')
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, device=device)
    
    vocals_idx = model.sources.index('vocals')
    vocals = sources[0, vocals_idx].cpu().numpy()
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_dir = os.path.join(temp_dir, "htdemucs", base_name)
    os.makedirs(vocals_dir, exist_ok=True)
    
    vocals_path = os.path.join(vocals_dir, "vocals.wav")
    if vocals.ndim == 2 and vocals.shape[0] < vocals.shape[1]:
        vocals = vocals.T
    sf.write(vocals_path, vocals, sr)
    
    return vocals_path, temp_dir


def process_song(audio_path, output_folder):
    """
    Process an audio file to create vocal stems.
    
    Args:
        audio_path: Path to input audio file
        output_folder: Folder to save processed stems
    
    Returns:
        tuple: (dict of output files, status string)
    """
    output_files = {}

    try:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # Step 1: Separate vocals
        print("\n[1/4] Separating vocals from music...")
        vocals_path, temp_dir = separate_vocals(audio_path)

        # Step 2: Load vocals
        print("[2/4] Loading separated vocals...")
        y_vocals, sr = librosa.load(vocals_path, sr=None, mono=True)
        print(f"  Sample rate: {sr} Hz")
        print(f"  Duration: {len(y_vocals)/sr:.2f} seconds")

        # Step 3: Define frequency ranges for vocal filtering
        print("[3/4] Creating frequency-filtered vocal stems...")
        freq_ranges = {
            "soprano_range": (250, 2500),
            "alto_range":    (180, 1800),
            "tenor_range":   (120, 1500),
            "lead_vocals":   (150, 2000),
            "full_vocals":   None
        }

        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Step 4: Create filtered stems
        for part, freq_range in freq_ranges.items():
            if freq_range is None:
                # Save unfiltered vocals
                print(f"  Creating {part} (unfiltered)...")
                output_path = os.path.join(output_folder, f"{base_name}_{part}.wav")
                sf.write(output_path, y_vocals, sr)
                output_files[part] = output_path
                print(f"    ✓ Saved: {os.path.basename(output_path)}")
            else:
                low, high = freq_range
                print(f"  Creating {part} ({low}-{high} Hz)...")
                filtered = gentle_bandpass_filter(y_vocals, sr, low, high, boost_amount=2.0)
                output_path = os.path.join(output_folder, f"{base_name}_{part}.wav")
                sf.write(output_path, filtered, sr)
                output_files[part] = output_path
                print(f"    ✓ Saved: {os.path.basename(output_path)}")

        # Cleanup temporary files
        print("[4/4] Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("  ✓ Cleanup complete")

        print(f"\n{'='*60}")
        print(f"✓ Processing completed successfully!")
        print(f"  Created {len(output_files)} vocal stems")
        print(f"{'='*60}\n")

        return output_files, "success"

    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        return None, str(e)