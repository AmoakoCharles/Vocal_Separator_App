import os
import sys

# Force soundfile backend BEFORE any imports
os.environ["TORCHAUDIO_USE_SOUNDFILE"] = "1"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"

def patch_torchaudio():
    """
    Patch torchaudio to use soundfile backend and avoid torchcodec completely.
    """
    try:
        import soundfile as sf
        import numpy as np
        
        # Patch torchaudio.save before it's imported anywhere
        import torchaudio
        
        original_save = torchaudio.save
        
        def patched_torchaudio_save(filepath, src, sample_rate, **kwargs):
            """
            Replacement for torchaudio.save that uses soundfile.
            
            Args:
                filepath: Output file path
                src: Audio tensor (torch.Tensor)
                sample_rate: Sample rate in Hz
                **kwargs: Additional arguments (ignored)
            """
            # Convert torch tensor to numpy
            if hasattr(src, 'numpy'):
                audio_np = src.numpy()
            elif hasattr(src, 'cpu'):
                audio_np = src.cpu().numpy()
            else:
                audio_np = np.array(src)
            
            # Handle shape: torchaudio expects (channels, samples)
            # soundfile expects (samples, channels)
            if audio_np.ndim == 2 and audio_np.shape[0] <= audio_np.shape[1]:
                # Likely (channels, samples) - transpose
                audio_np = audio_np.T
            elif audio_np.ndim == 1:
                # Mono - reshape to (samples, 1)
                audio_np = audio_np.reshape(-1, 1)
            
            # Ensure float32
            if audio_np.dtype != np.float32:
                audio_np = audio_np.astype(np.float32)
            
            # Save with soundfile
            sf.write(str(filepath), audio_np, sample_rate, subtype='PCM_16')
        
        # Replace the function
        torchaudio.save = patched_torchaudio_save
        
        print("✓ Patched torchaudio.save to use soundfile")
        return True
        
    except ImportError as e:
        print(f"✗ Could not patch torchaudio: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error patching torchaudio: {e}")
        return False


def patch_demucs_audio():
    """
    Monkey-patch Demucs to use soundfile instead of torchaudio.
    This bypasses the torchcodec dependency issues on Windows.
    """
    try:
        import soundfile as sf
        import torch
        import numpy as np
        
        # We need to patch BEFORE demucs.audio is imported
        import demucs.audio
        
        # Save original function reference
        original_save = demucs.audio.save_audio
        
        def patched_save_audio(wav, path, samplerate=44100, **kwargs):
            """
            Replacement save function using soundfile instead of torchaudio.
            
            Args:
                wav: Audio tensor (torch.Tensor or numpy array)
                path: Output file path
                samplerate: Sample rate in Hz
                **kwargs: Additional arguments (ignored for soundfile)
            """
            # Convert torch tensor to numpy if needed
            if isinstance(wav, torch.Tensor):
                wav_np = wav.detach().cpu().numpy()
            else:
                wav_np = np.array(wav)
            
            # Handle shape conversion
            # Demucs outputs (channels, samples), soundfile expects (samples, channels)
            if wav_np.ndim == 2:
                if wav_np.shape[0] < wav_np.shape[1]:
                    # Likely (channels, samples) - transpose to (samples, channels)
                    wav_np = wav_np.T
            elif wav_np.ndim == 1:
                # Mono audio - reshape to (samples, 1)
                wav_np = wav_np.reshape(-1, 1)
            
            # Ensure float32 for soundfile
            if wav_np.dtype != np.float32:
                wav_np = wav_np.astype(np.float32)
            
            # Save using soundfile (handles WAV format natively)
            sf.write(str(path), wav_np, samplerate, subtype='PCM_16')
            
        # Replace the save_audio function
        demucs.audio.save_audio = patched_save_audio
        
        print("✓ Patched demucs.audio.save_audio to use soundfile")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to patch Demucs - missing dependency: {e}")
        print("  Make sure soundfile is installed: pip install soundfile")
        return False
    except Exception as e:
        print(f"✗ Failed to patch Demucs: {e}")
        return False

# Auto-patch when this module is imported
print("\n" + "="*60)
print("Applying audio backend patches...")
print("="*60)
patch_torchaudio()
patch_demucs_audio()
print("="*60 + "\n")