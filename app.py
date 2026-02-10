import os

# CRITICAL: Set environment variables BEFORE any other imports
os.environ["TORCHAUDIO_USE_SOUNDFILE"] = "1"
os.environ["TORCHAUDIO_BACKEND"] = "soundfile"
os.environ["DEMUC_AUDIO_BACKEND"] = "ffmpeg"

# Import the patch IMMEDIATELY to override Demucs audio functions
import demucs_patch

from flask import Flask, render_template, request, jsonify, send_file
import subprocess
import tempfile
import shutil
import sys

import torch
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt
from werkzeug.utils import secure_filename

# --------------------------------------------------
# Check FFmpeg installation
# --------------------------------------------------
def check_ffmpeg():
    """Verify FFmpeg is installed and accessible"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print("✓ FFmpeg is installed and accessible")
        return True
    except FileNotFoundError:
        print("✗ FFmpeg is not installed or not in your PATH.")
        print("  Please install FFmpeg and add it to PATH.")
        print("  Download from: https://ffmpeg.org/download.html")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("✗ FFmpeg exists but could not be executed properly.")
        sys.exit(1)

check_ffmpeg()

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --------------------------------------------------
# Audio utilities
# --------------------------------------------------
def separate_vocals(audio_path):
    """
    Separate vocals using Demucs Python API (not subprocess).
    This allows our monkey-patch to work properly.
    
    Args:
        audio_path: Path to input audio file
        
    Returns:
        tuple: (vocals_path, temp_dir) - Path to separated vocals and temp directory
    """
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    
    temp_dir = tempfile.mkdtemp(prefix="demucs_")
    
    print(f"Processing audio: {audio_path}")
    print(f"Temporary directory: {temp_dir}")

    try:
        # Load audio using torchaudio (will use our patched backend)
        print("Loading audio file...")
        
        # Use librosa to load audio (avoids torchaudio issues)
        audio, sr = librosa.load(audio_path, sr=44100, mono=False)
        
        # Convert to torch tensor
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # Add channel dimension
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add batch dimension if needed
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        print(f"  Audio shape: {audio_tensor.shape}")
        print(f"  Sample rate: {sr} Hz")
        
        # Load Demucs model
        print("Loading Demucs model (this may take a moment on first run)...")
        model = get_model('htdemucs')
        model.eval()
        
        # Move to CPU (or GPU if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        audio_tensor = audio_tensor.to(device)
        
        print(f"  Using device: {device}")
        
        # Apply separation
        print("Separating vocals (this may take 1-3 minutes)...")
        with torch.no_grad():
            sources = apply_model(model, audio_tensor, device=device)
        
        # Extract vocals (index depends on model.sources)
        # For htdemucs: ['drums', 'bass', 'other', 'vocals']
        vocals_idx = model.sources.index('vocals')
        vocals = sources[0, vocals_idx].cpu().numpy()
        
        print("✓ Demucs separation completed")
        
        # Save vocals using soundfile (our patched method)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_dir = os.path.join(temp_dir, "htdemucs", base_name)
        os.makedirs(vocals_dir, exist_ok=True)
        
        vocals_path = os.path.join(vocals_dir, "vocals.wav")
        
        # Transpose if needed: (channels, samples) -> (samples, channels)
        if vocals.ndim == 2 and vocals.shape[0] < vocals.shape[1]:
            vocals = vocals.T
        
        # Save using soundfile
        sf.write(vocals_path, vocals, sr)
        
        print(f"✓ Vocals saved to: {vocals_path}")
        return vocals_path, temp_dir
        
    except Exception as e:
        print(f"✗ Demucs separation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def gentle_bandpass_filter(y, sr, lowcut, highcut, boost_amount=1.5):
    """
    A gentler approach: Soft bandpass that preserves more harmonics.
    
    Args:
        y: Audio time series
        sr: Sample rate
        lowcut: Low frequency cutoff (Hz)
        highcut: High frequency cutoff (Hz)
        boost_amount: Multiplier for the passed frequencies (default: 1.5)
        
    Returns:
        numpy.ndarray: Filtered audio
    """
    # Use a gentler 3rd order filter
    sos = butter(
        3,
        [lowcut, highcut],
        btype='bandpass',
        fs=sr,
        output='sos'
    )
    
    # Apply filter
    filtered = sosfilt(sos, y)
    
    # Mix with original for a gentler effect
    # This preserves some full-spectrum content
    mixed = (filtered * boost_amount + y * 0.3) / (boost_amount + 0.3)
    
    # Normalize to prevent clipping
    max_val = np.abs(mixed).max()
    if max_val > 0:
        mixed = mixed * 0.95 / max_val
    
    return mixed


# --------------------------------------------------
# Main processing
# --------------------------------------------------
def process_song(audio_path):
    """
    Process audio file to create vocal stems.
    
    IMPORTANT: This creates frequency-filtered versions of the SAME vocal track.
    It does NOT separate individual singers or vocal parts.
    
    Steps:
    1. Separate vocals from music using Demucs
    2. Load separated vocals
    3. Create filtered versions for different frequency ranges
    4. Save all stems
    
    Args:
        audio_path: Path to input audio file
        
    Returns:
        tuple: (output_files dict, status string)
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
        
        # Note: These are NOT individual singers, just frequency-filtered versions
        # of the same vocal track to emphasize different pitch ranges
        freq_ranges = {
            "soprano_range": (250, 2500),    # High pitch emphasis
            "alto_range":    (180, 1800),    # Mid-high pitch emphasis
            "tenor_range":   (120, 1500),    # Mid pitch emphasis
            "lead_vocals":   (150, 2000),    # Full lead vocal range
            "full_vocals":   None            # Unfiltered vocals
        }

        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # Step 4: Create filtered stems
        for part, freq_range in freq_ranges.items():
            if freq_range is None:
                # Save unfiltered vocals
                print(f"  Creating {part} (unfiltered)...")
                output_path = os.path.join(
                    app.config['OUTPUT_FOLDER'],
                    f"{base_name}_{part}.wav"
                )
                sf.write(output_path, y_vocals, sr)
                output_files[part] = output_path
                print(f"    ✓ Saved: {os.path.basename(output_path)}")
            else:
                low, high = freq_range
                print(f"  Creating {part} ({low}-{high} Hz)...")
                
                # Apply gentle bandpass filter
                filtered = gentle_bandpass_filter(y_vocals, sr, low, high, boost_amount=2.0)

                # Save to output folder
                output_path = os.path.join(
                    app.config['OUTPUT_FOLDER'],
                    f"{base_name}_{part}.wav"
                )

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
        
        # Cleanup on error
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        return None, str(e)


# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and processing.
    
    Expected:
        - POST request with 'file' in request.files
        
    Returns:
        JSON response with processing results or error
    """
    # Validate file upload
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file extension
    allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
        }), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n✓ File uploaded: {filename}")
        
        # Process the song
        output_files, status = process_song(filepath)

        if status == "success":
            return jsonify({
                "message": "Processing complete",
                "files": {
                    part: os.path.basename(path)
                    for part, path in output_files.items()
                }
            })
        else:
            return jsonify({"error": status}), 500
            
    except Exception as e:
        print(f"✗ Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """
    Download a processed audio file.
    
    Args:
        filename: Name of the file to download
        
    Returns:
        File download response
    """
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Security check - prevent directory traversal
        if not os.path.abspath(file_path).startswith(
            os.path.abspath(app.config['OUTPUT_FOLDER'])
        ):
            return jsonify({"error": "Invalid file path"}), 403
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "demucs_patched": True,
        "filter_type": "gentle_bandpass"
    })


# --------------------------------------------------
# Run app
# --------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("VOCAL SEPARATOR APPLICATION")
    print("="*60)
    print("Note: This app separates vocals from music,")
    print("then creates frequency-filtered versions.")
    print("It does NOT separate individual singers.")
    print("Server starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)