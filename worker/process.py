import sys
import os

# CRITICAL: patch FIRST
import demucs_patch

from audio_utils import process_song

if __name__ == "__main__":
    input_audio = sys.argv[1]
    output_dir = sys.argv[2]

    os.makedirs(output_dir, exist_ok=True)

    files, status = process_song(input_audio)

    if status != "success":
        raise RuntimeError(status)

    # Move outputs into shared folder
    for _, path in files.items():
        os.rename(path, os.path.join(output_dir, os.path.basename(path)))