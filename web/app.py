import os
import shutil
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from worker.audio_utils import process_song  # the decoupled process_song function

# --------------------------------------------------
# Flask setup
# --------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
@app.route('/')
def index():
    """Render the main upload page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process audio"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"✓ File uploaded: {filename}")

        # Process file using the decoupled process_song
        output_files, status = process_song(filepath, app.config['OUTPUT_FOLDER'])

        if status == "success":
            return jsonify({
                "message": "Processing complete",
                "files": {part: os.path.basename(path) for part, path in output_files.items()}
            })
        else:
            return jsonify({"error": status}), 500

    except Exception as e:
        print(f"✗ Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download a processed audio file"""
    file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)

    # Security: prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(app.config['OUTPUT_FOLDER'])):
        return jsonify({"error": "Invalid file path"}), 403

    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True, download_name=filename)


@app.route('/health')
def health_check():
    """Simple health check"""
    return jsonify({
        "status": "healthy",
        "message": "Flask app running and ready"
    })


# --------------------------------------------------
# Run server
# --------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*60)
    print("VOCAL SEPARATOR FLASK APP")
    print("="*60)
    print("Server starting on http://0.0.0.0:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)