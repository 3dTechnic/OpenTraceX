import os
import time
import random
from flask import Flask, request, jsonify, render_template, send_from_directory, after_this_request
from werkzeug.utils import secure_filename
import trace_to_svg
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def is_allowed_file(file):
    # Check extension and verify the biological 'mimetype' is an image
    extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    return extension in ALLOWED_EXTENSIONS and file.content_type.startswith('image/')
def cleanup_folder(folder_path=UPLOAD_FOLDER):
    # Iterate through all items in the directory
    with os.scandir(folder_path) as entries:
        for entry in entries:
            # 1. Ensure it's a file (not a subfolder)
            if entry.is_file():
                
                # 2. Define your condition
                # Example: Delete if file starts with 'temp' AND is older than 1 hour
                b4time = time.time() - 60
                # file_stats = entry.stat()
                
                if entry.name.startswith("temp") and int(entry.name.split("_")[1]) < b4time and entry.name.endswith(".svg"):
                    try:
                        os.remove(entry.path)
                        print(f"Deleted: {entry.name}")
                    except OSError as e:
                        print(f"Error deleting {entry.name}: {e}")
@app.route('/upload', methods=['POST'])
def upload_image():
    cleanup_folder()
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and is_allowed_file(file):
        # Generate the specific filename requested
        # Format: temp_{timestamp}_{random}.jpg
        timestamp = round(time.time())
        rand_val = random.randint(0, 500)
        filename = f"temp_{timestamp}_{rand_val}.jpg"
        
        # Join with folder path
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Save the file
        file.save(filepath)
        a = trace_to_svg.process_frame(filepath,filepath.removesuffix(".jpg")+".svg")
        
        @after_this_request
        def remove_file(response):
            try:
                os.remove(filepath)
                print(f"Successfully deleted {filepath}")
            except Exception as error:
                app.logger.error(f"Error deleting file: {error}")
            return response
        if a!= True:
            return a
        return send_from_directory(UPLOAD_FOLDER, filename.removesuffix(".jpg")+".svg", as_attachment=True)
    
    return jsonify({"error": "File type not allowed. Please upload an image."}), 400

@app.route("/")
def home():
    return render_template("index.html")
if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")