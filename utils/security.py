import functools
from flask import request, jsonify, current_app
import pandas as pd
from werkzeug.utils import secure_filename
import os
import hashlib
import mimetypes
from pathlib import Path

def require_api_key(f):
    """Decorator to require API key for access."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "No API key provided"}), 401
            
        if not is_valid_api_key(api_key):
            return jsonify({"error": "Invalid API key"}), 401
            
        return f(*args, **kwargs)
    return decorated

def is_valid_api_key(api_key):
    """Validate API key."""
    # Implement your API key validation logic here
    # This is just a placeholder that always returns True
    return True

def validate_file(file):
    """Validate uploaded file."""
    try:
        if not file:
            return False
            
        # Check filename
        filename = secure_filename(file.filename)
        if not filename:
            return False
            
        # Check file extension
        if not allowed_file_extension(filename):
            return False
            
        # Check if file is actually a CSV by trying to read it
        try:
            # Read just the first few lines to verify it's a valid CSV
            df = pd.read_csv(file, nrows=5)
            file.seek(0)  # Reset file pointer
            return True
        except Exception as e:
            current_app.logger.error(f"CSV validation error: {str(e)}")
            return False
            
    except Exception as e:
        current_app.logger.error(f"File validation error: {str(e)}")
        return False

def allowed_file_extension(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def compute_file_hash(file_path):
    """Compute SHA-256 hash of file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def sanitize_filename(filename):
    """Sanitize filename and add unique identifier."""
    # Get base name and extension
    base = os.path.splitext(secure_filename(filename))[0]
    extension = os.path.splitext(filename)[1]
    
    # Add timestamp hash
    timestamp_hash = hashlib.md5(str(pd.Timestamp.now()).encode()).hexdigest()[:8]
    
    # Combine parts
    sanitized_name = f"{base}_{timestamp_hash}{extension}"
    
    return sanitized_name

def ensure_upload_folder(app):
    """Ensure upload folder exists and is writable."""
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(parents=True, exist_ok=True)
    
    # Test if folder is writable
    try:
        test_file = upload_folder / '.test'
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        current_app.logger.error(f"Upload folder is not writable: {str(e)}")
        raise RuntimeError("Upload folder is not writable")

def get_safe_filepath(filename, folder):
    """Get a safe filepath that doesn't overwrite existing files."""
    base = os.path.splitext(secure_filename(filename))[0]
    extension = os.path.splitext(filename)[1]
    counter = 1
    
    filepath = Path(folder) / f"{base}{extension}"
    while filepath.exists():
        filepath = Path(folder) / f"{base}_{counter}{extension}"
        counter += 1
        
    return filepath

class FileValidator:
    """Class for handling file validation and processing."""
    
    @staticmethod
    def validate_csv_content(file_path):
        """Validate CSV file content."""
        try:
            df = pd.read_csv(file_path)
            
            # Check if file is empty
            if df.empty:
                return False, "File is empty"
                
            # Check minimum number of columns
            if len(df.columns) < 2:
                return False, "File must contain at least 2 columns"
                
            # Check minimum number of rows
            if len(df) < 10:
                return False, "File must contain at least 10 rows"
                
            return True, "File is valid"
            
        except Exception as e:
            return False, f"Invalid CSV file: {str(e)}"
    
    @staticmethod
    def validate_file_size(file_path, max_size_mb=100):
        """Validate file size."""
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        return file_size <= max_size_mb
    

    def validate_file_size(file):
        
        try:
            file.seek(0, 2)  # Seek to end of file
            size = file.tell()  # Get current position (file size)
            file.seek(0)  # Reset file pointer to beginning
            
            max_size = 16 * 1024 * 1024  # 16MB in bytes
            if size > max_size:
                return False, f"File size ({size / 1024 / 1024:.1f}MB) exceeds maximum allowed size (16MB)"
            return True, ""
        except Exception as e:
            return False, f"Error checking file size: {str(e)}"