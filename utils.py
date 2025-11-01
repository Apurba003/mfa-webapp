# utils.py - Utility Functions

import os
import json
import shutil
from datetime import datetime


def create_folders(app):
    """
    Create required folders
    
    Args:
        app: Flask app instance
    """
    folders = [
        app.config['UPLOAD_FOLDER'],
        app.config['FACE_PROFILES_FOLDER'],
        app.config['KEYSTROKE_PROFILES_FOLDER'],
        app.config['MODEL_FOLDER']
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def get_client_ip(request):
    """
    Get client IP address
    
    Args:
        request: Flask request object
        
    Returns:
        str: Client IP
    """
    try:
        if request.environ.get('HTTP_X_FORWARDED_FOR'):
            return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0]
        return request.environ.get('REMOTE_ADDR')
    except:
        return 'Unknown'


def save_json(path, data):
    """
    Save data as JSON
    
    Args:
        path: File path
        data: Data to save
        
    Returns:
        bool: Success status
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False


def load_json(path):
    """
    Load data from JSON
    
    Args:
        path: File path
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return None


def log_event(event_type, user_id, details):
    """
    Log system events
    
    Args:
        event_type: Type of event
        user_id: User ID
        details: Event details
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'user_id': user_id,
        'details': details
    }
    
    log_file = 'logs/events.json'
    os.makedirs('logs', exist_ok=True)
    
    try:
        events = load_json(log_file) or []
        events.append(log_entry)
        save_json(log_file, events[-100:])  # Keep last 100 events
    except:
        pass


def cleanup_old_files(directory, days=30):
    """
    Clean up old files
    
    Args:
        directory: Directory to clean
        days: Age threshold in days
    """
    try:
        import time
        now = time.time()
        threshold = now - (days * 86400)
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                if os.stat(filepath).st_mtime < threshold:
                    os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up files: {e}")