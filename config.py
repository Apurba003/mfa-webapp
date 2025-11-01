# config.py - Complete Configuration File

import os
from datetime import timedelta

class Config:
    """Base configuration"""
    
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-this')
    DEBUG = False
    TESTING = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'SQLALCHEMY_DATABASE_URI',
        'sqlite:///mfa_system.db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT Configuration
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-dev-key-change-this')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    
    # Upload Configuration
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # MFA Configuration
    KEYSTROKE_THRESHOLD = float(os.getenv('KEYSTROKE_THRESHOLD', 0.7))
    FACE_THRESHOLD = float(os.getenv('FACE_THRESHOLD', 0.6))
    COMBINED_THRESHOLD = float(os.getenv('COMBINED_THRESHOLD', 0.75))
    
    # Biometric Configuration
    KEYSTROKE_SAMPLES_NEEDED = int(os.getenv('KEYSTROKE_SAMPLES_NEEDED', 5))
    FACE_SAMPLES_NEEDED = int(os.getenv('FACE_SAMPLES_NEEDED', 3))
    
    # Folder Configuration
    FACE_PROFILES_FOLDER = os.getenv('FACE_PROFILES_FOLDER', 'data/face_embeddings')
    KEYSTROKE_PROFILES_FOLDER = os.getenv('KEYSTROKE_PROFILES_FOLDER', 'data/keystroke_profiles')
    MODEL_FOLDER = os.getenv('MODEL_FOLDER', 'data/models')
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = True


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}