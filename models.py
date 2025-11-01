# models.py - Complete Database Models

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(db.Model):
    """User model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    
    # MFA Status
    mfa_enabled = db.Column(db.Boolean, default=False)
    keystroke_enabled = db.Column(db.Boolean, default=False)
    face_enabled = db.Column(db.Boolean, default=False)
    
    # Biometric Samples
    keystroke_samples = db.Column(db.Integer, default=0)
    face_samples = db.Column(db.Integer, default=0)
    
    # Thresholds
    keystroke_threshold = db.Column(db.Float, default=0.7)
    face_threshold = db.Column(db.Float, default=0.6)
    
    # Account Status
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    keystroke_records = db.relationship('KeystrokeRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    face_records = db.relationship('FaceRecord', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    auth_attempts = db.relationship('AuthenticationAttempt', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password, method='pbkdf2:sha256')
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'mfa_enabled': self.mfa_enabled,
            'keystroke_enabled': self.keystroke_enabled,
            'face_enabled': self.face_enabled,
            'keystroke_samples': self.keystroke_samples,
            'face_samples': self.face_samples,
            'keystroke_threshold': self.keystroke_threshold,
            'face_threshold': self.face_threshold,
            'is_active': self.is_active,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat()
        }


class KeystrokeRecord(db.Model):
    """Keystroke record model"""
    __tablename__ = 'keystroke_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Features stored as JSON
    features = db.Column(db.Text, nullable=False)
    
    # Status
    is_training = db.Column(db.Boolean, default=True)
    is_authentic = db.Column(db.Boolean, default=True)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'features': json.loads(self.features),
            'is_training': self.is_training,
            'is_authentic': self.is_authentic,
            'timestamp': self.timestamp.isoformat()
        }


class FaceRecord(db.Model):
    """Face record model"""
    __tablename__ = 'face_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Image data
    image_path = db.Column(db.String(255), nullable=False)
    embedding = db.Column(db.Text, nullable=False)  # JSON array
    
    # Status
    is_training = db.Column(db.Boolean, default=True)
    is_authentic = db.Column(db.Boolean, default=True)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'image_path': self.image_path,
            'embedding': json.loads(self.embedding),
            'is_training': self.is_training,
            'is_authentic': self.is_authentic,
            'timestamp': self.timestamp.isoformat()
        }


class AuthenticationAttempt(db.Model):
    """Authentication attempt model"""
    __tablename__ = 'authentication_attempts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Scores
    keystroke_score = db.Column(db.Float, nullable=True)
    keystroke_passed = db.Column(db.Boolean, nullable=True)
    face_score = db.Column(db.Float, nullable=True)
    face_passed = db.Column(db.Boolean, nullable=True)
    combined_score = db.Column(db.Float, nullable=True)
    
    # Result
    success = db.Column(db.Boolean, default=False)
    mfa_method = db.Column(db.String(50), default='combined')  # keystroke, face, combined
    
    # Request Info
    ip_address = db.Column(db.String(50), nullable=True)
    user_agent = db.Column(db.String(255), nullable=True)
    
    # Timestamps
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'keystroke_score': self.keystroke_score,
            'keystroke_passed': self.keystroke_passed,
            'face_score': self.face_score,
            'face_passed': self.face_passed,
            'combined_score': self.combined_score,
            'success': self.success,
            'mfa_method': self.mfa_method,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat()
        }