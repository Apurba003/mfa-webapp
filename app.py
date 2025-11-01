# app.py - Complete Flask Application (ERROR-FREE)

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from datetime import datetime, timedelta
import numpy as np
import json

# ==================== IMPORTS (NO CIRCULAR IMPORTS) ====================

print("" + "="*70)
print("LOADING MODULES")
print("="*70)

try:
    from config import config
    print("✓ config imported")
except ImportError as e:
    print(f"✗ Error importing config: {e}")
    sys.exit(1)

try:
    from models import db, User, KeystrokeRecord, FaceRecord, AuthenticationAttempt
    print("✓ models imported")
except ImportError as e:
    print(f"✗ Error importing models: {e}")
    sys.exit(1)

try:
    from keystroke_dynamics import KeystrokeDynamics
    print("✓ keystroke_dynamics imported")
except ImportError as e:
    print(f"✗ Error importing keystroke_dynamics: {e}")
    sys.exit(1)

try:
    from face_recognition_module import FaceRecognitionModule
    print("✓ face_recognition_module imported")
except ImportError as e:
    print(f"✗ Error importing face_recognition_module: {e}")
    sys.exit(1)

try:
    from distance_metrics import DistanceMetrics
    print("✓ distance_metrics imported")
except ImportError as e:
    print(f"✗ Error importing distance_metrics: {e}")
    sys.exit(1)

try:
    from classification_engine import ClassificationEngine
    print("✓ classification_engine imported")
except ImportError as e:
    print(f"✗ Error importing classification_engine: {e}")
    sys.exit(1)

try:
    from feature_optimization import FeatureOptimization
    print("✓ feature_optimization imported")
except ImportError as e:
    print(f"✗ Error importing feature_optimization: {e}")
    sys.exit(1)

try:
    from auth_pipeline import AuthenticationPipeline
    print("✓ auth_pipeline imported")
except ImportError as e:
    print(f"✗ Error importing auth_pipeline: {e}")
    sys.exit(1)

try:
    from utils import create_folders, get_client_ip, save_json, load_json
    print("✓ utils imported")
except ImportError as e:
    print(f"✗ Error importing utils: {e}")
    sys.exit(1)

print("✓ All modules loaded successfully!")


# ==================== CREATE APP FUNCTION ====================

def create_app(config_name='development'):
    """Factory function to create Flask app - NO CIRCULAR IMPORTS"""
    
    app = Flask(__name__)
    
    # Load configuration
    try:
        app.config.from_object(config[config_name])
        print(f"✓ Configuration loaded: {config_name}")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        raise
    
    # Initialize extensions
    db.init_app(app)
    jwt = JWTManager(app)
    CORS(app, origins="*", supports_credentials=True)
    
    # Create folders
    try:
        create_folders(app)
        print("✓ Folders created")
    except Exception as e:
        print(f"⚠ Warning creating folders: {e}")
    
    # Initialize biometric modules
    try:
        keystroke_dynamics = KeystrokeDynamics()
        face_recognition = FaceRecognitionModule()
        classification_engine = ClassificationEngine()
        feature_optimization = FeatureOptimization()
        distance_metrics = DistanceMetrics()
        auth_pipeline = AuthenticationPipeline(
            keystroke_config={'threshold': app.config['KEYSTROKE_THRESHOLD']},
            face_config={'threshold': app.config['FACE_THRESHOLD']}
        )
        print("✓ Biometric modules initialized")
    except Exception as e:
        print(f"✗ Error initializing modules: {e}")
        raise
    
    # Make modules accessible throughout app
    app.keystroke_dynamics = keystroke_dynamics
    app.face_recognition = face_recognition
    app.classification_engine = classification_engine
    app.feature_optimization = feature_optimization
    app.distance_metrics = distance_metrics
    app.auth_pipeline = auth_pipeline
    
    # ==================== BASIC ROUTES ====================
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/register')
    def register_page():
        """Register page"""
        return render_template('register.html')
    
    @app.route('/login')
    def login_page():
        """Login page"""
        return render_template('login.html')
    
    @app.route('/dashboard')
    def dashboard_page():
        """Dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/profile')
    def profile_page():
        """Profile page"""
        return render_template('profile.html')
    
    @app.route('/keystroke_enroll')
    def keystroke_enroll_page():
        """Keystroke enrollment page"""
        return render_template('keystroke_enroll.html')
    
    @app.route('/face_enroll')
    def face_enroll_page():
        """Face enrollment page"""
        return render_template('face_enroll.html')
    
    @app.route('/keystroke_auth')
    def keystroke_auth_page():
        """Keystroke authentication page"""
        return render_template('keystroke_auth.html')
    
    @app.route('/face_auth')
    def face_auth_page():
        """Face authentication page"""
        return render_template('face_auth.html')
    
    @app.route('/mfa_verify')
    def mfa_verify_page():
        """MFA verification page"""
        return render_template('mfa_verify.html')
    
    # ==================== AUTHENTICATION ROUTES ====================
    
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        """Register new user"""
        try:
            data = request.get_json()
            
            # Validate input
            if not data.get('username') or not data.get('password') or not data.get('email'):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Validate username length
            if len(data['username']) < 3:
                return jsonify({'error': 'Username must be at least 3 characters'}), 400
            
            # Validate password length
            if len(data['password']) < 6:
                return jsonify({'error': 'Password must be at least 6 characters'}), 400
            
            # Check if user exists
            if User.query.filter_by(username=data['username']).first():
                return jsonify({'error': 'Username already exists'}), 409
            
            if User.query.filter_by(email=data['email']).first():
                return jsonify({'error': 'Email already exists'}), 409
            
            # Create user
            user = User(
                username=data['username'],
                email=data['email']
            )
            user.set_password(data['password'])
            
            db.session.add(user)
            db.session.commit()
            
            return jsonify({
                'message': 'User registered successfully',
                'user': user.to_dict()
            }), 201
        
        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        """Initial password authentication"""
        try:
            data = request.get_json()
            
            if not data.get('username') or not data.get('password'):
                return jsonify({'error': 'Missing username or password'}), 400
            
            user = User.query.filter_by(username=data.get('username')).first()
            
            if not user or not user.check_password(data.get('password')):
                return jsonify({'error': 'Invalid credentials'}), 401
            
            if not user.is_active:
                return jsonify({'error': 'User account is inactive'}), 403
            
            # Create temporary token for MFA (5 minutes)
            temp_token = create_access_token(
                identity=user.id,
                expires_delta=timedelta(minutes=5),
                additional_claims={'mfa_pending': True}
            )
            
            return jsonify({
                'temp_token': temp_token,
                'mfa_enabled': user.mfa_enabled,
                'keystroke_enabled': user.keystroke_enabled,
                'face_enabled': user.face_enabled,
                'user': user.to_dict()
            }), 200
        
        except Exception as e:
            print(f"Login error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== KEYSTROKE ENROLLMENT ====================
    
    @app.route('/api/keystroke/enroll', methods=['POST'])
    @jwt_required()
    def enroll_keystroke():
        """Enroll user keystroke dynamics"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            data = request.get_json()
            keystroke_data = data.get('keystroke_data')
            
            if not keystroke_data:
                return jsonify({'error': 'No keystroke data provided'}), 400
            
            # Extract features using KEYSTROKE DYNAMICS ALGORITHM
            features = app.keystroke_dynamics.extract_features(keystroke_data)
            
            if not features:
                return jsonify({'error': 'Could not extract keystroke features'}), 400
            
            # Store keystroke record
            record = KeystrokeRecord(
                user_id=user_id,
                features=json.dumps(features),
                is_training=True,
                is_authentic=True
            )
            db.session.add(record)
            db.session.commit()
            
            # Update sample count
            user.keystroke_samples = KeystrokeRecord.query.filter_by(
                user_id=user_id, is_training=True
            ).count()
            db.session.commit()
            
            # Check if enough samples collected
            samples_needed = app.config['KEYSTROKE_SAMPLES_NEEDED']
            if user.keystroke_samples >= samples_needed:
                # Build profile from all training samples
                keystroke_records = KeystrokeRecord.query.filter_by(
                    user_id=user_id, is_training=True
                ).all()
                
                keystroke_list = [json.loads(r.features) for r in keystroke_records]
                
                # Get profile statistics using FEATURE OPTIMIZATION
                profile = app.keystroke_dynamics.get_profile_stats(keystroke_list)
                
                if profile:
                    # Save profile
                    profile_path = os.path.join(
                        app.config['KEYSTROKE_PROFILES_FOLDER'],
                        f'user_{user_id}_keystroke_profile.json'
                    )
                    save_json(profile_path, profile)
                    
                    user.keystroke_enabled = True
                    user.keystroke_threshold = app.config['KEYSTROKE_THRESHOLD']
                    user.mfa_enabled = True
                    db.session.commit()
            
            return jsonify({
                'message': 'Keystroke sample recorded',
                'samples_collected': user.keystroke_samples,
                'samples_needed': samples_needed,
                'ready': user.keystroke_enabled,
                'features': features
            }), 200
        
        except Exception as e:
            db.session.rollback()
            print(f"Keystroke enroll error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== KEYSTROKE AUTHENTICATION ====================
    
    @app.route('/api/keystroke/authenticate', methods=['POST'])
    @jwt_required()
    def authenticate_keystroke():
        """Authenticate user via keystroke dynamics"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            if not user.keystroke_enabled:
                return jsonify({'error': 'Keystroke authentication not enabled'}), 400
            
            data = request.get_json()
            keystroke_data = data.get('keystroke_data')
            
            if not keystroke_data:
                return jsonify({'error': 'No keystroke data provided'}), 400
            
            # Extract features from test data using KEYSTROKE DYNAMICS
            test_features = app.keystroke_dynamics.extract_features(keystroke_data)
            
            if not test_features:
                return jsonify({'error': 'Could not extract keystroke features'}), 400
            
            # Load user profile
            profile_path = os.path.join(
                app.config['KEYSTROKE_PROFILES_FOLDER'],
                f'user_{user_id}_keystroke_profile.json'
            )
            
            if not os.path.exists(profile_path):
                return jsonify({'error': 'Keystroke profile not found'}), 400
            
            user_profile = load_json(profile_path)
            
            if not user_profile:
                return jsonify({'error': 'Could not load keystroke profile'}), 400
            
            # Convert features to array
            feature_values = list(test_features.values())
            
            # Authenticate using AUTHENTICATION PIPELINE (uses multiple distance metrics)
            authenticated, confidence, details = app.auth_pipeline.authenticate_keystroke(
                feature_values,
                user_profile
            )
            
            # Store authentication attempt
            attempt = AuthenticationAttempt(
                user_id=user_id,
                keystroke_score=confidence,
                keystroke_passed=authenticated,
                mfa_method='keystroke',
                ip_address=get_client_ip(request),
                user_agent=request.headers.get('User-Agent')
            )
            db.session.add(attempt)
            db.session.commit()
            
            return jsonify({
                'authenticated': authenticated,
                'confidence': confidence,
                'details': details
            }), 200
        
        except Exception as e:
            print(f"Keystroke auth error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== FACE ENROLLMENT ====================
    
    @app.route('/api/face/enroll', methods=['POST'])
    @jwt_required()
    def enroll_face():
        """Enroll user face"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Get base64 image
            data = request.get_json()
            image_base64 = data.get('image')
            
            if not image_base64:
                return jsonify({'error': 'No image provided'}), 400
            
            # Process image
            timestamp = str(datetime.utcnow().timestamp()).replace('.', '_')
            image_path = os.path.join(
                app.config['FACE_PROFILES_FOLDER'],
                f'user_{user_id}_face_{timestamp}.jpg'
            )
            
            image_array = app.face_recognition.process_base64_image(image_base64, image_path)
            
            if image_array is None:
                return jsonify({'error': 'Could not process image'}), 400
            
            # Check image quality
            quality = app.face_recognition.get_face_quality(image_array)
            if quality < 0.5:
                return jsonify({
                    'error': 'Image quality too low',
                    'quality': quality
                }), 400
            
            # Extract embedding using DEEPFACE & VGGFace2 ALGORITHM
            embedding = app.face_recognition.extract_embedding(image_array)
            
            if embedding is None:
                return jsonify({'error': 'Could not detect face in image'}), 400
            
            # Store face record
            record = FaceRecord(
                user_id=user_id,
                image_path=image_path,
                embedding=json.dumps(embedding.tolist()),
                is_training=True,
                is_authentic=True
            )
            db.session.add(record)
            db.session.commit()
            
            # Update sample count
            user.face_samples = FaceRecord.query.filter_by(
                user_id=user_id, is_training=True
            ).count()
            db.session.commit()
            
            # Check if enough samples collected
            samples_needed = app.config['FACE_SAMPLES_NEEDED']
            if user.face_samples >= samples_needed:
                # Build profile from all training samples
                face_records = FaceRecord.query.filter_by(
                    user_id=user_id, is_training=True
                ).all()
                
                embeddings = [np.array(json.loads(r.embedding)) for r in face_records]
                
                # Get profile statistics using FEATURE OPTIMIZATION
                profile = app.face_recognition.get_profile_embeddings(embeddings)
                
                if profile:
                    # Save profile
                    profile_path = os.path.join(
                        app.config['FACE_PROFILES_FOLDER'],
                        f'user_{user_id}_face_profile.json'
                    )
                    save_json(profile_path, profile)
                    
                    user.face_enabled = True
                    user.face_threshold = app.config['FACE_THRESHOLD']
                    user.mfa_enabled = True
                    db.session.commit()
            
            return jsonify({
                'message': 'Face sample recorded',
                'samples_collected': user.face_samples,
                'samples_needed': samples_needed,
                'ready': user.face_enabled,
                'quality': quality
            }), 200
        
        except Exception as e:
            db.session.rollback()
            print(f"Face enroll error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== FACE AUTHENTICATION ====================
    
    @app.route('/api/face/authenticate', methods=['POST'])
    @jwt_required()
    def authenticate_face():
        """Authenticate user via face recognition"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            if not user.face_enabled:
                return jsonify({'error': 'Face authentication not enabled'}), 400
            
            data = request.get_json()
            image_base64 = data.get('image')
            
            if not image_base64:
                return jsonify({'error': 'No image provided'}), 400
            
            # Process image
            image_array = app.face_recognition.process_base64_image(image_base64)
            
            if image_array is None:
                return jsonify({'error': 'Could not process image'}), 400
            
            # Extract embedding using DEEPFACE & VGGFace2
            embedding = app.face_recognition.extract_embedding(image_array)
            
            if embedding is None:
                return jsonify({'error': 'Could not detect face in image'}), 400
            
            # Load user profile
            profile_path = os.path.join(
                app.config['FACE_PROFILES_FOLDER'],
                f'user_{user_id}_face_profile.json'
            )
            
            if not os.path.exists(profile_path):
                return jsonify({'error': 'Face profile not found'}), 400
            
            user_profile = load_json(profile_path)
            
            if not user_profile:
                return jsonify({'error': 'Could not load face profile'}), 400
            
            # Authenticate using AUTHENTICATION PIPELINE (multiple distance metrics)
            authenticated, confidence, details = app.auth_pipeline.authenticate_face(
                embedding,
                user_profile
            )
            
            # Store authentication attempt
            attempt = AuthenticationAttempt(
                user_id=user_id,
                face_score=confidence,
                face_passed=authenticated,
                mfa_method='face',
                ip_address=get_client_ip(request),
                user_agent=request.headers.get('User-Agent')
            )
            db.session.add(attempt)
            db.session.commit()
            
            return jsonify({
                'authenticated': authenticated,
                'confidence': confidence,
                'details': details
            }), 200
        
        except Exception as e:
            print(f"Face auth error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== MFA VERIFICATION ====================
    
    @app.route('/api/mfa/verify', methods=['POST'])
    @jwt_required()
    def verify_mfa():
        """Complete MFA verification"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            data = request.get_json()
            keystroke_score = data.get('keystroke_score', 0)
            keystroke_result = data.get('keystroke_passed', False)
            face_score = data.get('face_score', 0)
            face_result = data.get('face_passed', False)
            
            # Combined authentication using AUTHENTICATION PIPELINE
            combined_result, combined_score, details = app.auth_pipeline.authenticate_combined(
                keystroke_score, keystroke_result,
                face_score, face_result
            )
            
            # Log attempt
            attempt = AuthenticationAttempt(
                user_id=user_id,
                keystroke_score=keystroke_score,
                keystroke_passed=keystroke_result,
                face_score=face_score,
                face_passed=face_result,
                combined_score=combined_score,
                success=combined_result,
                mfa_method='combined',
                ip_address=get_client_ip(request),
                user_agent=request.headers.get('User-Agent')
            )
            
            if combined_result:
                user.last_login = datetime.utcnow()
                db.session.add(attempt)
                db.session.commit()
                
                # Create full access token
                access_token = create_access_token(identity=user_id)
                
                return jsonify({
                    'success': True,
                    'access_token': access_token,
                    'combined_score': combined_score,
                    'user': user.to_dict()
                }), 200
            else:
                db.session.add(attempt)
                db.session.commit()
                
                return jsonify({
                    'success': False,
                    'combined_score': combined_score,
                    'message': 'MFA verification failed',
                    'details': details
                }), 401
        
        except Exception as e:
            db.session.rollback()
            print(f"MFA verify error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==================== USER PROFILE ROUTES ====================
    
    @app.route('/api/user/profile', methods=['GET'])
    @jwt_required()
    def get_profile():
        """Get user profile"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            return jsonify(user.to_dict()), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/user/update-profile', methods=['POST'])
    @jwt_required()
    def update_profile():
        """Update user profile"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            data = request.get_json()
            
            if 'email' in data:
                if User.query.filter_by(email=data['email']).filter(User.id != user_id).first():
                    return jsonify({'error': 'Email already in use'}), 409
                user.email = data['email']
            
            if 'keystroke_threshold' in data:
                user.keystroke_threshold = float(data['keystroke_threshold'])
            
            if 'face_threshold' in data:
                user.face_threshold = float(data['face_threshold'])
            
            db.session.commit()
            
            return jsonify({
                'message': 'Profile updated',
                'user': user.to_dict()
            }), 200
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/user/enable-mfa', methods=['POST'])
    @jwt_required()
    def enable_mfa():
        """Enable/disable MFA"""
        try:
            user_id = get_jwt_identity()
            user = User.query.get(user_id)
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            data = request.get_json()
            method = data.get('method')
            enabled = data.get('enabled', True)
            
            if method == 'keystroke':
                if enabled and not user.keystroke_enabled:
                    return jsonify({'error': 'Keystroke not enrolled'}), 400
                user.keystroke_enabled = enabled
            elif method == 'face':
                if enabled and not user.face_enabled:
                    return jsonify({'error': 'Face not enrolled'}), 400
                user.face_enabled = enabled
            
            # Update MFA status
            user.mfa_enabled = user.keystroke_enabled or user.face_enabled
            
            db.session.commit()
            
            return jsonify({
                'mfa_enabled': user.mfa_enabled,
                'keystroke_enabled': user.keystroke_enabled,
                'face_enabled': user.face_enabled,
                'message': 'MFA configuration updated'
            }), 200
        
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/user/auth-history', methods=['GET'])
    @jwt_required()
    def get_auth_history():
        """Get authentication history"""
        try:
            user_id = get_jwt_identity()
            
            limit = request.args.get('limit', 20, type=int)
            attempts = AuthenticationAttempt.query.filter_by(user_id=user_id).order_by(
                AuthenticationAttempt.timestamp.desc()
            ).limit(limit).all()
            
            return jsonify([attempt.to_dict() for attempt in attempts]), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # ==================== STATISTICS ROUTES ====================
    
    @app.route('/api/user/stats', methods=['GET'])
    @jwt_required()
    def get_user_stats():
        """Get user statistics"""
        try:
            user_id = get_jwt_identity()
            
            total_attempts = AuthenticationAttempt.query.filter_by(user_id=user_id).count()
            successful_attempts = AuthenticationAttempt.query.filter_by(
                user_id=user_id, success=True
            ).count()
            
            success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
            
            # Get recent attempts
            recent_attempts = AuthenticationAttempt.query.filter_by(user_id=user_id).order_by(
                AuthenticationAttempt.timestamp.desc()
            ).limit(10).all()
            
            keystroke_attempts = AuthenticationAttempt.query.filter_by(
                user_id=user_id, mfa_method='keystroke'
            ).count()
            
            face_attempts = AuthenticationAttempt.query.filter_by(
                user_id=user_id, mfa_method='face'
            ).count()
            
            combined_attempts = AuthenticationAttempt.query.filter_by(
                user_id=user_id, mfa_method='combined'
            ).count()
            
            return jsonify({
                'total_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'success_rate': success_rate,
                'keystroke_attempts': keystroke_attempts,
                'face_attempts': face_attempts,
                'combined_attempts': combined_attempts,
                'recent_attempts': [attempt.to_dict() for attempt in recent_attempts]
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/admin/stats', methods=['GET'])
    @jwt_required()
    def get_admin_stats():
        """Get system statistics"""
        try:
            total_users = User.query.count()
            mfa_enabled_users = User.query.filter_by(mfa_enabled=True).count()
            keystroke_users = User.query.filter_by(keystroke_enabled=True).count()
            face_users = User.query.filter_by(face_enabled=True).count()
            
            total_attempts = AuthenticationAttempt.query.count()
            successful_attempts = AuthenticationAttempt.query.filter_by(success=True).count()
            
            success_rate = (successful_attempts / total_attempts * 100) if total_attempts > 0 else 0
            
            return jsonify({
                'total_users': total_users,
                'mfa_enabled_users': mfa_enabled_users,
                'keystroke_users': keystroke_users,
                'face_users': face_users,
                'total_auth_attempts': total_attempts,
                'successful_attempts': successful_attempts,
                'success_rate': success_rate
            }), 200
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # ==================== HEALTH CHECK ====================
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'ok',
            'message': 'MFA Biometric System is running',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    
    # ==================== ERROR HANDLERS ====================
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        print(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized access'}), 401
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    print("✓ Flask application created successfully!")
    
    return app