# run.py - Application Entry Point (ERROR-FREE)

import os
import sys
from datetime import datetime

print("" + "="*70)
print("MFA BIOMETRIC SYSTEM - STARTUP SEQUENCE")
print("="*70 + "")

# ==================== STEP 1: LOAD ENVIRONMENT ====================

print("[1/5] Loading environment variables...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Environment variables loaded")
except ImportError:
    print("⚠ python-dotenv not found (optional)")

# ==================== STEP 2: IMPORT APP & DB ====================

print("[2/5] Importing application modules...")

try:
    from app import create_app, db
    print("✓ app module imported")
except ImportError as e:
    print(f"✗ FATAL ERROR: Cannot import app: {e}")
    print("Make sure app.py exists in the project root")
    sys.exit(1)

try:
    from models import User, KeystrokeRecord, FaceRecord, AuthenticationAttempt
    print("✓ models module imported")
except ImportError as e:
    print(f"✗ FATAL ERROR: Cannot import models: {e}")
    print("Make sure models.py exists in the project root")
    sys.exit(1)

print()

# ==================== STEP 3: CREATE APP INSTANCE ====================

print("[3/5] Creating application instance...")

config_name = os.getenv('FLASK_ENV', 'development')

try:
    app = create_app(config_name)
    print(f"✓ Application created")
except Exception as e:
    print(f"✗ FATAL ERROR: Cannot create application: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== STEP 4: SETUP SHELL CONTEXT ====================

print("[4/5] Setting up shell context...")

@app.shell_context_processor
def make_shell_context():
    """Make database models available in flask shell"""
    return {
        'db': db,
        'User': User,
        'KeystrokeRecord': KeystrokeRecord,
        'FaceRecord': FaceRecord,
        'AuthenticationAttempt': AuthenticationAttempt
    }

print("✓ Shell context configured")

# ==================== STEP 5: REGISTER CLI COMMANDS ====================

print("[5/5] Registering CLI commands...")

@app.cli.command()
def init_db():
    """Initialize the database"""
    try:
        with app.app_context():
            db.create_all()
            print("✓ Database initialized successfully!")
            print(f"✓ Location: {os.path.abspath('mfa_system.db')}")
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)

@app.cli.command()
def drop_db():
    """Drop all database tables (WARNING: This deletes all data!)"""
    try:
        confirm = input("⚠ This will delete all data. Continue? (y/n): ")
        if confirm.lower() == 'y':
            with app.app_context():
                db.drop_all()
                print("✓ Database dropped successfully!")
        else:
            print("✗ Operation cancelled")
    except Exception as e:
        print(f"✗ Error dropping database: {e}")
        sys.exit(1)

@app.cli.command()
def create_admin():
    """Create an admin user"""
    try:
        print("" + "="*50)
        print("CREATE ADMIN USER")
        print("="*50)
        
        username = input("Enter admin username: ").strip()
        email = input("Enter admin email: ").strip()
        password = input("Enter admin password: ").strip()
        
        # Validate input
        if len(username) < 3:
            print("✗ Username must be at least 3 characters")
            return
        
        if len(password) < 6:
            print("✗ Password must be at least 6 characters")
            return
        
        if '@' not in email:
            print("✗ Invalid email address")
            return
        
        with app.app_context():
            # Check if user already exists
            if User.query.filter_by(username=username).first():
                print(f"✗ Admin user '{username}' already exists!")
                return
            
            if User.query.filter_by(email=email).first():
                print(f"✗ Email '{email}' already in use!")
                return
            
            # Create admin user
            admin = User(username=username, email=email)
            admin.set_password(password)
            
            db.session.add(admin)
            db.session.commit()
            
            print("" + "="*50)
            print("✓ ADMIN USER CREATED SUCCESSFULLY")
            print("="*50)
            print(f"Username: {username}")
            print(f"Email: {email}")
            print("="*50 + "")
    
    except Exception as e:
        print(f"✗ Error creating admin user: {e}")
        sys.exit(1)

@app.cli.command()
def list_users():
    """List all users in the database"""
    try:
        with app.app_context():
            users = User.query.all()
            
            if not users:
                print("No users found in database")
                return
            
            print("" + "="*80)
            print("REGISTERED USERS")
            print("="*80)
            print(f"{'ID':<5} {'Username':<15} {'Email':<30} {'MFA':<5} {'Created':<20}")
            print("-"*80)
            
            for user in users:
                mfa_status = "✓" if user.mfa_enabled else "✗"
                created = user.created_at.strftime("%Y-%m-%d %H:%M:%S")
                print(f"{user.id:<5} {user.username:<15} {user.email:<30} {mfa_status:<5} {created:<20}")
            
            print("="*80 + "")
    
    except Exception as e:
        print(f"✗ Error listing users: {e}")
        sys.exit(1)

@app.cli.command()
def reset_db():
    """Reset database (drop and recreate)"""
    try:
        confirm = input("⚠ This will delete all data. Continue? (y/n): ")
        if confirm.lower() == 'y':
            with app.app_context():
                print("Dropping database...")
                db.drop_all()
                print("Creating new database...")
                db.create_all()
                print("✓ Database reset successfully!")
        else:
            print("✗ Operation cancelled")
    except Exception as e:
        print(f"✗ Error resetting database: {e}")
        sys.exit(1)

print("✓ CLI commands registered:")
print("  - flask init-db         Initialize database")
print("  - flask drop-db         Drop all tables")
print("  - flask create-admin    Create admin user")
print("  - flask list-users      List all users")
print("  - flask reset-db        Reset database")
print()

# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    
    # Initialize database on startup
    print("="*70)
    print("INITIALIZING DATABASE")
    print("="*70 + "")
    
    try:
        with app.app_context():
            db.create_all()
            print("✓ Database check complete")
            
            # Count users
            user_count = User.query.count()
            print(f"✓ Total users in database: {user_count}")
    
    except Exception as e:
        print(f"✗ Error during database initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ==================== START SERVER ====================
    
    print("="*70)
    print("STARTING MFA BIOMETRIC SYSTEM")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Environment: {config_name}")
    print(f"Debug Mode: {app.debug}")
    print(f"Host: 0.0.0.0")
    print(f"Port: 5000")
    print("" + "="*70)
    print("ACCESS THE APPLICATION AT:")
    print("  → http://localhost:5000")
    print("  → http://127.0.0.1:5000")
    print("="*70)
    print("API ENDPOINTS:")
    print("  • POST   /api/auth/register          - Register new user")
    print("  • POST   /api/auth/login             - Login user")
    print("  • POST   /api/keystroke/enroll       - Enroll keystroke")
    print("  • POST   /api/keystroke/authenticate - Authenticate keystroke")
    print("  • POST   /api/face/enroll            - Enroll face")
    print("  • POST   /api/face/authenticate      - Authenticate face")
    print("  • POST   /api/mfa/verify             - Verify MFA")
    print("  • GET    /api/user/profile           - Get user profile")
    print("  • GET    /api/user/stats             - Get user statistics")
    print("  • GET    /api/user/auth-history      - Get auth history")
    print("  • GET    /api/health                 - Health check")
    print("ALGORITHMS USED:")
    print("  • Keystroke Dynamics: Dwell Time, Flight Time, Typing Speed, Rhythm")
    print("  • Face Recognition: DeepFace + VGGFace2 Deep Learning Model")
    print("  • Distance Metrics: Euclidean, Mahalanobis, Cosine, Manhattan")
    print("  • ML Algorithms: Neural Network (MLP), Random Forest, SVM, k-NN")
    print("  • Feature Opt: PCA, Normalization, Standardization, Statistics")
    print("" + "="*70)
    print("PRESS Ctrl+C TO STOP THE SERVER")
    print("="*70 + "")
    
    try:
        # Determine debug mode based on environment
        debug_mode = config_name == 'development'
        
        # Run the Flask application
        app.run(
            debug=debug_mode,
            host='0.0.0.0',
            port=5000,
            use_reloader=debug_mode,
            use_debugger=debug_mode
        )
    
    except KeyboardInterrupt:
        print("✓ Server stopped by user")
        print("="*70 + "")
        sys.exit(0)
    
    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)