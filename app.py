from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from bson import ObjectId
from config import Config
from database import db_manager
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

# Initialize Flask app
# Add after your imports in app.py

# Initialize directories
base_dir = os.path.dirname(os.path.abspath(__file__))
for directory in ['uploads', 'models', 'logs']:
    dir_path = os.path.join(base_dir, directory)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.chmod(dir_path, 0o755)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Set up logging
handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# Import db_manager after app initialization
from database import db_manager

# Set up logging
handler = RotatingFileHandler('logs/app.log', maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('AutoML platform startup')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['id'])
        self.username = user_data['username']
        self.email = user_data.get('email')
        self.user_data = user_data

    def get_id(self):
        return str(self.id)

def validate_file_size(file):
    try:
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        
        max_size = app.config.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024)
        if size > max_size:
            return False, f"File size ({size / 1024 / 1024:.1f}MB) exceeds maximum allowed size ({max_size / 1024 / 1024:.0f}MB)"
        return True, ""
    except Exception as e:
        return False, f"Error checking file size: {str(e)}"

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': f'File too large. Maximum file size is {app.config["MAX_CONTENT_LENGTH"] / 1024 / 1024}MB'
    }), 413

@login_manager.user_loader
def load_user(user_id):
    try:
        with db_manager.get_mysql_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user_data = cursor.fetchone()
            return User(user_data) if user_data else None
    except Exception as e:
        app.logger.error(f"Error loading user: {e}")
        return None

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            with db_manager.get_mysql_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                
                if user and check_password_hash(user['password'], password):
                    user_obj = User(user)
                    login_user(user_obj)
                    
                    cursor.execute("""
                        UPDATE users 
                        SET last_login = NOW() 
                        WHERE id = %s
                    """, (user['id'],))
                    conn.commit()
                    
                    next_page = request.args.get('next')
                    if not next_page or not next_page.startswith('/'):
                        next_page = url_for('dashboard')
                    return redirect(next_page)
                
                flash('Invalid username or password', 'error')
                
        except Exception as e:
            app.logger.error(f"Login error: {e}")
            flash('An error occurred during login', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            email = request.form.get('email')
            name = request.form.get('name')
            organization = request.form.get('organization', '')

            with db_manager.get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                # Check if username exists
                cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
                if cursor.fetchone():
                    flash('Username already exists', 'error')
                    return render_template('register.html')
                
                # Check if email exists
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cursor.fetchone():
                    flash('Email already exists', 'error')
                    return render_template('register.html')

                # Create new user
                cursor.execute("""
                    INSERT INTO users (username, password, email, name, organization, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (
                    username,
                    generate_password_hash(password),
                    email,
                    name,
                    organization
                ))
                conn.commit()

            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))

        except Exception as e:
            app.logger.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'error')

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        files = []
        try:
            with db_manager.get_mongo_connection() as mongo_db:
                files = list(mongo_db.uploads.find(
                    {"username": current_user.username}
                ).sort("upload_date", -1))
                
                # Get reports for each file
                for file in files:
                    file['reports'] = list(mongo_db.model_reports.find({
                        "file_id": str(file['_id']),
                        "username": current_user.username
                    }).sort("created_at", -1))
                    
        except Exception as e:
            app.logger.error(f"MongoDB error: {e}")
            flash('Error loading files from database', 'warning')

        return render_template(
            'dashboard.html',
            username=current_user.username,
            files=files
        )

    except Exception as e:
        app.logger.error(f"Dashboard error: {str(e)}")
        flash('Error loading dashboard', 'error')
        return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    try:
        app.logger.info("Upload request received")
        
        # Check for file
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        file = request.files['file']
        task_type = request.form.get('task_type')
        target_column = request.form.get('target_column')

        app.logger.info(f"Received file: {file.filename}, task_type: {task_type}, target_column: {target_column}")

        # Validate inputs
        if not file or not file.filename:
            app.logger.error("No file selected")
            return jsonify({'success': False, 'error': 'No file selected'}), 400

        if not task_type or not target_column:
            app.logger.error("Missing task type or target column")
            return jsonify({'success': False, 'error': 'Task type and target column are required'}), 400

        if not file.filename.endswith('.csv'):
            app.logger.error("Invalid file type")
            return jsonify({'success': False, 'error': 'Only CSV files are allowed'}), 400

        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save file
        try:
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")
        except Exception as e:
            app.logger.error(f"Error saving file: {e}")
            return jsonify({'success': False, 'error': 'Error saving file'}), 500

        # Validate CSV
        try:
            df = pd.read_csv(filepath)
            if target_column not in df.columns:
                os.remove(filepath)
                app.logger.error(f"Target column {target_column} not found in CSV")
                return jsonify({'success': False, 'error': f'Target column "{target_column}" not found in CSV'}), 400
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"Error reading CSV: {e}")
            return jsonify({'success': False, 'error': 'Invalid CSV format'}), 400

        # Store in MongoDB
        try:
            file_info = {
                "user_id": current_user.id,
                "username": current_user.username,
                "filename": unique_filename,
                "original_filename": filename,
                "filepath": filepath,
                "task_type": task_type,
                "target_column": target_column,
                "upload_date": datetime.utcnow(),
                "status": "uploaded",
                "columns": list(df.columns),
                "rows": len(df)
            }

            with db_manager.get_mongo_connection() as mongo_db:
                result = mongo_db.uploads.insert_one(file_info)
                app.logger.info(f"File info saved to MongoDB with id: {result.inserted_id}")

            return jsonify({
                'success': True,
                'file_id': str(result.inserted_id),
                'message': 'File uploaded successfully'
            })

        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            app.logger.error(f"MongoDB error: {e}")
            return jsonify({'success': False, 'error': 'Error saving to database'}), 500

    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/train/<file_id>', methods=['POST'])
@login_required
def train_model(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Read dataset
            df = pd.read_csv(file_info['filepath'])
            target_column = file_info['target_column']

            # Prepare data
            X = df.drop(target_column, axis=1)
            y = df[target_column]

            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

            if file_info['task_type'] == 'classification':
                le_y = LabelEncoder()
                y = le_y.fit_transform(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train models and get results
            results = {}
            
            if file_info['task_type'] == 'classification':
                models = {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingClassifier(random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1': f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    metrics['cv_score_mean'] = cv_scores.mean()
                    metrics['cv_score_std'] = cv_scores.std()

                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    metrics['feature_importance'] = feature_importance
                    
                    results[name] = metrics
                    
            else:  # Regression
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    metrics = {
                        'rmse': np.sqrt(mse),
                        'r2': r2_score(y_test, y_pred)
                    }
                    
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    metrics['cv_score_mean'] = cv_scores.mean()
                    metrics['cv_score_std'] = cv_scores.std()

                    feature_importance = dict(zip(X.columns, model.feature_importances_))
                    metrics['feature_importance'] = feature_importance
                    
                    results[name] = metrics

            # Save results to MongoDB
            report_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'file_id': file_id,
                'results': results,
                'created_at': datetime.utcnow(),
                'status': 'pending_approval',
                'task_type': file_info['task_type'],
                'target_column': target_column,
                'parameters': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cv_folds': 5,
                    'n_estimators': 100
                }
            }
            
            report_id = mongo_db.model_reports.insert_one(report_data)

            return jsonify({
                'success': True,
                'report_id': str(report_id.inserted_id),
                'results': results,
                'message': 'Models trained successfully. View report for details.'
            })

    except Exception as e:
        app.logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_dataset/<file_id>', methods=['POST'])
@login_required
def delete_dataset(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'Dataset not found'}), 404

            # Delete file from filesystem
            if os.path.exists(file_info['filepath']):
                os.remove(file_info['filepath'])

            # Delete from MongoDB
            mongo_db.uploads.delete_one({"_id": ObjectId(file_id)})
            mongo_db.model_reports.delete_many({"file_id": file_id})

            return jsonify({
                'success': True,
                'message': 'Dataset and associated reports deleted successfully'
            })

    except Exception as e:
        app.logger.error(f"Error deleting dataset: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/view_dataset/<file_id>')
@login_required
def view_dataset(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                flash('Dataset not found', 'error')
                return redirect(url_for('dashboard'))

            df = pd.read_csv(file_info['filepath'])
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }

            return render_template(
                'view_dataset.html',
                file_info=file_info,
                preview_data=df.head(10).to_dict('records'),
                columns=list(df.columns),
                stats=stats,
                shape=df.shape,
                dtypes=df.dtypes.astype(str).to_dict()
            )

    except Exception as e:
        app.logger.error(f"Error viewing dataset: {e}")
        flash('Error loading dataset', 'error')
        return redirect(url_for('dashboard'))

@app.route('/get_model_features/<report_id>')
@login_required
def get_model_features(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(report['file_id'])
            })
            
            if not file_info:
                return jsonify({'error': 'Dataset not found'}), 404

            return jsonify({
                'success': True,
                'features': list(report['results'][next(iter(report['results']))]['feature_importance'].keys()),
                'feature_importance': {
                    model_name: result['feature_importance']
                    for model_name, result in report['results'].items()
                    if 'feature_importance' in result
                }
            })

    except Exception as e:
        app.logger.error(f"Error getting model features: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='127.0.0.1', port=5000)