# Flask and related imports
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from asgiref.sync import async_to_sync
from functools import wraps

# Werkzeug utilities
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Standard library imports
import os
import logging
from logging.handlers import RotatingFileHandler
import json
import warnings

# Data processing and analysis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# MongoDB
from bson import ObjectId

# Local imports
from config import Config
from database import db_manager
from profiling import DataProfiler

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    mean_squared_error, 
    r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Statistics
import scipy.stats as stats

from ai_assistant import AIAssistant
import google.generativeai as genai
from flask import current_app

# Initialize Gemini at the start of app.py
GOOGLE_API_KEY = 'AIzaSyCMemd6wrMxIzEsbhbYajJY0-ee5wXBrcw'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Suppress warnings
warnings.filterwarnings('ignore')

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
app.config['WTF_CSRF_ENABLED'] = True
app.config['WTF_CSRF_SECRET_KEY'] = os.urandom(32)
app.config['WTF_CSRF_TIME_LIMIT'] = 3600  # 1 hour

# Set up logging
LOG_DIR = os.path.join(str(Path.home()), 'automl_logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_file = os.path.join(LOG_DIR, 'app.log')
handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# Session configuration
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=1)
)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

def async_route(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        return async_to_sync(f)(*args, **kwargs)
    return decorated_function

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['id'])
        self.username = user_data['username']
        self.email = user_data.get('email')
        self.user_data = user_data

    def get_id(self):
        return str(self.id)

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

def perform_enhanced_eda(df, target_column):
    """Perform comprehensive EDA"""
    # Get missing value information
    missing_values = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            missing_values[col] = {
                'count': int(missing_count),
                'percentage': float((missing_count / len(df)) * 100)
            }
    
    # Get numerical and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Build comprehensive EDA results
    eda_results = {
        'basic_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
            'duplicate_rows': int(df.duplicated().sum())
        },
        'missing_values': missing_values,
        'numerical_cols': list(numerical_cols),
        'categorical_cols': list(categorical_cols),
        'correlations': analyze_correlations(df),
        'numerical_stats': {},
        'categorical_stats': {},
        'target_analysis': analyze_target_distribution(df, target_column)
    }
    
    # Calculate numerical statistics
    for col in numerical_cols:
        stats = df[col].describe()
        eda_results['numerical_stats'][col] = {
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'quartiles': {
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%'])
            }
        }
    
    # Calculate categorical statistics
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        eda_results['categorical_stats'][col] = {
            'unique_values': int(value_counts.count()),
            'top_values': value_counts.head(5).to_dict(),
            'value_counts': value_counts.to_dict()
        }
    
    return eda_results

def analyze_target_distribution(df, target_column):
    """Analyze target variable distribution"""
    target_analysis = {
        'type': 'categorical' if df[target_column].dtype == 'object' else 'numeric',
        'unique_count': int(df[target_column].nunique()),
        'missing_count': int(df[target_column].isnull().sum())
    }
    
    if target_analysis['type'] == 'numeric':
        stats = df[target_column].describe()
        target_analysis.update({
            'mean': float(stats['mean']),
            'std': float(stats['std']),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'quartiles': {
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%'])
            }
        })
    else:
        value_counts = df[target_column].value_counts()
        target_analysis.update({
            'value_counts': value_counts.to_dict(),
            'percentages': (value_counts / len(df) * 100).to_dict()
        })
    
    return target_analysis

def analyze_correlations(df):
    """Analyze correlations between features"""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    if len(numeric_df.columns) < 2:
        return {
            'has_correlations': False,
            'message': 'Not enough numeric columns for correlation analysis'
        }
    
    correlation_matrix = numeric_df.corr().round(4)
    
    return {
        'has_correlations': True,
        'correlation_matrix': correlation_matrix.to_dict()
    }

def preprocess_dataset(df, target_column, task_type):
    """Preprocess dataset for ML"""
    preprocessing_steps = {
        'steps_taken': [],
        'dropped_columns': [],
        'encoded_columns': [],
        'scaled_columns': []
    }
    
    # Create a copy
    data = df.copy()
    encoders = {}
    
    try:
        # Handle missing values first
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(data[column]):
                    data[column].fillna(data[column].mean(), inplace=True)
                    preprocessing_steps['steps_taken'].append(f"Filled missing values in {column} with mean")
                else:
                    data[column].fillna(data[column].mode()[0], inplace=True)
                    preprocessing_steps['steps_taken'].append(f"Filled missing values in {column} with mode")
        
        # Handle target variable
        y = data[target_column].copy()
        if task_type == 'classification' or y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            encoders['target'] = le
            preprocessing_steps['steps_taken'].append(f"Label encoded target column: {target_column}")
        
        # Handle feature columns
        X = data.drop(columns=[target_column])
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
                encoders[column] = le
                preprocessing_steps['encoded_columns'].append(column)
                preprocessing_steps['steps_taken'].append(f"Label encoded column: {column}")
        
        # Scale numeric features
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
            preprocessing_steps['scaled_columns'] = numeric_columns.tolist()
            preprocessing_steps['steps_taken'].append("Scaled numeric features")
            encoders['scaler'] = scaler
        
        preprocessing_steps['final_features'] = list(X.columns)
        return X, y, preprocessing_steps, encoders
        
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

def calculate_feature_importance(X, y, task_type):
    """Calculate feature importance"""
    importance_scores = {}
    
    # Random Forest Feature Importance
    if task_type == 'classification':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        importance_metric = mutual_info_classif
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        importance_metric = mutual_info_regression
    
    rf.fit(X, y)
    importance_scores['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
    
    # Mutual Information Feature Importance
    mi_scores = importance_metric(X, y)
    importance_scores['mutual_information'] = dict(zip(X.columns, mi_scores))
    
    # Aggregate and sort features by importance
    feature_ranks = {}
    for feature in X.columns:
        feature_ranks[feature] = (
            importance_scores['random_forest'][feature] +
            importance_scores['mutual_information'][feature]
        ) / 2
    
    return {
        'detailed_scores': importance_scores,
        'aggregate_ranks': dict(sorted(feature_ranks.items(), key=lambda x: x[1], reverse=True))
    }

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
        
        with db_manager.get_mysql_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if user and check_password_hash(user['password'], password):
                user_obj = User(user)
                login_user(user_obj)
                return redirect(url_for('dashboard'))
            
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        name = request.form.get('name')
        organization = request.form.get('organization')
        
        with db_manager.get_mysql_connection() as conn:
            cursor = conn.cursor()
            # Check if username or email already exists
            cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
            if cursor.fetchone():
                flash('Username or email already exists', 'error')
                return render_template('register.html')
                
            hashed_password = generate_password_hash(password)
            cursor.execute(
                """INSERT INTO users (username, password, email, name, organization, created_at) 
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (username, hashed_password, email, name, organization, datetime.utcnow())
            )
            conn.commit()
            
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            files = list(mongo_db.uploads.find({"username": current_user.username}))
            return render_template('dashboard.html', files=files, username=current_user.username)
    except Exception as e:
        app.logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard', 'error')
        return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
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
    
# Replace the generate_profile route with this:
@app.route('/generate_profile/<file_id>')
@login_required
@async_route
async def generate_profile(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Load data
            df = pd.read_csv(file_info['filepath'])
            
            # Generate profile report
            profiler = DataProfiler(df)
            html_report = profiler.generate_html_report()
            
            # Save report
            report_path = os.path.join(app.config['UPLOAD_FOLDER'], f'profile_{file_id}.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            return send_file(report_path, as_attachment=False)

    except Exception as e:
        app.logger.error(f"Profile generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    pass

@app.route('/train/<file_id>', methods=['GET', 'POST'])
@login_required
@async_route
async def train_model(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Load data
            df = pd.read_csv(file_info['filepath'])
            target_column = file_info['target_column']
            task_type = file_info['task_type']

            if request.method == 'GET':
                # Get AI-powered analysis
                analysis_result = await ai_assistant.analyze_data(df, target_column, task_type)
                
                if not analysis_result['success']:
                    flash('Error performing data analysis', 'error')
                    return redirect(url_for('dashboard'))
                
                return render_template(
                    'train_model.html',
                    file_info=file_info,
                    eda_results=analysis_result['analysis']
                )

            # Get model recommendations
            model_recommendations = await ai_assistant.get_model_recommendations(df, task_type, target_column)
            
            if not model_recommendations['success']:
                return jsonify({
                    'success': False,
                    'error': 'Failed to get model recommendations'
                }), 500

            # Preprocess data
            X, y, preprocessing_info, encoders = preprocess_dataset(
                df, target_column, task_type
            )

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == 'classification' else None
            )

            # Calculate feature importance
            feature_importance = calculate_feature_importance(
                X, y, task_type
            )

            # Train and evaluate models
            results = {}
            if task_type == 'classification':
                models = {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingClassifier(random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    metrics = {
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision': float(precision_score(y_test, y_pred, average='weighted')),
                        'recall': float(recall_score(y_test, y_pred, average='weighted')),
                        'f1': float(f1_score(y_test, y_pred, average='weighted'))
                    }
                    
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                    metrics['cv_score_mean'] = float(cv_scores.mean())
                    metrics['cv_score_std'] = float(cv_scores.std())
                    
                    results[name] = metrics
            else:
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(random_state=42)
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    metrics = {
                        'rmse': float(np.sqrt(mse)),
                        'r2': float(r2_score(y_test, y_pred))
                    }
                    
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    metrics['cv_score_mean'] = float(cv_scores.mean())
                    metrics['cv_score_std'] = float(cv_scores.std())
                    
                    results[name] = metrics

            # Save report
            report_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'file_id': file_id,
                'eda_results': analysis_result['analysis'],
                'preprocessing_info': preprocessing_info,
                'feature_importance': feature_importance,
                'results': results,
                'created_at': datetime.utcnow(),
                'status': 'completed',
                'task_type': task_type,
                'target_column': target_column
            }
            
            report_id = mongo_db.model_reports.insert_one(report_data)

            return jsonify({
                'success': True,
                'report_id': str(report_id.inserted_id),
                'results': results,
                'preprocessing_info': preprocessing_info,
                'feature_importance': feature_importance,
                'message': 'Models trained successfully.'
            })

    except Exception as e:
        app.logger.error(f"Training error: {str(e)}")
        return jsonify({'error': str(e)}), 500

from ai_assistant import AIAssistant

# Initialize AI Assistant
ai_assistant = AIAssistant()

@app.route('/ai/analyze_data/<file_id>')
@login_required
@async_route
async def ai_analyze_data(file_id):  # Remove async since Flask doesn't support it directly
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Load and analyze data
            df = pd.read_csv(file_info['filepath'])
            
            # Create dataset summary
            dataset_info = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'task_type': file_info['task_type'],
                    'target_column': file_info['target_column']
                },
                'columns': {
                    col: {
                        'type': str(df[col].dtype),
                        'unique_values': int(df[col].nunique()),
                        'missing_values': int(df[col].isnull().sum())
                    }
                    for col in df.columns
                }
            }

            # Add numeric statistics
            for col in df.select_dtypes(include=['int64', 'float64']).columns:
                stats = df[col].describe()
                dataset_info['columns'][col].update({
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                })

            # Generate analysis prompt
            prompt = f"""
            Analyze this dataset for machine learning:

            Basic Information:
            - Task Type: {file_info['task_type']}
            - Target Column: {file_info['target_column']}
            - Number of Rows: {dataset_info['basic_info']['rows']}
            - Number of Columns: {dataset_info['basic_info']['columns']}

            Column Details:
            {json.dumps(dataset_info['columns'], indent=2)}

            Provide a comprehensive analysis including:
            1. Data Quality Assessment
            2. Feature Engineering Suggestions
            3. Preprocessing Recommendations
            4. Modeling Approach
            5. Potential Challenges and Solutions

            Format the response with markdown headings and bullet points.
            """

            try:
                # Get Gemini response
                response = model.generate_content(prompt)
                
                return jsonify({
                    'success': True,
                    'analysis': response.text
                })

            except Exception as e:
                current_app.logger.error(f"Gemini API error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error generating AI analysis'
                }), 500

    except Exception as e:
        current_app.logger.error(f"AI analysis error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
        pass

@app.route('/ai/model_insights/<report_id>')
@login_required
@async_route
async def get_model_insights(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Structure the model results for analysis
            model_summary = {
                'task_type': report['task_type'],
                'target_column': report['target_column'],
                'models': report['results'],
                'preprocessing': report['preprocessing_info'],
                'feature_importance': report.get('feature_importance', {})
            }

            # Generate prompt for model analysis
            prompt = f"""
            Analyze these machine learning model results:

            Task Type: {model_summary['task_type']}
            Target Column: {model_summary['target_column']}

            Model Performance:
            {json.dumps(model_summary['models'], indent=2)}

            Preprocessing Steps:
            {json.dumps(model_summary['preprocessing'], indent=2)}

            Feature Importance:
            {json.dumps(model_summary['feature_importance'], indent=2)}

            Provide a comprehensive analysis including:
            1. Model Performance Comparison
            2. Key Insights about Feature Importance
            3. Potential Areas for Improvement
            4. Cross-validation Stability Analysis
            5. Recommendations for Model Optimization

            Format your response with clear sections using markdown.
            """

            try:
                # Get Gemini response
                response = model.generate_content(prompt)
                
                return jsonify({
                    'success': True,
                    'insights': response.text
                })

            except Exception as e:
                current_app.logger.error(f"Gemini API error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error generating model insights'
                }), 500

    except Exception as e:
        current_app.logger.error(f"Model insights error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
        pass

@app.route('/ai/feature_recommendations/<file_id>')
@login_required
@async_route
async def get_feature_recommendations(file_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Load data
            df = pd.read_csv(file_info['filepath'])
            
            # Analyze dataset characteristics
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            # Calculate correlations for numeric columns
            correlations = {}
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                correlations = corr_matrix.to_dict()

            # Create dataset summary
            dataset_summary = {
                'task_type': file_info['task_type'],
                'target_column': file_info['target_column'],
                'numeric_features': numeric_cols,
                'categorical_features': categorical_cols,
                'correlations': correlations
            }

            # Generate recommendations prompt
            prompt = f"""
            Generate feature engineering recommendations for this dataset:

            Dataset Summary:
            - Task Type: {dataset_summary['task_type']}
            - Target Column: {dataset_summary['target_column']}
            - Numeric Features: {', '.join(dataset_summary['numeric_features'])}
            - Categorical Features: {', '.join(dataset_summary['categorical_features'])}

            Given this information, provide specific feature engineering recommendations including:
            1. Feature transformations (e.g., scaling, normalization)
            2. Feature interactions that might be meaningful
            3. Dimensionality reduction if needed
            4. Handling of categorical variables
            5. Creation of new derived features
            6. Feature selection strategies

            Format your response with clear sections and specific examples.
            """

            try:
                # Get Gemini response
                response = model.generate_content(prompt)
                
                return jsonify({
                    'success': True,
                    'recommendations': response.text
                })

            except Exception as e:
                current_app.logger.error(f"Gemini API error: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'Error generating feature recommendations'
                }), 500

    except Exception as e:
        current_app.logger.error(f"Feature recommendations error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500
        pass



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

            # Read the CSV file
            df = pd.read_csv(file_info['filepath'])
            
            # Get basic statistics
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }

            # Get past model reports
            past_reports = list(mongo_db.model_reports.find({
                "file_id": str(file_id),
                "username": current_user.username,
                "status": "completed"
            }).sort("created_at", -1))

            return render_template(
                'view_dataset.html',
                file_info=file_info,
                preview_data=df.head(10).to_dict('records'),
                columns=list(df.columns),
                stats=stats,
                shape=df.shape,
                dtypes=df.dtypes.astype(str).to_dict(),
                past_reports=past_reports
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

@app.route('/view_report/<report_id>')
@login_required
def view_report(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                flash('Report not found', 'error')
                return redirect(url_for('dashboard'))

            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(report['file_id'])
            })
            
            if not file_info:
                flash('Associated dataset not found', 'error')
                return redirect(url_for('dashboard'))

            return render_template(
                'view_report.html',
                report=report,
                file_info=file_info,
                model_results=report['results']
            )

    except Exception as e:
        app.logger.error(f"Error viewing report: {e}")
        flash('Error loading report', 'error')
        return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='127.0.0.1', port=5000)