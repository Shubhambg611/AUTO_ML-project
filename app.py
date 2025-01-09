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
import joblib
from flask import jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score
import optuna
from typing import Dict, List, Any, Optional, Tuple
import logging
import google.generativeai as genai
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    mean_squared_error,
    mean_absolute_error,  # Added this
    r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import numpy as np
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.metrics import make_scorer, accuracy_score, f1_score, mean_squared_error, r2_score
import optuna
from typing import Dict, List, Any, Optional, Tuple
import logging
import google.generativeai as genai
from datetime import datetime

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
class RobustClassificationPipeline:
    def __init__(self, df, target_column='Failure Type'):
        self.df = df.copy()
        self.target_column = target_column
        self.label_encoders = {}
        self.logger = logging.getLogger(__name__)
        
    def preprocess_data(self):
        """Preprocess data with robust type handling"""
        try:
            # 1. Handle mixed data types
            for column in self.df.columns:
                if column == self.target_column:
                    continue
                    
                # Check if column contains machine IDs (starts with 'M')
                if self.df[column].dtype == 'object' and self.df[column].str.startswith('M').any():
                    # Encode machine IDs as categories
                    self.label_encoders[column] = LabelEncoder()
                    self.df[column] = self.label_encoders[column].fit_transform(self.df[column])
                    self.logger.info(f"Encoded machine IDs in column: {column}")
                    
                elif self.df[column].dtype == 'object':
                    try:
                        # Try converting to numeric if possible
                        self.df[column] = pd.to_numeric(self.df[column], errors='raise')
                        self.logger.info(f"Converted {column} to numeric")
                    except ValueError:
                        # If conversion fails, use label encoding
                        self.label_encoders[column] = LabelEncoder()
                        self.df[column] = self.label_encoders[column].fit_transform(self.df[column])
                        self.logger.info(f"Label encoded column: {column}")
            
            # 2. Handle missing values
            numeric_columns = self.df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                self.df[col] = self.df[col].fillna(self.df[col].median())
            
            # 3. Prepare features and target
            X = self.df.drop(columns=[self.target_column])
            y = self.df[self.target_column]
            
            # 4. Encode target variable
            if y.dtype == 'object':
                self.label_encoders['target'] = LabelEncoder()
                y = self.label_encoders['target'].fit_transform(y)
            
            # 5. Scale numeric features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            
            return X_scaled, y
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {str(e)}")
            raise

    def train_model(self):
        """Train classification model with error handling"""
        try:
            # 1. Preprocess data
            X, y = self.preprocess_data()
            
            # 2. Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 3. Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # 4. Get predictions
            y_pred = model.predict(X_test)
            
            # 5. Calculate metrics
            metrics = {
                'train_score': model.score(X_train, y_train),
                'test_score': model.score(X_test, y_test),
                'feature_importance': dict(zip(X.columns, model.feature_importances_))
            }
            
            # 6. Save encoders for prediction pipeline
            self.model = model
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            raise

def process_maintenance_data(filepath):
    """Process predictive maintenance dataset"""
    try:
        # 1. Load data with proper encoding
        df = pd.read_csv(filepath)
        
        # 2. Validate required columns
        required_columns = ['Failure Type']  # Add other required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # 3. Create and run pipeline
        pipeline = RobustClassificationPipeline(df)
        results = pipeline.train_model()
        
        return {
            'success': True,
            'model': pipeline.model,
            'encoders': pipeline.label_encoders,
            'metrics': results
        }
        
    except Exception as e:
        logging.error(f"Failed to process maintenance data: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
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


class EnhancedModelTraining:
    def __init__(self, api_key: str):
        self.genai = genai
        self.genai.configure(api_key=api_key)
        self.model = self.genai.GenerativeModel('gemini-pro')
        self.logger = logging.getLogger(__name__)
        self.best_preprocessing = None
        self.feature_importance = None

    async def get_optimal_preprocessing(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Get preprocessing recommendations from Gemini."""
        try:
            data_summary = {
                'shape': df.shape,
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'numeric_features': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                'categorical_features': df.select_dtypes(include=['object']).columns.tolist()
            }

            prompt = f"""
            Analyze this dataset and recommend optimal preprocessing steps:
            {data_summary}

            Provide specific recommendations for:
            1. Handling missing values
            2. Feature scaling method
            3. Encoding categorical variables
            4. Feature selection criteria
            5. Handling outliers

            Return in JSON format:
            {{
                "missing_values": {{"method": "", "columns": []}},
                "scaling": {{"method": "", "columns": []}},
                "encoding": {{"method": "", "columns": []}},
                "feature_selection": {{"method": "", "n_features": 0}},
                "outlier_treatment": {{"method": "", "columns": []}}
            }}
            """

            response = self.model.generate_content(prompt)
            recommendations = eval(response.text)
            return recommendations

        except Exception as e:
            self.logger.error(f"Error getting preprocessing recommendations: {str(e)}")
            return self._get_default_preprocessing(df)

    def _get_default_preprocessing(self, df: pd.DataFrame) -> Dict:
        """Fallback preprocessing recommendations."""
        return {
            'missing_values': {
                'method': 'mean',
                'columns': df.columns[df.isnull().any()].tolist()
            },
            'scaling': {
                'method': 'standard',
                'columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            },
            'encoding': {
                'method': 'label',
                'columns': df.select_dtypes(include=['object']).columns.tolist()
            },
            'feature_selection': {
                'method': 'mutual_info',
                'n_features': min(df.shape[1] - 1, 20)
            },
            'outlier_treatment': {
                'method': 'clip',
                'columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            }
        }

    def preprocess_data(self, df: pd.DataFrame, target_column: str, preprocessing_config: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Apply preprocessing based on recommendations."""
        df_processed = df.copy()
        preprocessing_info = {'steps': []}

        # Handle missing values
        for col in preprocessing_config['missing_values']['columns']:
            if preprocessing_config['missing_values']['method'] == 'mean':
                df_processed[col].fillna(df_processed[col].mean(), inplace=True)
            elif preprocessing_config['missing_values']['method'] == 'median':
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            preprocessing_info['steps'].append(f"Filled missing values in {col}")

        # Scale features
        numeric_cols = preprocessing_config['scaling']['columns']
        if preprocessing_config['scaling']['method'] == 'standard':
            scaler = StandardScaler()
        elif preprocessing_config['scaling']['method'] == 'robust':
            scaler = RobustScaler()
        elif preprocessing_config['scaling']['method'] == 'power':
            scaler = PowerTransformer()
        
        if numeric_cols:
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            preprocessing_info['steps'].append(f"Applied {preprocessing_config['scaling']['method']} scaling")

        # Handle outliers
        if preprocessing_config['outlier_treatment']['method'] == 'clip':
            for col in preprocessing_config['outlier_treatment']['columns']:
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                df_processed[col] = df_processed[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
                preprocessing_info['steps'].append(f"Clipped outliers in {col}")

        return df_processed, preprocessing_info

    async def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, task_type: str) -> Dict:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'random_forest': {
                    'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10)
                },
                'xgboost': {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0)
                },
                'lightgbm': {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.6, 1.0)
                }
            }

            scores = []
            for model_name, model_params in params.items():
                if task_type == 'classification':
                    model = self._get_classifier(model_name, model_params)
                    score = cross_val_score(model, X, y, cv=5, scoring='f1_weighted').mean()
                else:
                    model = self._get_regressor(model_name, model_params)
                    score = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'optimization_history': [
                {'trial': t.number, 'score': t.value}
                for t in study.trials
            ]
        }

    def _get_classifier(self, name: str, params: Dict):
        """Get classifier with specified parameters."""
        if name == 'random_forest':
            return RandomForestClassifier(**params)
        elif name == 'xgboost':
            return XGBClassifier(**params)
        elif name == 'lightgbm':
            return LGBMClassifier(**params)

    def _get_regressor(self, name: str, params: Dict):
        """Get regressor with specified parameters."""
        if name == 'random_forest':
            return RandomForestRegressor(**params)
        elif name == 'xgboost':
            return XGBRegressor(**params)
        elif name == 'lightgbm':
            return LGBMRegressor(**params)

    def select_features(self, X: pd.DataFrame, y: pd.Series, task_type: str, n_features: int) -> List[str]:
        """Select most important features."""
        if task_type == 'classification':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            selector = SelectKBest(score_func=mutual_info_regression, k=n_features)

        selector.fit(X, y)
        feature_scores = dict(zip(X.columns, selector.scores_))
        self.feature_importance = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [feat for feat, _ in self.feature_importance[:n_features]]

    async def train_models(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict:
        """Enhanced model training pipeline."""
        try:
            # Get preprocessing recommendations
            preprocessing_config = await self.get_optimal_preprocessing(df, target_column)
            
            # Preprocess data
            df_processed, preprocessing_info = self.preprocess_data(df, target_column, preprocessing_config)
            
            # Prepare features and target
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
            
            # Select features
            selected_features = self.select_features(
                X, y, task_type, 
                preprocessing_config['feature_selection']['n_features']
            )
            X = X[selected_features]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Optimize hyperparameters
            best_params = await self.optimize_hyperparameters(X_train, y_train, task_type)
            
            # Train models with optimized parameters
            models = {}
            metrics = {}
            
            for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                if task_type == 'classification':
                    model = self._get_classifier(model_name, best_params['best_params'])
                else:
                    model = self._get_regressor(model_name, best_params['best_params'])
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                models[model_name] = model
                metrics[model_name] = self._calculate_metrics(y_test, y_pred, task_type)

            return {
                'models': models,
                'metrics': metrics,
                'preprocessing_info': preprocessing_info,
                'feature_importance': self.feature_importance,
                'selected_features': selected_features,
                'hyperparameter_optimization': best_params
            }

        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, task_type: str) -> Dict:
        """Calculate performance metrics."""
        if task_type == 'classification':
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted')),
                'predictions': y_pred.tolist()
            }
        else:
            return {
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2_score': float(r2_score(y_true, y_pred)),
                'predictions': y_pred.tolist()
            }
            
class GeminiModelOptimizer:
    def __init__(self, model, genai_key):
        self.genai = genai
        self.genai.configure(api_key=genai_key)
        self.model = self.genai.GenerativeModel('gemini-pro')
        
    async def get_optimal_params(self, df, target_column, task_type, model_name):
        """Get optimal hyperparameters from Gemini API"""
        data_summary = {
            'shape': df.shape,
            'target_stats': df[target_column].describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'correlations': df.corr()[target_column].to_dict()
        }
        
        prompt = f"""
        Given this dataset information:
        {json.dumps(data_summary, indent=2)}
        
        Task Type: {task_type}
        Model: {model_name}
        
        Suggest optimal hyperparameters for best performance.
        Consider:
        1. Dataset characteristics
        2. Target variable distribution
        3. Feature correlations
        4. Model-specific requirements
        
        Return ONLY a JSON object with parameter names and values. Example:
        {{
            "n_estimators": 200,
            "max_depth": 10,
            "learning_rate": 0.1
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            params = json.loads(response.text)
            return params
        except Exception as e:
            # Fallback to default parameters on error
            if model_name == 'random_forest':
                return {'n_estimators': 200, 'max_depth': 15}
            elif model_name == 'gradient_boosting':
                return {'n_estimators': 150, 'learning_rate': 0.1}
            elif model_name == 'xgboost':
                return {'n_estimators': 200, 'max_depth': 12, 'learning_rate': 0.1}
            elif model_name == 'lightgbm':
                return {'n_estimators': 200, 'num_leaves': 31}
            else:
                return {'iterations': 200, 'depth': 10}

    async def get_feature_engineering_steps(self, df, target_column):
        """Get feature engineering recommendations"""
        data_info = {
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': df.describe().to_dict(),
        }
        
        prompt = f"""
        Analyze this dataset and suggest feature engineering steps:
        {json.dumps(data_info, indent=2)}
        
        Target Column: {target_column}
        
        Provide ONLY a JSON object with engineering steps. Example:
        {{
            "interactions": [["feat1", "feat2"]],
            "polynomials": ["feat1", "feat2"],
            "transformations": {{"feat1": "log", "feat2": "sqrt"}},
            "bin_features": ["feat3"],
            "groupby_aggregations": [
                {{"group": "feat4", "agg_column": "feat5", "aggs": ["mean", "std"]}}
            ]
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            steps = json.loads(response.text)
            return steps
        except Exception as e:
            return {
                'interactions': [],
                'polynomials': [],
                'transformations': {},
                'bin_features': [],
                'groupby_aggregations': []
            }

    def apply_feature_engineering(self, df, steps):
        """Apply recommended feature engineering steps"""
        df_new = df.copy()
        
        # Create interaction features
        for feat1, feat2 in steps.get('interactions', []):
            if feat1 in df.columns and feat2 in df.columns:
                df_new[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
        
        # Create polynomial features
        for feat in steps.get('polynomials', []):
            if feat in df.columns:
                df_new[f'{feat}_squared'] = df[feat] ** 2
        
        # Apply transformations
        for feat, transform in steps.get('transformations', {}).items():
            if feat in df.columns:
                if transform == 'log':
                    df_new[f'{feat}_log'] = np.log1p(df[feat])
                elif transform == 'sqrt':
                    df_new[f'{feat}_sqrt'] = np.sqrt(df[feat])
        
        # Create bins
        for feat in steps.get('bin_features', []):
            if feat in df.columns:
                df_new[f'{feat}_bins'] = pd.qcut(df[feat], q=5, labels=False, duplicates='drop')
        
        # Groupby aggregations
        for agg in steps.get('groupby_aggregations', []):
            group_col = agg['group']
            agg_col = agg['agg_column']
            if group_col in df.columns and agg_col in df.columns:
                for agg_func in agg['aggs']:
                    col_name = f'{group_col}_{agg_col}_{agg_func}'
                    df_new[col_name] = df.groupby(group_col)[agg_col].transform(agg_func)
        
        return df_new
            
            
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
    """
    Route for training machine learning models on uploaded datasets.
    Handles both GET (display training page) and POST (perform training) requests.
    """
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            # Verify file exists and belongs to user
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(file_id),
                "username": current_user.username
            })
            
            if not file_info:
                return jsonify({'error': 'File not found'}), 404

            # Load data
            df = pd.read_csv(file_info['filepath'])
            target_column = file_info['target_column']
            
            # Auto-detect task type based on target column
            is_categorical = df[target_column].dtype == 'object' or df[target_column].nunique() < 10
            task_type = 'classification' if is_categorical else 'regression'
            
            # Update task type in database
            mongo_db.uploads.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {"task_type": task_type}}
            )
            file_info['task_type'] = task_type

            # Handle GET request - display training page
            if request.method == 'GET':
                analysis_result = await ai_assistant.analyze_data(df, target_column, task_type)
                return render_template(
                    'train_model.html',
                    file_info=file_info,
                    eda_results=analysis_result['analysis']
                )

            # Start training process
            app.logger.info(f"Starting model training for file {file_id}")
            training_start_time = datetime.utcnow()

            # Initialize preprocessing info
            preprocessing_info = {'steps_taken': []}
            data = df.copy()
            categorical_encoders = {}

            # Handle machine IDs and categorical variables
            for column in data.columns:
                if column != target_column:
                    if data[column].dtype == 'object':
                        if data[column].str.startswith('M').any():
                            # Handle machine IDs
                            le = LabelEncoder()
                            data[column] = le.fit_transform(data[column])
                            categorical_encoders[column] = le
                            preprocessing_info['steps_taken'].append(f"Encoded machine IDs in {column}")
                        else:
                            # Handle other categorical variables
                            le = LabelEncoder()
                            data[column] = le.fit_transform(data[column])
                            categorical_encoders[column] = le
                            preprocessing_info['steps_taken'].append(f"Label encoded {column}")

            # Handle missing values
            for column in data.columns:
                missing_count = data[column].isnull().sum()
                if missing_count > 0:
                    if data[column].dtype in ['int64', 'float64']:
                        data[column].fillna(data[column].median(), inplace=True)
                        preprocessing_info['steps_taken'].append(f"Filled missing values in {column} with median")
                    else:
                        data[column].fillna(data[column].mode()[0], inplace=True)
                        preprocessing_info['steps_taken'].append(f"Filled missing values in {column} with mode")

            # Prepare features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Encode target for classification
            if task_type == 'classification':
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
                categorical_encoders['target'] = target_encoder

            # Scale numeric features
            numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_columns) > 0:
                scaler = StandardScaler()
                X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
                preprocessing_info['steps_taken'].append("Scaled numeric features")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if task_type == 'classification' else None
            )

            # Initialize models based on task type
            if task_type == 'classification':
                models = {
                    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingClassifier(random_state=42),
                    'xgboost': XGBClassifier(random_state=42),
                    'lightgbm': LGBMClassifier(random_state=42),
                    'catboost': CatBoostClassifier(random_state=42, verbose=False)
                }
            else:
                models = {
                    'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'gradient_boosting': GradientBoostingRegressor(random_state=42),
                    'xgboost': XGBRegressor(random_state=42),
                    'lightgbm': LGBMRegressor(random_state=42),
                    'catboost': CatBoostRegressor(random_state=42, verbose=False)
                }

            # Train models and collect results
            results = {}
            fold_scores = []
            fold_precisions = []

            for name, model in models.items():
                try:
                    app.logger.info(f"Training {name} model...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    if task_type == 'classification':
                        metrics = {
                            'accuracy': float(accuracy_score(y_test, y_pred)),
                            'precision': float(precision_score(y_test, y_pred, average='weighted')),
                            'recall': float(recall_score(y_test, y_pred, average='weighted')),
                            'f1': float(f1_score(y_test, y_pred, average='weighted'))
                        }
                        
                        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
                        cv_precisions = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision_weighted')
                    else:
                        metrics = {
                            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            'mae': float(mean_absolute_error(y_test, y_pred)),
                            'r2': float(r2_score(y_test, y_pred))
                        }
                        
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
                    
                    metrics['cv_score_mean'] = float(cv_scores.mean())
                    metrics['cv_score_std'] = float(cv_scores.std())
                    
                    results[name] = metrics
                    fold_scores.extend(cv_scores)
                    if task_type == 'classification':
                        fold_precisions.extend(cv_precisions)
                    
                    app.logger.info(f"Successfully trained {name} model")
                    
                except Exception as model_error:
                    app.logger.error(f"Error training {name} model: {str(model_error)}")
                    continue

            if not results:
                raise Exception("All models failed to train")

            # Calculate feature importance
            feature_importance = calculate_feature_importance(X, y, task_type)

            # Find best model
            if task_type == 'classification':
                best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
            else:
                best_model_name = min(results.items(), key=lambda x: x[1]['rmse'])[0]

            # Save best model file
            model_filepath = os.path.join(
                app.config['MODELS_FOLDER'],
                f"{file_id}_{best_model_name}.joblib"
            )
            joblib.dump(models[best_model_name], model_filepath)

            # Create performance timeline
            if task_type == 'classification':
                performance_timeline = [
                    {
                        'name': f'Fold {i+1}',
                        'accuracy': float(fold_scores[i]),
                        'precision': float(fold_precisions[i])
                    }
                    for i in range(len(fold_scores))
                ]
            else:
                performance_timeline = [
                    {
                        'name': f'Fold {i+1}',
                        'rmse': float(fold_scores[i])
                    }
                    for i in range(len(fold_scores))
                ]

            # Calculate training time
            training_end_time = datetime.utcnow()
            training_duration = (training_end_time - training_start_time).total_seconds()

            # Create dashboard data structure
            dashboard_data = {
                'best_model': {
                    'name': best_model_name,
                    'metrics': results[best_model_name],
                    'parameters': models[best_model_name].get_params()
                },
                'feature_count': len(X.columns),
                'training_time': training_duration,
                'cv_score': float(results[best_model_name]['cv_score_mean']),
                'model_comparison': [
                    {
                        'name': model_name,
                        'metrics': metrics
                    }
                    for model_name, metrics in results.items()
                ],
                'performance_timeline': performance_timeline,
                'feature_importance': [
                    {'feature': feat, 'importance': float(score)}
                    for feat, score in feature_importance['aggregate_ranks'].items()
                ]
            }

            # Create report data
            report_data = {
                'user_id': current_user.id,
                'username': current_user.username,
                'file_id': file_id,
                'preprocessing_info': preprocessing_info,
                'feature_importance': feature_importance,
                'results': results,
                'model_filepath': model_filepath,
                'created_at': training_start_time,
                'completed_at': training_end_time,
                'training_duration': training_duration,
                'status': 'completed',
                'task_type': task_type,
                'target_column': target_column,
                'dashboard_data': dashboard_data,
                'best_model': {
                    'name': best_model_name,
                    'metrics': results[best_model_name]
                }
            }

            # Save report to MongoDB
            report_id = mongo_db.model_reports.insert_one(report_data).inserted_id

            # Update file status
            mongo_db.uploads.update_one(
                {"_id": ObjectId(file_id)},
                {"$set": {
                    "status": "trained",
                    "last_training": training_end_time,
                    "best_model": best_model_name,
                    "latest_report_id": str(report_id)
                }}
            )

            return jsonify({
                'success': True,
                'report_id': str(report_id),
                'results': results,
                'dashboard_data': dashboard_data,
                'message': f'Models trained successfully. Best model: {best_model_name}'
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
    
@app.route('/insights/<report_id>')
@login_required
@async_route
async def get_training_insights(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Format training results for analysis
            analysis_data = {
                'best_model': report['best_model'],
                'feature_importance': report['feature_importance'],
                'preprocessing_steps': report['preprocessing_info']['steps'],
                'model_metrics': report['metrics'],
                'training_duration': report['training_duration'],
                'optimization_history': report['hyperparameter_optimization']['optimization_history']
            }

            # Generate prompt for Gemini
            prompt = f"""
            Analyze these machine learning training results and provide insights:

            Training Results:
            {json.dumps(analysis_data, indent=2)}

            Provide insights about:
            1. Model Performance Analysis
            2. Feature Importance Interpretation
            3. Preprocessing Effectiveness
            4. Optimization Results
            5. Recommendations for Improvement

            Format your response with markdown.
            """

            # Get AI insights
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            
            # Save insights to MongoDB
            mongo_db.model_reports.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {
                    "ai_insights": response.text,
                    "insights_generated_at": datetime.utcnow()
                }}
            )

            return jsonify({
                'success': True,
                'insights': response.text
            })

    except Exception as e:
        app.logger.error(f"Error generating insights: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
    
# Add these new routes to app.py

@app.route('/deploy/<report_id>', methods=['POST'])
@login_required
def deploy_model(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Get the best performing model
            best_model = None
            best_score = -1
            for model_name, metrics in report['results'].items():
                if metrics['accuracy'] > best_score:
                    best_model = model_name
                    best_score = metrics['accuracy']

            # Create deployment record
            deployment_info = {
                'report_id': str(report_id),
                'model_name': best_model,
                'username': current_user.username,
                'task_type': report['task_type'],
                'target_column': report['target_column'],
                'accuracy': float(best_score),
                'deployment_date': datetime.utcnow(),
                'status': 'active',
                'preprocessing_info': report['preprocessing_info']
            }
            
            # Save deployment record
            deployment = mongo_db.deployments.insert_one(deployment_info)
            
            # Update report status
            mongo_db.model_reports.update_one(
                {"_id": ObjectId(report_id)},
                {"$set": {"deployment_id": str(deployment.inserted_id)}}
            )

            return jsonify({
                'success': True,
                'message': f'Model {best_model} deployed successfully',
                'deployment_id': str(deployment.inserted_id)
            })

    except Exception as e:
        app.logger.error(f"Deployment error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/analyze/quality/<file_id>')
@login_required
def analyze_data_quality(file_id):
    """Analyze data quality"""
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
            
            # Calculate quality scores
            quality_scores = calculate_data_quality_score(df)
            
            return jsonify(quality_scores)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze/errors/<report_id>')
@login_required
def analyze_prediction_errors(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Get original file
            file_info = mongo_db.uploads.find_one({
                "_id": ObjectId(report['file_id'])
            })
            
            if not file_info:
                return jsonify({'error': 'Original file not found'}), 404

            # Load data
            df = pd.read_csv(file_info['filepath'])
            
            # Calculate error metrics
            error_analysis = {
                'model_performance': report['results'],
                'error_distribution': {
                    model: {
                        'correct_predictions': metrics['accuracy'] * 100,
                        'incorrect_predictions': (1 - metrics['accuracy']) * 100
                    }
                    for model, metrics in report['results'].items()
                },
                'feature_importance': report.get('feature_importance', {})
            }
            
            return jsonify(error_analysis)

    except Exception as e:
        app.logger.error(f"Error analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/download_report/<report_id>')
@login_required
def download_report(report_id):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Generate PDF report
            report_data = {
                'task_info': {
                    'type': report['task_type'],
                    'target': report['target_column'],
                    'created': report['created_at'].strftime('%Y-%m-%d %H:%M:%S')
                },
                'model_performance': report['results'],
                'preprocessing': report['preprocessing_info'],
                'feature_importance': report.get('feature_importance', {})
            }
            
            return jsonify(report_data)

    except Exception as e:
        app.logger.error(f"Report download error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download_model/<report_id>/<model_name>')
@login_required
def download_model(report_id, model_name):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Create model download file
            model_info = {
                'model_name': model_name,
                'task_type': report['task_type'],
                'target_column': report['target_column'],
                'performance_metrics': report['results'][model_name],
                'preprocessing_info': report['preprocessing_info']
            }
            
            # Save as JSON
            model_file = f"model_{model_name}_{report_id}.json"
            model_path = os.path.join(app.config['MODELS_FOLDER'], model_file)
            
            with open(model_path, 'w') as f:
                json.dump(model_info, f, indent=4)
            
            return send_file(
                model_path,
                as_attachment=True,
                download_name=model_file
            )

    except Exception as e:
        app.logger.error(f"Model download error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/download_model_pkl/<report_id>/<model_name>')
@login_required
def download_model_pkl(report_id, model_name):
    try:
        with db_manager.get_mongo_connection() as mongo_db:
            report = mongo_db.model_reports.find_one({
                "_id": ObjectId(report_id),
                "username": current_user.username
            })
            
            if not report:
                return jsonify({'error': 'Report not found'}), 404

            # Create directory for model files if it doesn't exist
            model_dir = os.path.join(app.config['MODELS_FOLDER'], str(report_id))
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model to a .pkl file
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
            # Get model object based on model name
            model = None
            if model_name == 'random_forest':
                model = RandomForestClassifier()
            elif model_name == 'gradient_boosting':
                model = GradientBoostingClassifier()
            elif model_name == 'xgboost':
                model = XGBClassifier()
            elif model_name == 'lightgbm':
                model = LGBMClassifier()
            elif model_name == 'catboost':
                model = CatBoostClassifier()
            
            # Save model
            joblib.dump(model, model_path)
            
            return send_file(
                model_path,
                as_attachment=True,
                download_name=f"{model_name}_{report_id}.pkl",
                mimetype='application/octet-stream'
            )

    except Exception as e:
        app.logger.error(f"Model download error: {str(e)}")
        return jsonify({'error': str(e)}), 500   
     
@app.route('/ai/chat', methods=['POST'])
@login_required
async def chat():
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Generate response using Gemini
        try:
            response = model.generate_content(message)
            return jsonify({
                'success': True,
                'response': response.text
            })
        except Exception as e:
            app.logger.error(f"Gemini API error: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Error generating response'
            }), 500

    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='127.0.0.1', port=5000)