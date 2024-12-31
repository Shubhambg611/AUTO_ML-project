import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report,
    precision_score, recall_score, f1_score, r2_score
)
import xgboost as xgb
import lightgbm as lgb
from sklearn.impute import SimpleImputer
import joblib
import logging
from datetime import datetime
from pathlib import Path

class MLPipeline:
    def __init__(self, filepath, target_column, task_type, config=None):
        self.filepath = Path(filepath)
        self.target_column = target_column
        self.task_type = task_type
        self.config = config or {}
        self.model = None
        self.preprocessors = {}
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize best model and metrics
        self.best_model = None
        self.best_model_name = None
        self.best_metrics = None

    def _handle_missing_values(self, df, is_training=True):
        """Handle missing values in the dataset."""
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        categorical_columns = df.select_dtypes(include=['object']).columns

        # Handle numeric columns
        if is_training:
            self.preprocessors['numeric_imputer'] = SimpleImputer(strategy='mean')
            self.preprocessors['numeric_imputer'].fit(df[numeric_columns])
        
        if numeric_columns.size > 0:
            df[numeric_columns] = self.preprocessors['numeric_imputer'].transform(df[numeric_columns])

        # Handle categorical columns
        if is_training:
            self.preprocessors['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
            self.preprocessors['categorical_imputer'].fit(df[categorical_columns])
        
        if categorical_columns.size > 0:
            df[categorical_columns] = self.preprocessors['categorical_imputer'].transform(df[categorical_columns])

        return df

    def _encode_categorical(self, df, is_training=True):
        """Encode categorical variables."""
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != self.target_column:
                if is_training:
                    self.preprocessors[f'encoder_{col}'] = LabelEncoder()
                    df[col] = self.preprocessors[f'encoder_{col}'].fit_transform(df[col])
                else:
                    # Handle unseen categories for prediction
                    unique_values = self.preprocessors[f'encoder_{col}'].classes_
                    df[col] = df[col].map(lambda x: x if x in unique_values else unique_values[0])
                    df[col] = self.preprocessors[f'encoder_{col}'].transform(df[col])
        
        return df

    def _scale_features(self, df, is_training=True):
        """Scale numeric features."""
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        if is_training:
            self.preprocessors['scaler'] = StandardScaler()
            if numeric_columns.size > 0:
                df[numeric_columns] = self.preprocessors['scaler'].fit_transform(df[numeric_columns])
        else:
            if numeric_columns.size > 0:
                df[numeric_columns] = self.preprocessors['scaler'].transform(df[numeric_columns])
        
        return df

    def preprocess_data(self, df, is_training=True):
        """Preprocess the dataset for training or prediction."""
        try:
            # Store feature names if training
            if is_training:
                self.feature_names = [col for col in df.columns if col != self.target_column]

            # Make a copy to avoid modifying original data
            df = df.copy()

            # Handle missing values
            df = self._handle_missing_values(df, is_training)

            # Encode categorical variables
            df = self._encode_categorical(df, is_training)

            # Scale features
            df = self._scale_features(df, is_training)

            return df

        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise

    def _get_models(self):
        """Get dictionary of models based on task type."""
        if self.task_type == 'classification':
            return {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                )
            }
        else:
            return {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=self.config.get('random_state', 42)
                )
            }

    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model and return metrics."""
        y_pred = model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'classification_report': classification_report(y_test, y_pred)
            }
        else:
            metrics = {
                'rmse': mean_squared_error(y_test, y_pred, squared=False),
                'mse': mean_squared_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
        return metrics

    def train(self):
        """Train multiple models and return the best one with metrics."""
        try:
            # Load and preprocess data
            df = pd.read_csv(self.filepath)
            df = self.preprocess_data(df, is_training=True)

            # Prepare target variable
            if self.task_type == 'classification':
                le = LabelEncoder()
                y = le.fit_transform(df[self.target_column])
                self.preprocessors['target_encoder'] = le
            else:
                y = df[self.target_column].values

            # Prepare features
            X = df.drop(self.target_column, axis=1)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42)
            )

            # Train and evaluate all models
            models = self._get_models()
            best_score = float('-inf')
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate model
                metrics = self._evaluate_model(model, X_test, y_test)
                
                # Update best model if better
                current_score = metrics['accuracy'] if self.task_type == 'classification' else -metrics['rmse']
                if current_score > best_score:
                    best_score = current_score
                    self.best_model = model
                    self.best_model_name = name
                    self.best_metrics = metrics

            # Store the best model
            self.model = self.best_model

            return {
                'best_model': self.best_model_name,
                'metrics': self.best_metrics
            }

        except Exception as e:
            self.logger.error(f"Error in training: {str(e)}")
            raise

    def predict(self, features):
        """Make predictions on new data."""
        try:
            if not self.model:
                raise ValueError("Model not trained yet")

            # Convert features to DataFrame if necessary
            if isinstance(features, dict):
                features = pd.DataFrame([features])
            elif isinstance(features, list):
                features = pd.DataFrame(features)

            # Preprocess features
            features = self.preprocess_data(features, is_training=False)

            # Make predictions
            predictions = self.model.predict(features)

            # Decode predictions if classification
            if self.task_type == 'classification':
                predictions = self.preprocessors['target_encoder'].inverse_transform(predictions)

            return predictions

        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise

    def save(self, filepath):
        """Save the model and preprocessors."""
        try:
            model_data = {
                'model': self.model,
                'preprocessors': self.preprocessors,
                'feature_names': self.feature_names,
                'target_column': self.target_column,
                'task_type': self.task_type,
                'config': self.config,
                'best_model_name': self.best_model_name,
                'best_metrics': self.best_metrics
            }
            joblib.dump(model_data, filepath)
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load(cls, filepath):
        """Load a saved model."""
        try:
            model_data = joblib.load(filepath)
            pipeline = cls(
                filepath=None,
                target_column=model_data['target_column'],
                task_type=model_data['task_type'],
                config=model_data['config']
            )
            pipeline.model = model_data['model']
            pipeline.preprocessors = model_data['preprocessors']
            pipeline.feature_names = model_data['feature_names']
            pipeline.best_model_name = model_data.get('best_model_name')
            pipeline.best_metrics = model_data.get('best_metrics')
            return pipeline
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise