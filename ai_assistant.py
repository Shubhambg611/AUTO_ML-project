import google.generativeai as genai
from typing import Dict, Any, Optional
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Configure Gemini
GOOGLE_API_KEY = 'AIzaSyCMemd6wrMxIzEsbhbYajJY0-ee5wXBrcw'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

class AIAssistant:
    def __init__(self):
        self.model = model
    
    async def analyze_data(self, df: pd.DataFrame, target_column: str, task_type: str) -> Dict:
        """Perform comprehensive data analysis"""
        try:
            dataset_info = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'task_type': task_type,
                    'target_column': target_column
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

            prompt = f"""
            Analyze this dataset for machine learning:

            Basic Information:
            - Task Type: {task_type}
            - Target Column: {target_column}
            - Number of Rows: {dataset_info['basic_info']['rows']}
            - Number of Columns: {dataset_info['basic_info']['columns']}

            Column Details:
            {json.dumps(dataset_info['columns'], indent=2)}

            Provide:
            1. Data Quality Assessment
            2. Feature Engineering Suggestions
            3. Preprocessing Recommendations
            4. Modeling Approach
            5. Potential Challenges

            Format the response with markdown headings and bullet points.
            """
            
            response = self.model.generate_content(prompt)
            return {'success': True, 'analysis': response.text}

        except Exception as e:
            logging.error(f"Data analysis error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_model_recommendations(self, df: pd.DataFrame, task_type: str, target_column: str) -> Dict:
        """Get AI recommendations for model selection and hyperparameters"""
        try:
            data_summary = {
                'shape': df.shape,
                'numeric_features': len(df.select_dtypes(include=['int64', 'float64']).columns),
                'categorical_features': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum(),
                'target_distribution': df[target_column].value_counts().to_dict() if task_type == 'classification' else {
                    'mean': float(df[target_column].mean()),
                    'std': float(df[target_column].std())
                }
            }

            prompt = f"""
            As an ML expert, recommend the best 5 models and their hyperparameters for this dataset:

            Task Type: {task_type}
            Dataset Info: {json.dumps(data_summary, indent=2)}

            For {task_type}, suggest:
            1. 3 {'traditional' if task_type == 'regression' else 'basic'} algorithms
            2. 2 ensemble learning methods

            For each model provide:
            1. Model name and rationale
            2. Optimal hyperparameters
            3. Expected performance characteristics
            4. Potential challenges

            Return in JSON format:
            {{
                "models": [
                    {{
                        "name": "model_name",
                        "type": "traditional/ensemble",
                        "hyperparameters": {{}},
                        "rationale": "",
                        "expected_performance": ""
                    }}
                ]
            }}
            """

            response = self.model.generate_content(prompt)
            recommendations = json.loads(response.text)
            return {'success': True, 'recommendations': recommendations}

        except Exception as e:
            logging.error(f"Model recommendation error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_model_insights(self, results: Dict, task_type: str) -> Dict:
        """Get insights about model performance"""
        try:
            prompt = f"""
            Analyze these model results for a {task_type} task:
            {json.dumps(results, indent=2)}

            Provide insights about:
            1. Model performance comparison
            2. Areas for improvement
            3. Feature importance analysis
            4. Optimization suggestions

            Format the response with markdown headings and bullet points.
            """

            response = self.model.generate_content(prompt)
            return {'success': True, 'insights': response.text}

        except Exception as e:
            logging.error(f"Model insights error: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def get_feature_recommendations(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Get feature engineering recommendations"""
        try:
            correlations = {}
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                correlations = corr_matrix.to_dict()

            df_info = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'correlations': correlations,
                'target_column': target_column
            }

            prompt = f"""
            Based on this dataset information:
            {json.dumps(df_info, indent=2)}

            Suggest:
            1. Feature transformations
            2. Feature interactions
            3. Feature selection approaches
            4. New features that could be created

            Format the response with markdown headings and bullet points.
            """

            response = self.model.generate_content(prompt)
            return {'success': True, 'recommendations': response.text}

        except Exception as e:
            logging.error(f"Feature recommendations error: {str(e)}")
            return {'success': False, 'error': str(e)}