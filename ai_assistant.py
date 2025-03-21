from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Debugging GPU usage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AIAssistant:
    def __init__(self, api_url=None):
        """Initialize AI Assistant using LM Studio API."""
        self.api_url = api_url or os.getenv('API_URL', 'http://localhost:1234/v1')
        self.headers = {"Content-Type": "application/json"}

    def generate_response(self, prompt: str, max_tokens: int = 300):
        """Generate response using a local LLM via LM Studio API."""
        data = {
            "model": "gemma-1.1-2b-it",  # Example: "mistralai/Mistral-7B"
            "prompt": prompt,
            "max_tokens": max_tokens
        }

        # Log the URL being used
        logging.info(f"Sending request to API: {self.api_url}")

        response = requests.post(f"{self.api_url}/completions", headers=self.headers, json=data)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"].strip()
        else:
            return f"Error: {response.json()}"


    async def analyze_data(self, df, target_column: str, task_type: str):
        """Perform local AI-based dataset analysis"""
        try:
            # Extract dataset info
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

            # Generate a structured prompt
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
            response = self.generate_response(prompt)
            return {'success': True, 'analysis': response}

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

            Return in JSON format.
            """

            response = self.generate_response(prompt)
            recommendations = json.loads(response)  # Parse JSON response
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
            - Model performance comparison
            - Areas for improvement
            - Feature importance analysis
            - Optimization suggestions

            Format the response with markdown headings and bullet points.
            """

            response = self.generate_response(prompt)
            return {'success': True, 'insights': response}

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
            - Feature transformations
            - Feature interactions
            - Feature selection approaches
            - New features that could be created

            Format the response with markdown headings and bullet points.
            """

            response = self.generate_response(prompt)
            return {'success': True, 'recommendations': response}

        except Exception as e:
            logging.error(f"Feature recommendations error: {str(e)}")
            return {'success': False, 'error': str(e)}