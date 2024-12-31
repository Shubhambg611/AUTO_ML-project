import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
import json

class DataProcessor:
    def __init__(self, filepath: Union[str, Path]):
        """Initialize DataProcessor with a filepath.
        
        Args:
            filepath: Path to the CSV file
        """
        self.filepath = Path(filepath)
        self.logger = logging.getLogger(__name__)
        self.df = None
        self.column_types = {}
        self.data_summary = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file and perform initial analysis."""
        try:
            self.df = pd.read_csv(self.filepath)
            return self.df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze(self) -> Dict:
        """Perform comprehensive data analysis.
        
        Returns:
            Dictionary containing data analysis results
        """
        try:
            if self.df is None:
                self.load_data()
            
            analysis = {
                'basic_info': self._get_basic_info(),
                'data_types': self._analyze_data_types(),
                'missing_values': self._analyze_missing_values(),
                'numeric_analysis': self._analyze_numeric_columns(),
                'categorical_analysis': self._analyze_categorical_columns(),
                'correlations': self._analyze_correlations(),
                'data_quality_issues': self._check_data_quality()
            }
            
            self.data_summary = analysis
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in data analysis: {str(e)}")
            raise

    def _get_basic_info(self) -> Dict:
        """Get basic information about the dataset."""
        return {
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum(),
            'columns': list(self.df.columns)
        }

    def _analyze_data_types(self) -> Dict:
        """Analyze and categorize data types of columns."""
        type_mapping = {
            'numeric': ['int64', 'float64'],
            'categorical': ['object', 'category', 'bool'],
            'datetime': ['datetime64[ns]']
        }
        
        column_types = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            for type_name, dtypes in type_mapping.items():
                if any(dtype.startswith(t) for t in dtypes):
                    column_types[col] = type_name
                    break
            if col not in column_types:
                column_types[col] = 'other'
        
        self.column_types = column_types
        return column_types

    def _analyze_missing_values(self) -> Dict:
        """Analyze missing values in the dataset."""
        missing_info = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_info[col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(self.df) * 100)
                }
        return missing_info

    def _analyze_numeric_columns(self) -> Dict:
        """Analyze numeric columns in the dataset."""
        numeric_analysis = {}
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        for col in numeric_cols:
            stats = self.df[col].describe()
            numeric_analysis[col] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'quartiles': {
                    '25%': float(stats['25%']),
                    '50%': float(stats['50%']),
                    '75%': float(stats['75%'])
                },
                'skewness': float(self.df[col].skew()),
                'kurtosis': float(self.df[col].kurtosis()),
                'zero_count': int((self.df[col] == 0).sum()),
                'negative_count': int((self.df[col] < 0).sum())
            }
        
        return numeric_analysis

    def _analyze_categorical_columns(self) -> Dict:
        """Analyze categorical columns in the dataset."""
        categorical_analysis = {}
        categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns
        
        for col in categorical_cols:
            value_counts = self.df[col].value_counts()
            unique_count = len(value_counts)
            
            categorical_analysis[col] = {
                'unique_values': unique_count,
                'top_values': value_counts.head(10).to_dict(),
                'value_distribution': {
                    'value': value_counts.head(10).index.tolist(),
                    'count': value_counts.head(10).tolist()
                }
            }
        
        return categorical_analysis

    def _analyze_correlations(self) -> Dict:
        """Analyze correlations between numeric columns."""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            return {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': self._get_high_correlations(corr_matrix)
            }
        return {}

    def _get_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict]:
        """Get pairs of highly correlated features."""
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        return high_corr

    def _check_data_quality(self) -> Dict:
        """Check for various data quality issues."""
        quality_issues = {
            'duplicate_rows': int(self.df.duplicated().sum()),
            'constant_columns': [],
            'high_cardinality_columns': [],
            'suspicious_patterns': []
        }
        
        # Check for constant columns
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                quality_issues['constant_columns'].append(col)
        
        # Check for high cardinality in categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if unique_ratio > 0.9:  # More than 90% unique values
                quality_issues['high_cardinality_columns'].append({
                    'column': col,
                    'unique_ratio': float(unique_ratio)
                })
        
        return quality_issues

    def suggest_target_column(self) -> List[Dict]:
        """Suggest potential target columns based on data characteristics."""
        suggestions = []
        
        # Check column names for common target indicators
        target_keywords = ['target', 'label', 'class', 'outcome', 'result', 
                         'price', 'salary', 'revenue', 'sales', 'profit']
        
        for col in self.df.columns:
            score = 0
            reason = []
            
            # Check column name
            if any(keyword in col.lower() for keyword in target_keywords):
                score += 2
                reason.append("Column name suggests target variable")
            
            # Check if categorical with few unique values
            if self.df[col].dtype == 'object' and self.df[col].nunique() < 10:
                score += 1
                reason.append("Categorical with few unique values")
            
            # Check if numeric with reasonable distribution
            if self.df[col].dtype in ['int64', 'float64']:
                if 0 < self.df[col].nunique() < 100:
                    score += 1
                    reason.append("Numeric with reasonable number of unique values")
            
            if score > 0:
                suggestions.append({
                    'column': col,
                    'score': score,
                    'reasons': reason,
                    'dtype': str(self.df[col].dtype),
                    'unique_values': int(self.df[col].nunique())
                })
        
        return sorted(suggestions, key=lambda x: x['score'], reverse=True)

    def get_recommended_ml_task(self, target_column: str) -> str:
        """Recommend whether the task is classification or regression."""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        col_data = self.df[target_column]
        unique_values = col_data.nunique()
        
        if col_data.dtype in ['object', 'bool', 'category']:
            return 'classification'
        elif col_data.dtype in ['int64', 'float64']:
            if unique_values < 10 or (unique_values / len(self.df) < 0.05):
                return 'classification'
            else:
                return 'regression'
        else:
            return 'unknown'

    def generate_feature_importance(self, target_column: str) -> Dict:
        """Calculate feature importance scores using mutual information."""
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Prepare feature matrix
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X_processed = X.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
        
        # Calculate mutual information scores
        if self.get_recommended_ml_task(target_column) == 'classification':
            mi_scores = mutual_info_classif(X_processed, y)
        else:
            mi_scores = mutual_info_regression(X_processed, y)
        
        # Create feature importance dictionary
        importance_dict = {
            col: float(score) 
            for col, score in zip(X.columns, mi_scores)
        }
        
        return {
            'feature_importance': importance_dict,
            'sorted_features': sorted(
                importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
        }

    def save_analysis(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """Save the data analysis results to a JSON file."""
        if not self.data_summary:
            self.analyze()
        
        if output_path is None:
            output_path = self.filepath.parent / f"{self.filepath.stem}_analysis.json"
        
        with open(output_path, 'w') as f:
            json.dump(self.data_summary, f, indent=2)
            
    def get_preprocessing_recommendations(self) -> Dict:
        """Get preprocessing recommendations based on data analysis."""
        if not self.data_summary:
            self.analyze()
            
        recommendations = {
            'missing_values': [],
            'encoding': [],
            'scaling': [],
            'feature_selection': [],
            'handling_outliers': []
        }
        
        # Missing values recommendations
        missing_info = self.data_summary['missing_values']
        for col, info in missing_info.items():
            if info['percentage'] < 5:
                recommendations['missing_values'].append({
                    'column': col,
                    'recommendation': 'Remove rows with missing values',
                    'reason': 'Low percentage of missing values'
                })
            elif info['percentage'] < 30:
                recommendations['missing_values'].append({
                    'column': col,
                    'recommendation': 'Impute with mean/mode',
                    'reason': 'Moderate percentage of missing values'
                })
            else:
                recommendations['missing_values'].append({
                    'column': col,
                    'recommendation': 'Consider dropping column',
                    'reason': 'High percentage of missing values'
                })
        
        # Encoding recommendations
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            if unique_count < 10:
                recommendations['encoding'].append({
                    'column': col,
                    'recommendation': 'One-hot encoding',
                    'reason': 'Low cardinality categorical variable'
                })
            else:
                recommendations['encoding'].append({
                    'column': col,
                    'recommendation': 'Label encoding or target encoding',
                    'reason': 'High cardinality categorical variable'
                })
        
        # Scaling recommendations
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if self.df[col].std() > 10:
                recommendations['scaling'].append({
                    'column': col,
                    'recommendation': 'Standard scaling',
                    'reason': 'High variance in features'
                })
        
        return recommendations