# utils/user_friendly_utils.py

class UserFriendlyWrapper:
    """Wrapper class to add user-friendly explanations to existing functionality"""
    
    def __init__(self, ml_pipeline):
        self.pipeline = ml_pipeline
        self.current_stage = None
        self.progress = 0
        
    def update_progress(self, stage: str, message: str, progress: float):
        """Update progress without interrupting the main pipeline"""
        self.current_stage = stage
        self.progress = progress
        return {
            'stage': stage,
            'message': self._get_friendly_message(stage),
            'progress': progress
        }
    
    def _get_friendly_message(self, stage: str) -> str:
        """Get user-friendly explanations without modifying core functionality"""
        friendly_messages = {
            'data_loading': 'Reading your data and checking its quality...',
            'preprocessing': 'Cleaning and organizing your data...',
            'model_selection': 'Finding the best AI model for your needs...',
            'training': 'Teaching the AI models using your data...',
            'evaluation': 'Checking how well the models learned...'
        }
        return friendly_messages.get(stage, '')
    
    def wrap_pipeline_result(self, result: dict) -> dict:
        """Wrap pipeline results with user-friendly explanations"""
        if not result.get('success'):
            return self._handle_error(result.get('error'))
            
        return {
            **result,  # Preserve original results
            'friendly_explanation': self._interpret_results(result),
            'visualization_data': self._prepare_visualization_data(result)
        }
    
    def _handle_error(self, error: str) -> dict:
        """Convert technical errors to user-friendly messages"""
        error_dict = {
            'ValueError: missing values': {
                'friendly': 'Some important information is missing in your data',
                'suggestion': 'Please check your data for empty cells'
            },
            'MemoryError': {
                'friendly': 'Your dataset is too large for processing',
                'suggestion': 'Try reducing the number of columns or rows'
            }
        }
        
        for error_type, message in error_dict.items():
            if error_type in str(error):
                return {
                    'success': False,
                    'error': error,
                    'friendly_error': message
                }
        
        return {
            'success': False,
            'error': error,
            'friendly_error': {
                'friendly': 'An unexpected error occurred',
                'suggestion': 'Please try again or contact support'
            }
        }