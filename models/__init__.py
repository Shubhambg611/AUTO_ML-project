from .ml_pipeline import MLPipeline
from .data_processor import DataProcessor

__all__ = [
    'MLPipeline',
    'DataProcessor',
]

# Version information
__version__ = '1.0.0'

# Package metadata
__author__ = 'Your Name'
__email__ = 'your.email@example.com'
__description__ = 'AutoML Models Package'

# Configuration for logging
import logging

# Create a logger for the models package
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not already exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)