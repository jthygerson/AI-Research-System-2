# config.py

import os

# General settings (default values)
NUM_IDEAS = 5
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')  # Default to 'gpt-4o', can be overridden
MAX_ATTEMPTS = 3

# OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Directories
LOG_DIR = 'logs'
REPORTS_DIR = 'reports'
BACKUP_DIR = 'backup'

# Benchmark settings
BENCHMARK_DATASET = 'glue'
BENCHMARK_TASK = 'sst2'

def set_config_parameters(args):
    global NUM_IDEAS, MODEL_NAME, MAX_ATTEMPTS
    NUM_IDEAS = args.num_ideas
    MODEL_NAME = args.model
    MAX_ATTEMPTS = args.max_attempts

# Ensure these variables are exported
__all__ = ['NUM_IDEAS', 'MODEL_NAME', 'MAX_ATTEMPTS', 'OPENAI_API_KEY',
           'LOG_DIR', 'REPORTS_DIR', 'BACKUP_DIR', 'BENCHMARK_DATASET',
           'BENCHMARK_TASK', 'set_config_parameters']
