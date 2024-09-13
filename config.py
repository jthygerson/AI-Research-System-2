# config.py

import os

# General settings
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

# Other parameters can be added as needed
