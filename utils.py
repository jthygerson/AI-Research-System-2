# utils.py

import logging
import os
from datetime import datetime

# Try to import from config, but use default values if import fails
try:
    from config import LOG_DIR, REPORTS_DIR
except ImportError:
    LOG_DIR = 'logs'
    REPORTS_DIR = 'reports'

def initialize_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, f'ai_research_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log to console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

def save_report(idea, content):
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    filename = f"{idea[:50].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, 'w') as f:
        f.write(content)
    logging.info(f"Report saved to {filepath}")

def parse_experiment_plan(experiment_plan):
    # A simple parser to extract parameters from the experiment plan
    parameters = {}
    sections = experiment_plan.split('\n\n')
    for section in sections:
        if ':' in section:
            title, content = section.split(':', 1)
            key = title.strip().lower().replace(' ', '_')
            parameters[key] = content.strip()
    return parameters

def get_latest_log_file():
    log_files = [f for f in os.listdir(LOG_DIR) if f.startswith('ai_research_') and f.endswith('.log')]
    if not log_files:
        return None
    return os.path.join(LOG_DIR, max(log_files))
