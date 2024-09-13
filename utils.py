# utils.py

import logging
import os
from datetime import datetime
from config import LOG_DIR, REPORTS_DIR

def initialize_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'system.log'),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
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
