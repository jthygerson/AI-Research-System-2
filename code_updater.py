# code_updater.py

import logging
import shutil
import os
from config import BACKUP_DIR

def update_code(rollback=False):
    if rollback:
        # Rollback code from backup
        try:
            if os.path.exists(BACKUP_DIR):
                for filename in os.listdir(BACKUP_DIR):
                    shutil.copy(os.path.join(BACKUP_DIR, filename), '.')
                logging.info("Code rollback completed.")
            else:
                logging.warning("Backup directory does not exist. Cannot rollback.")
        except Exception as e:
            logging.error(f"Error during code rollback: {e}")
    else:
        # Backup current code and apply updates
        try:
            if not os.path.exists(BACKUP_DIR):
                os.makedirs(BACKUP_DIR)
            for filename in os.listdir('.'):
                if filename.endswith('.py'):
                    shutil.copy(filename, BACKUP_DIR)
            # Apply code updates here (e.g., from suggestions in self_optimization.py)
            # For safety, this is left as a placeholder
            logging.info("Code updated successfully.")
        except Exception as e:
            logging.error(f"Error during code update: {e}")

