# code_updater.py

import logging
import shutil
import os
from config import BACKUP_DIR

PROTECTED_FILES = ['config.py']  # Add this line at the top of the file

def update_code(code_changes=None, rollback=False):
    if rollback:
        # Rollback code from backup
        try:
            if os.path.exists(BACKUP_DIR):
                for filename in os.listdir(BACKUP_DIR):
                    shutil.copy(os.path.join(BACKUP_DIR, filename), '.')
                logging.info("Code rollback completed.")
            else:
                logging.warning(f"Backup directory '{BACKUP_DIR}' does not exist. Cannot rollback.")
        except Exception as e:
            logging.error(f"Error during code rollback: {e}")
    else:
        # Backup current code and apply updates
        try:
            if not os.path.exists(BACKUP_DIR):
                os.makedirs(BACKUP_DIR)
                logging.info(f"Backup directory '{BACKUP_DIR}' created.")
            # Backup code files
            for filename in os.listdir('.'):
                if filename.endswith('.py'):
                    shutil.copy(filename, BACKUP_DIR)
            logging.info("Code backup completed.")

            # Apply code changes
            if code_changes:
                for change in code_changes:
                    filename = change['filename']
                    updated_code = change['updated_code']
                    
                    # Ensure the file is in the current directory, has a .py extension,
                    # and is not in the PROTECTED_FILES list
                    if (os.path.dirname(filename) or 
                        not filename.endswith('.py') or 
                        filename in PROTECTED_FILES):
                        logging.warning(f"Skipping invalid or protected file: {filename}")
                        continue
                    
                    with open(filename, 'w') as f:
                        f.write(updated_code)
                logging.info("Code updated successfully.")
            else:
                logging.info("No code changes to apply.")

        except Exception as e:
            logging.error(f"Error during code update: {e}")
