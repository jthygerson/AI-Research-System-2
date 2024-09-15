# experiment_execution.py

import os
import tempfile
import logging
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from huggingface_hub import HfFolder
from pathlib import Path
import sys

# Set cache directories to specific locations in the user's home folder
cache_dir = str(Path.home() / '.cache' / 'huggingface')
os.environ['HF_HOME'] = cache_dir
os.environ['HF_DATASETS_CACHE'] = cache_dir

# Make sure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

alternative_datasets = [
    ('glue', 'mrpc'),
    ('imdb', None),
    ('squad', None),
    ('conll2003', None)
]

def execute_experiment(parameters):
    try:
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Get Hugging Face token
        hf_token = os.environ.get('HF_API_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN') or HfFolder.get_token()
        if not hf_token:
            logging.warning("No Hugging Face token found. Some datasets may not be accessible.")
        else:
            logging.info("Hugging Face token found.")

        # Load dataset
        dataset_name = parameters.get('datasets', ['ag_news'])[0]
        logging.info(f"Attempting to load dataset: {dataset_name}")
        raw_datasets = load_dataset_with_retry(dataset_name, use_auth_token=hf_token)
        if raw_datasets is None:
            logging.info("Primary dataset load failed, trying alternatives...")
            for alt_dataset_name, config in alternative_datasets:
                logging.info(f"Attempting to load alternative dataset: {alt_dataset_name}")
                raw_datasets = load_dataset_with_retry(alt_dataset_name, use_auth_token=hf_token, config=config)
                if raw_datasets is not None:
                    logging.info(f"Successfully loaded alternative dataset: {alt_dataset_name}")
                    break
                else:
                    logging.info(f"Failed to load alternative dataset: {alt_dataset_name}")

        if raw_datasets is None:
            raise Exception("Failed to load any dataset. Please check your internet connection and dataset accessibility.")

        logging.info(f"Dataset loaded successfully. Dataset info: {raw_datasets}")

        # Tokenizer and model
        model_name = parameters.get('model_architecture', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Adjust the model for the dataset
        if 'label' in raw_datasets['train'].features:
            num_labels = len(set(raw_datasets['train']['label']))
        elif 'labels' in raw_datasets['train'].features:
            num_labels = len(set(raw_datasets['train']['labels']))
        else:
            num_labels = 2  # Default to binary classification if no label column found
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)

        # Tokenize datasets
        def tokenize_function(example):
            return tokenizer(example['text'] if 'text' in example else example['sentence'], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # Prepare datasets
        train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # For quick testing
        eval_dataset = tokenized_datasets['test' if 'test' in tokenized_datasets else 'validation'].shuffle(seed=42).select(range(500))

        # Training arguments
        hyperparameters = parameters.get('hyperparameters', {})
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=float(hyperparameters.get('epochs', 1)),
            per_device_train_batch_size=int(hyperparameters.get('batch_size', 8)),
            per_device_eval_batch_size=int(hyperparameters.get('batch_size', 8)),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
        )

        # Compute metrics
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            acc = accuracy_score(labels, predictions)
            return {'accuracy': acc}

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        # Training
        logging.info("Starting training...")
        trainer.train()

        # Evaluation
        logging.info("Evaluating the model...")
        eval_results = trainer.evaluate()

        logging.info(f"Experiment Results: {eval_results}")
        return eval_results

    except Exception as e:
        logging.error(f"Error in experiment execution: {e}")
        return None

def load_dataset_with_retry(dataset_name, use_auth_token, config=None):
    try:
        kwargs = {"download_mode": "force_redownload"}
        if config:
            kwargs["name"] = config
        
        # Try loading without auth token first
        try:
            return load_dataset(dataset_name, **kwargs)
        except Exception as e:
            if "use_auth_token" in str(e):
                # If the error is about use_auth_token, try again with the token
                kwargs["use_auth_token"] = use_auth_token
                return load_dataset(dataset_name, **kwargs)
            else:
                # If it's a different error, raise it
                raise
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        return None
