# experiment_execution.py

import os

# Set cache directories
os.environ['HF_DATASETS_CACHE'] = '/tmp/huggingface_datasets_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_transformers_cache'

import logging
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

def execute_experiment(parameters):
    try:
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        # Load dataset
        dataset_name = parameters.get('datasets', ['ag_news'])[0]
        raw_datasets = load_dataset(dataset_name, cache_dir=None)  # Disable caching
        logging.info(f"Loaded dataset: {dataset_name}")

        # Tokenizer and model
        model_name = parameters.get('model_architecture', 'distilbert-base-uncased')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        num_labels = len(set(raw_datasets['train']['label']))
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        model.to(device)

        # Tokenize datasets
        def tokenize_function(example):
            return tokenizer(example['text'], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

        # Prepare datasets
        train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(1000))  # For quick testing
        eval_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(500))

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
