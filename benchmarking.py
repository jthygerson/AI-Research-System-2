# benchmarking.py

import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np
import torch

def benchmark_system():
    try:
        # Use a standard benchmark dataset (e.g., SST-2 from GLUE)
        raw_datasets = load_dataset('glue', 'sst2')
        logging.info("Loaded benchmark dataset: GLUE SST-2")

        # Load pre-trained model and tokenizer
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Tokenize data
        def tokenize_function(example):
            return tokenizer(example['sentence'], padding="max_length", truncation=True)

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        eval_dataset = tokenized_datasets['validation']

        # Prepare dataset
        eval_dataset = eval_dataset.remove_columns(['sentence', 'idx'])
        eval_dataset.set_format('torch')

        # Evaluation
        training_args = TrainingArguments(
            output_dir='./benchmark_results',
            per_device_eval_batch_size=8,
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = (predictions == labels).mean()
            return {'accuracy': accuracy}

        trainer = Trainer(
            model=model,
            args=training_args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        eval_results = trainer.evaluate()
        logging.info(f"Benchmark Results: {eval_results}")

        return eval_results

    except Exception as e:
        logging.error(f"Error during benchmarking: {e}")
        return None
