from transformers import Trainer, TrainingArguments, AdamW

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Increased number of epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=3e-5,  # Increased learning rate
)

# Using a different optimizer
optimizer = AdamW(model.parameters(), lr=training_args.learning_rate)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, None),  # Set the optimizer
)

trainer.train()